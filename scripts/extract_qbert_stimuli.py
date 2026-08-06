"""Extract Q*bert stimuli from the three source model networks.

48 stimuli total: 4 levels × 12 frames per level.
  - 1 start per level  (first frame when that level begins)
  - 1 end per level    (last frame before level transitions out)
  - 10 intermediate per level, split ~3-4 per source network

Starts and ends are each assigned to one network per level, rotating
so different levels come from different networks. Intermediates are
spread across all three networks.

Level detection uses ALE RAM address 99 (cumulative levels-completed count).

Usage:
    python scripts/extract_qbert_stimuli.py [--dry-run] [--max-episodes N]

Output:
    output/production/qbert/stimuli.npz
      inputs       : (48, 4, 84, 84) float32
      levels       : (48,) int32
      frame_types  : (48,) str     — 'start', 'end', 'intermediate'
      source_nets  : (48,) str
    output/production/qbert/stimuli_images/
      overview.png                      — 4×12 grid (rows=levels, cols=stimuli per level)
      00_lvl1_start_run_model_r1.png    — individual stimuli (most recent frame, channel 3)
      ...
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.distributions import Categorical

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.qbert.train import ImprovedA2CNetwork, QBERT_LEVEL_RAM_ADDR

OUTPUT_DIR   = REPO_ROOT / "output" / "production" / "qbert"
STIMULI_PATH = OUTPUT_DIR / "stimuli.npz"

TARGET_LEVELS = (1, 2, 3, 4)

_NETS = ["run_model_test1", "run_model_r0", "run_model_r1"]

# Per-level assignments: which network provides start, end, and how many
# intermediate frames each network contributes (must sum to 10).
LEVEL_ASSIGNMENTS = {
    1: {"start": "run_model_r1",    "end": "run_model_r0",
        "intermediate": {"run_model_test1": 3, "run_model_r0": 3, "run_model_r1": 4}},
    2: {"start": "run_model_r0",    "end": "run_model_test1",
        "intermediate": {"run_model_test1": 3, "run_model_r0": 4, "run_model_r1": 3}},
    3: {"start": "run_model_test1", "end": "run_model_r1",
        "intermediate": {"run_model_test1": 4, "run_model_r0": 3, "run_model_r1": 3}},
    4: {"start": "run_model_r1",    "end": "run_model_test1",
        "intermediate": {"run_model_test1": 3, "run_model_r0": 4, "run_model_r1": 3}},
}

SOURCE_NETWORKS = [
    {
        "name":   "run_model_test1",
        "weights": OUTPUT_DIR / "run_model_test1" / "best_weights.pt",
        "config": dict(use_batch_norm=True, use_attention=True, use_residual=True,
                       hidden_size=512, depth=1),
    },
    {
        "name":   "run_model_r0",
        "weights": OUTPUT_DIR / "run_model_r0" / "best_weights.pt",
        "config": dict(use_batch_norm=True, use_attention=True, use_residual=True,
                       hidden_size=512, depth=1),
    },
    {
        "name":   "run_model_r1",
        "weights": OUTPUT_DIR / "run_model_r1" / "best_weights.pt",
        "config": dict(use_batch_norm=True, use_attention=True, use_residual=True,
                       hidden_size=512, depth=1),
    },
]


def make_env():
    import ale_py
    import gymnasium as gym
    from gymnasium.wrappers import FrameStackObservation
    from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
    gym.register_envs(ale_py)
    env = gym.make("ALE/Qbert-v5", render_mode=None)
    env = AtariPreprocessing(env, noop_max=30, frame_skip=1, screen_size=84,
                             terminal_on_life_loss=False, grayscale_obs=True,
                             grayscale_newaxis=False, scale_obs=True)
    return FrameStackObservation(env, 4)


def load_network(env, config, weights_path):
    net = ImprovedA2CNetwork(env.observation_space.shape, env.action_space.n, **config)
    sd = torch.load(weights_path, map_location="cpu")
    result = net.load_state_dict(sd, strict=False)
    if result.missing_keys:
        raise RuntimeError(f"Missing keys: {result.missing_keys}")
    net.eval()
    return net


def collect_level_pools(net, env, ale, max_episodes):
    """Run episodes and collect per-level frame sequences.

    Returns: dict level -> list of frame-lists (one per episode that reached it).
    Each frame is a (4, 84, 84) float32 array.
    """
    level_eps = {L: [] for L in TARGET_LEVELS}

    for ep in range(max_episodes):
        obs, _ = env.reset()
        done = False
        ep_frames = {L: [] for L in TARGET_LEVELS}

        while not done:
            obs_array = np.array(obs, dtype=np.float32)
            level = int(ale.getRAM()[QBERT_LEVEL_RAM_ADDR]) + 1
            if level in TARGET_LEVELS:
                ep_frames[level].append(obs_array)
            with torch.no_grad():
                t = torch.tensor(obs_array).unsqueeze(0)
                policy, *_ = net(t)
            action = Categorical(policy).sample().item()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        levels_reached = [L for L in TARGET_LEVELS if len(ep_frames[L]) >= 3]
        for L in levels_reached:
            level_eps[L].append(ep_frames[L])

        print(f"    ep {ep+1:2d}: reached {levels_reached}  "
              f"pool sizes {[len(level_eps[L]) for L in TARGET_LEVELS]}", flush=True)

        if all(level_eps[L] for L in TARGET_LEVELS):
            break

    return level_eps


def pick_episode(level_eps):
    """Return the longest episode frame-list for a given level pool."""
    return max(level_eps, key=len)


def sample_intermediates(frames, n):
    """Sample n evenly-spaced frames from the interior (excluding first and last)."""
    interior = frames[1:-1]
    assert len(interior) >= n, f"Not enough interior frames ({len(interior)}) for {n} samples"
    indices = np.linspace(0, len(interior) - 1, n, dtype=int)
    return [interior[i] for i in indices]


def save_images(inputs_arr, levels_arr, types_arr, nets_arr, out_dir):
    """Save individual PNGs and a 4×12 overview grid."""
    img_dir = out_dir / "stimuli_images"
    img_dir.mkdir(exist_ok=True)

    SCALE   = 3          # upscale 84→252 px for legibility
    LABEL_H = 14         # pixels for text label below each cell
    CELL_W  = 84 * SCALE
    CELL_H  = 84 * SCALE + LABEL_H
    COLS    = 12         # stimuli per level
    ROWS    = 4          # levels

    # Short label: type initial + net suffix
    _net_suffix = {"run_model_test1": "t1", "run_model_r0": "r0", "run_model_r1": "r1"}
    _type_label = {"start": "S", "end": "E", "intermediate": "I"}

    def frame_to_img(arr):
        frame = arr[3]  # most recent channel
        px = (frame * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(px, mode="L").resize(
            (CELL_W, 84 * SCALE), resample=Image.NEAREST
        )

    # Individual images
    for i, (inp, lvl, ft, net) in enumerate(zip(inputs_arr, levels_arr, types_arr, nets_arr)):
        name = f"{i:02d}_lvl{lvl}_{ft}_{net}.png"
        frame_to_img(inp).save(img_dir / name)

    # Overview grid: rows = levels (1-4), within each row order is start, end, inter×10
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 10)
    except Exception:
        font = ImageFont.load_default()

    grid = Image.new("L", (COLS * CELL_W, ROWS * CELL_H), color=40)
    draw = ImageDraw.Draw(grid)

    # Group by level, preserving order within each level
    level_indices = {L: [] for L in range(1, 5)}
    for i, lvl in enumerate(levels_arr):
        level_indices[int(lvl)].append(i)

    for row, lvl in enumerate(range(1, 5)):
        idxs = level_indices[lvl]
        assert len(idxs) == COLS, f"Expected {COLS} stimuli for level {lvl}, got {len(idxs)}"
        for col, idx in enumerate(idxs):
            x, y = col * CELL_W, row * CELL_H
            grid.paste(frame_to_img(inputs_arr[idx]), (x, y))
            label = f"{_type_label[types_arr[idx]]}{_net_suffix[nets_arr[idx]]}"
            draw.text((x + 2, y + 84 * SCALE + 1), label, fill=200, font=font)

    grid.save(img_dir / "overview.png")
    print(f"Images saved to {img_dir}/  ({len(inputs_arr)} individual + overview.png)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-episodes", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if STIMULI_PATH.exists() and not args.dry_run:
        print(f"ERROR: {STIMULI_PATH} already exists. Delete it first to regenerate.")
        sys.exit(1)

    # Collect per-network, per-level frame pools
    pools = {}
    for source in SOURCE_NETWORKS:
        name = source["name"]
        print(f"\n[{name}] Running up to {args.max_episodes} episodes...")
        env = make_env()
        ale = env.unwrapped.ale
        net = load_network(env, source["config"], source["weights"])
        pools[name] = collect_level_pools(net, env, ale, args.max_episodes)
        env.close()
        for L in TARGET_LEVELS:
            n = len(pools[name][L])
            print(f"  level {L}: {n} episode(s)" if n else f"  level {L}: NOT REACHED")

    if args.dry_run:
        print("\nDry run complete.")
        return

    # Assemble stimuli according to LEVEL_ASSIGNMENTS
    all_inputs, all_levels, all_types, all_nets = [], [], [], []

    for L in TARGET_LEVELS:
        assign = LEVEL_ASSIGNMENTS[L]
        print(f"\nLevel {L}:")

        # Start frame
        start_net = assign["start"]
        if not pools[start_net][L]:
            # Fall back to any network that has data
            start_net = next(n for n in _NETS if pools[n][L])
            print(f"  WARNING: start fallback to {start_net}")
        frames = pick_episode(pools[start_net][L])
        all_inputs.append(frames[0])
        all_levels.append(L); all_types.append("start"); all_nets.append(start_net)
        print(f"  start  : {start_net}")

        # End frame
        end_net = assign["end"]
        if not pools[end_net][L]:
            end_net = next(n for n in _NETS if pools[n][L])
            print(f"  WARNING: end fallback to {end_net}")
        frames = pick_episode(pools[end_net][L])
        all_inputs.append(frames[-1])
        all_levels.append(L); all_types.append("end"); all_nets.append(end_net)
        print(f"  end    : {end_net}")

        # Intermediate frames
        for net_name, n_inter in assign["intermediate"].items():
            if n_inter == 0:
                continue
            src = net_name
            if not pools[src][L]:
                src = next(n for n in _NETS if pools[n][L])
                print(f"  WARNING: intermediate fallback {net_name} -> {src}")
            frames = pick_episode(pools[src][L])
            samples = sample_intermediates(frames, n_inter)
            for s in samples:
                all_inputs.append(s)
                all_levels.append(L); all_types.append("intermediate"); all_nets.append(src)
            print(f"  inter  : {src} × {n_inter}")

    inputs_arr = np.stack(all_inputs, axis=0).astype(np.float32)
    levels_arr = np.array(all_levels, dtype=np.int32)
    types_arr  = np.array(all_types)
    nets_arr   = np.array(all_nets)

    print(f"\nTotal: {len(inputs_arr)} stimuli, shape {inputs_arr.shape}")
    print(f"  Levels:  {dict(zip(*np.unique(levels_arr, return_counts=True)))}")
    print(f"  Types:   {dict(zip(*np.unique(types_arr,  return_counts=True)))}")
    print(f"  Sources: {dict(zip(*np.unique(nets_arr,   return_counts=True)))}")

    np.savez_compressed(
        str(STIMULI_PATH),
        inputs      = inputs_arr,
        levels      = levels_arr,
        frame_types = types_arr,
        source_nets = nets_arr,
    )
    print(f"\nSaved to {STIMULI_PATH}")

    save_images(inputs_arr, levels_arr, types_arr, nets_arr, OUTPUT_DIR)


if __name__ == "__main__":
    main()
