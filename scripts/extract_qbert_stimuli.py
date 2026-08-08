"""Extract Q*bert stimuli from four source networks.

53 stimuli total:
  - 1 level-start frame per level, levels 1–5  (5 frames)
  - 12 intermediate frames per level, levels 1–4 (3 per network × 4 networks = 48 frames)

Every stimulus comes from a DIFFERENT episode that reached level 5.
Frames from episodes that did not reach level 5 are never used.

Source networks:
  run_model_r1  — legacy BatchNorm2d in ResidualBlock (trained before GroupNorm fix)
  run_0000_r0   — GroupNorm, no attention, no residual, depth=2
  run_0003_r0   — GroupNorm, no attention, residual, depth=2
  run_0004_r0   — GroupNorm, no attention, no residual, depth=1

Usage:
    python scripts/extract_qbert_stimuli.py [--dry-run] [--max-episodes N]

Output:
    output/production/qbert/stimuli.npz
      inputs       : (53, 4, 84, 84) float32
      levels       : (53,) int32
      frame_types  : (53,) str  — 'start' or 'intermediate'
      source_nets  : (53,) str
    output/production/qbert/stimuli_images/
      overview_starts.png       — 1×5 grid of level starts
      overview_intermediate.png — 4×12 grid of intermediates (rows=levels, cols=frames)
      individual PNGs
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

INTERMEDIATE_LEVELS  = (1, 2, 3, 4)
START_LEVELS         = (1, 2, 3, 4, 5)
INTERMEDIATE_PER_NET = 3   # per level

_NETS = ["run_model_r1", "run_0000_r0", "run_0003_r0", "run_0004_r0"]

SOURCE_NETWORKS = [
    {
        "name":              "run_model_r1",
        "weights":           OUTPUT_DIR / "run_model_r1" / "best_weights.pt",
        "config":            dict(use_batch_norm=True, use_attention=True, use_residual=True,
                                  hidden_size=512, depth=1),
        "legacy_batch_norm": True,
        "n_episodes_needed": 14,  # start levels 1+5 + 4 levels × 3 intermediate
    },
    {
        "name":              "run_0000_r0",
        "weights":           OUTPUT_DIR / "run_0000_r0" / "best_weights.pt",
        "config":            dict(use_batch_norm=True, use_attention=False, use_residual=False,
                                  hidden_size=406, depth=2),
        "legacy_batch_norm": False,
        "n_episodes_needed": 13,  # start level 2 + 4 levels × 3 intermediate
    },
    {
        "name":              "run_0003_r0",
        "weights":           OUTPUT_DIR / "run_0003_r0" / "best_weights.pt",
        "config":            dict(use_batch_norm=True, use_attention=False, use_residual=True,
                                  hidden_size=742, depth=2),
        "legacy_batch_norm": False,
        "n_episodes_needed": 13,  # start level 3 + 4 levels × 3 intermediate
    },
    {
        "name":              "run_0004_r0",
        "weights":           OUTPUT_DIR / "run_0004_r0" / "best_weights.pt",
        "config":            dict(use_batch_norm=True, use_attention=False, use_residual=False,
                                  hidden_size=634, depth=1),
        "legacy_batch_norm": False,
        "n_episodes_needed": 13,  # start level 4 + 4 levels × 3 intermediate
    },
]

# Which network provides the level-start frame for each level.
START_ASSIGNMENTS = {1: "run_model_r1", 2: "run_0000_r0", 3: "run_0003_r0",
                     4: "run_0004_r0",  5: "run_model_r1"}

ALL_LEVELS = set(START_LEVELS) | set(INTERMEDIATE_LEVELS)  # 1–5


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


def load_network(env, config, weights_path, legacy_batch_norm=False):
    net = ImprovedA2CNetwork(env.observation_space.shape, env.action_space.n,
                             legacy_batch_norm=legacy_batch_norm, **config)
    net.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    net.eval()
    return net


def collect_successful_episodes(net, env, ale, n_needed, max_episodes):
    """Run episodes, keeping only those that reach level 5.

    Returns a list of dicts {level: [frames]} for each successful episode,
    in the order collected. Stops once n_needed successful episodes are found
    or max_episodes total episodes have been run.
    """
    successful = []
    total_eps  = 0

    while len(successful) < n_needed and total_eps < max_episodes:
        obs, _ = env.reset()
        done = False
        ep_frames = {L: [] for L in ALL_LEVELS}

        while not done:
            obs_array = np.array(obs, dtype=np.float32)
            level = int(ale.getRAM()[QBERT_LEVEL_RAM_ADDR]) + 1
            if level in ALL_LEVELS:
                ep_frames[level].append(obs_array)
            with torch.no_grad():
                t = torch.tensor(obs_array).unsqueeze(0)
                policy, *_ = net(t)
            action = Categorical(policy).sample().item()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        total_eps += 1
        reached_5 = len(ep_frames[5]) >= 1
        if reached_5:
            successful.append(ep_frames)

        print(f"    ep {total_eps:3d}: {'SUCCESS' if reached_5 else 'fail   '}  "
              f"collected {len(successful)}/{n_needed}", flush=True)

    if len(successful) < n_needed:
        print(f"  WARNING: only {len(successful)}/{n_needed} successful episodes "
              f"after {total_eps} attempts")

    return successful


def pick_random_interior_frame(frames, rng):
    """Pick a uniformly random frame from the interior (excluding first and last).

    Returns (frame, progress) where progress is the fraction through the level [0, 1].
    """
    interior = frames[1:-1]
    assert len(interior) >= 1, "Level segment too short for interior frame"
    idx = int(rng.integers(len(interior)))
    progress = idx / (len(interior) - 1) if len(interior) > 1 else 0.5
    return interior[idx], progress


_NET_SUFFIX = {
    "run_model_r1": "r1",
    "run_0000_r0":  "0000",
    "run_0003_r0":  "0003",
    "run_0004_r0":  "0004",
}


def save_images(inputs_arr, levels_arr, types_arr, nets_arr, out_dir):
    img_dir = out_dir / "stimuli_images"
    img_dir.mkdir(exist_ok=True)

    SCALE   = 3
    LABEL_H = 14
    CELL_W  = 84 * SCALE
    CELL_H  = 84 * SCALE + LABEL_H

    _type_label = {"start": "S", "intermediate": "I"}

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 10)
    except Exception:
        font = ImageFont.load_default()

    def frame_to_img(arr):
        px = (arr[3] * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(px, mode="L").resize((CELL_W, 84 * SCALE), resample=Image.NEAREST)

    def label_for(idx):
        return f"{_type_label[types_arr[idx]]}{_NET_SUFFIX[nets_arr[idx]]}"

    for i, (inp, lvl, ft, net) in enumerate(zip(inputs_arr, levels_arr, types_arr, nets_arr)):
        frame_to_img(inp).save(img_dir / f"{i:02d}_lvl{lvl}_{ft}_{net}.png")

    # Starts: 1 row × 5 cols
    start_idxs = sorted([i for i, t in enumerate(types_arr) if t == "start"],
                        key=lambda i: levels_arr[i])
    grid_s = Image.new("L", (len(start_idxs) * CELL_W, CELL_H), color=40)
    draw_s = ImageDraw.Draw(grid_s)
    for col, idx in enumerate(start_idxs):
        x = col * CELL_W
        grid_s.paste(frame_to_img(inputs_arr[idx]), (x, 0))
        draw_s.text((x + 2, 84 * SCALE + 1),
                    f"L{levels_arr[idx]} {label_for(idx)}", fill=200, font=font)
    grid_s.save(img_dir / "overview_starts.png")

    # Intermediates: 4 rows × 12 cols, sorted chronologically within each level
    inter_by_level = {L: [] for L in INTERMEDIATE_LEVELS}
    for i, (t, lvl) in enumerate(zip(types_arr, levels_arr)):
        if t == "intermediate":
            inter_by_level[int(lvl)].append(i)  # already sorted by progress

    cols = len(_NETS) * INTERMEDIATE_PER_NET
    grid_i = Image.new("L", (cols * CELL_W, len(INTERMEDIATE_LEVELS) * CELL_H), color=40)
    draw_i = ImageDraw.Draw(grid_i)
    for row, lvl in enumerate(INTERMEDIATE_LEVELS):
        for col, idx in enumerate(inter_by_level[lvl]):
            x, y = col * CELL_W, row * CELL_H
            grid_i.paste(frame_to_img(inputs_arr[idx]), (x, y))
            draw_i.text((x + 2, y + 84 * SCALE + 1), label_for(idx), fill=200, font=font)
    grid_i.save(img_dir / "overview_intermediate.png")

    print(f"Images saved to {img_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-episodes", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if STIMULI_PATH.exists() and not args.dry_run:
        print(f"ERROR: {STIMULI_PATH} already exists. Delete it first to regenerate.")
        sys.exit(1)

    # Collect successful episodes per network
    successful_eps = {}
    for source in SOURCE_NETWORKS:
        name = source["name"]
        n    = source["n_episodes_needed"]
        print(f"\n[{name}] Collecting {n} successful episodes (max {args.max_episodes} attempts)...")
        env = make_env()
        ale = env.unwrapped.ale
        net = load_network(env, source["config"], source["weights"],
                           source["legacy_batch_norm"])
        successful_eps[name] = collect_successful_episodes(net, env, ale, n, args.max_episodes)
        env.close()
        print(f"  → {len(successful_eps[name])} episodes collected")

    if args.dry_run:
        print("\nDry run complete.")
        return

    all_inputs, all_levels, all_types, all_nets, all_progress = [], [], [], [], []

    ep_idx = {name: 0 for name in _NETS}
    rng = np.random.default_rng(seed=42)

    print("\nLevel starts:")
    for L in START_LEVELS:
        net_name = START_ASSIGNMENTS[L]
        eps = successful_eps[net_name]
        assert ep_idx[net_name] < len(eps), \
            f"Ran out of successful episodes for {net_name} (level {L} start)"
        frames = eps[ep_idx[net_name]][L]
        assert len(frames) >= 1, f"No frames at level {L} in episode for {net_name}"
        all_inputs.append(frames[0])
        all_levels.append(L); all_types.append("start"); all_nets.append(net_name)
        all_progress.append(0.0)
        ep_idx[net_name] += 1
        print(f"  level {L}: {net_name}  (ep #{ep_idx[net_name]})")

    print("\nIntermediates:")
    for L in INTERMEDIATE_LEVELS:
        print(f"  level {L}:")
        for net_name in _NETS:
            eps = successful_eps[net_name]
            for _ in range(INTERMEDIATE_PER_NET):
                assert ep_idx[net_name] < len(eps), \
                    f"Ran out of successful episodes for {net_name} (level {L} intermediate)"
                frames = eps[ep_idx[net_name]][L]
                assert len(frames) >= 3, \
                    f"Too few frames at level {L} in episode for {net_name}"
                frame, progress = pick_random_interior_frame(frames, rng)
                all_inputs.append(frame)
                all_levels.append(L); all_types.append("intermediate"); all_nets.append(net_name)
                all_progress.append(progress)
                ep_idx[net_name] += 1
            print(f"    {net_name} × {INTERMEDIATE_PER_NET}  (eps #{ep_idx[net_name]-INTERMEDIATE_PER_NET+1}–{ep_idx[net_name]})")

    # Sort by (level, progress) for chronological order in RDMs
    sort_key = np.array(all_levels) + np.array(all_progress)
    order = np.argsort(sort_key, kind="stable")

    inputs_arr   = np.stack(all_inputs,    axis=0)[order].astype(np.float32)
    levels_arr   = np.array(all_levels,    dtype=np.int32)[order]
    types_arr    = np.array(all_types)[order]
    nets_arr     = np.array(all_nets)[order]
    progress_arr = np.array(all_progress,  dtype=np.float32)[order]

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
        progress    = progress_arr,
    )
    print(f"\nSaved to {STIMULI_PATH}")

    save_images(inputs_arr, levels_arr, types_arr, nets_arr, OUTPUT_DIR)


if __name__ == "__main__":
    main()
