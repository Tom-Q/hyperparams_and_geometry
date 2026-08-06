"""Probe ALE RAM to identify the Q*bert level register.

Runs the trained model (run_model_test1/best_weights.pt) for several episodes,
detects level transitions via flash detection, and reports which RAM bytes
change at each transition. Run once; the result goes into train.py.

Usage:
    python scripts/probe_qbert_ram.py [--n-episodes N]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path("/home/thomas/atari_acc")))

from ai_acc.image_analysis import level_flash_detection, SCOREBOX_MASK  # noqa: E402

MODEL_DIR     = REPO_ROOT / "output" / "production" / "qbert" / "run_model_test1"
WEIGHTS_PATH  = MODEL_DIR / "best_weights.pt"
FLASH_GAP_MAX = 8   # frames of non-flash allowed inside a sequence


def make_eval_env():
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


def load_model(env):
    from src.qbert.train import ImprovedA2CNetwork
    net = ImprovedA2CNetwork(
        env.single_observation_space.shape if hasattr(env, 'single_observation_space')
        else env.observation_space.shape,
        env.action_space.n,
        use_batch_norm=True,
        use_attention=True,
        use_residual=True,
        hidden_size=512,
        depth=1,
    )
    net.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    net.eval()
    return net


def get_action(net, obs):
    t = torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        policy, *_ = net(t)
    return Categorical(policy).sample().item()


def find_flash_sequence_ends(flash_flags):
    """Return indices where a flash sequence just ended."""
    ends = []
    in_seq = False
    gap = 0
    last_flash = -1
    for i, f in enumerate(flash_flags):
        if f:
            in_seq = True
            gap = 0
            last_flash = i
        elif in_seq:
            gap += 1
            if gap > FLASH_GAP_MAX:
                ends.append(last_flash)
                in_seq = False
                gap = 0
    if in_seq:
        ends.append(last_flash)
    return ends


def run_episode(net, env):
    """Run one episode; return list of (ram, is_flash) per step."""
    ale = env.unwrapped.ale
    obs, _ = env.reset()
    prev_screen = ale.getScreenRGB()
    records = []
    done = False
    while not done:
        ram        = ale.getRAM().copy()
        curr_screen = ale.getScreenRGB()
        is_flash   = level_flash_detection(prev_screen, curr_screen)
        records.append((ram, is_flash))
        action     = get_action(net, obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        prev_screen = curr_screen
    return records


def analyze_episode(records):
    """Return list of (transition_idx, ram_before, ram_after) at each level transition."""
    flash_flags = [r[1] for r in records]
    ends = find_flash_sequence_ends(flash_flags)
    transitions = []
    for end_idx in ends:
        # RAM just before the flash sequence started
        seq_start = end_idx
        while seq_start > 0 and flash_flags[seq_start - 1]:
            seq_start -= 1
        before_idx = max(0, seq_start - 1)
        after_idx  = min(len(records) - 1, end_idx + 1)
        ram_before = records[before_idx][0]
        ram_after  = records[after_idx][0]
        transitions.append((end_idx, ram_before, ram_after))
    return transitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=5)
    args = parser.parse_args()

    env = make_eval_env()
    net = load_model(env)
    print(f"Model loaded from {WEIGHTS_PATH}")
    print(f"Running {args.n_episodes} episodes...\n")

    # Accumulate votes: for each RAM address, count how many transitions it changed
    change_counts = np.zeros(128, dtype=int)
    total_transitions = 0

    for ep in range(args.n_episodes):
        records = run_episode(net, env)
        transitions = analyze_episode(records)
        total_transitions += len(transitions)
        print(f"  Episode {ep+1}: {len(records)} steps, {len(transitions)} level transition(s)")
        for t_idx, ram_before, ram_after in transitions:
            changed = np.where(ram_before != ram_after)[0]
            change_counts[changed] += 1
            print(f"    transition at step {t_idx}: RAM bytes changed: "
                  + ", ".join(f"{a}={ram_before[a]}->{ram_after[a]}" for a in changed))

    env.close()

    print(f"\nTotal transitions detected: {total_transitions}")
    print("\nRAM addresses that changed at transitions (sorted by frequency):")
    candidates = [(i, change_counts[i]) for i in range(128) if change_counts[i] > 0]
    candidates.sort(key=lambda x: -x[1])
    for addr, count in candidates:
        print(f"  addr {addr:3d} (0x{addr:02x}):  changed in {count}/{total_transitions} transitions")


if __name__ == "__main__":
    main()
