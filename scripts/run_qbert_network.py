"""Train a single Q*bert A2C network from the manual hyperparameter grid.

Usage:
    # Train the model network (for stimulus extraction, not included in analysis):
    python scripts/run_qbert_network.py --model

    # Train analysis network #N (0-indexed):
    python scripts/run_qbert_network.py --run-index 0

    # Dry-run: print config without training:
    python scripts/run_qbert_network.py --run-index 0 --dry-run

    # Override max steps (quick test):
    python scripts/run_qbert_network.py --run-index 0 --max-steps 100000

After all analysis networks are trained, update bo_state.json manually or
run this script with --update-bo-state to append all completed metadata.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "output" / "production" / "qbert"
STIMULI_PATH = OUTPUT_DIR / "stimuli.npz"
BO_STATE_PATH = OUTPUT_DIR / "bo_state.json"

# ─── Hyperparameter grid ──────────────────────────────────────────────────────
#
# 32 analysis networks: 7D Sobol, 32 draws. use_batch_norm fixed to True.
# All hyperparameters sampled jointly. Continuous dims mapped via log/linear
# scale; boolean dims thresholded at 0.5.
#
# Dimensions:
#   0  learning_rate  : [1e-4, 1e-3]  log scale
#   1  entropy_coef   : [5e-3, 1e-1]  log scale
#   2  gamma          : [0.98, 0.995] linear
#   3  hidden_size    : [256, 768]     log scale
#   4  use_attention  : >= 0.5 -> True
#   5  use_residual   : >= 0.5 -> True
#   6  depth          : >= 0.5 -> 2, else 1

_LR_LO,     _LR_HI     = 1e-4, 1e-3
_ENT_LO,    _ENT_HI    = 5e-3, 1e-1
_GAMMA_LO,  _GAMMA_HI  = 0.98, 0.995
_HIDDEN_LO, _HIDDEN_HI = 256,  768

_engine = torch.quasirandom.SobolEngine(dimension=7, scramble=True, seed=42)
_pts    = _engine.draw(32).numpy()

def _logmap(lo, hi, u): return float(f"{np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))):.6g}")
def _linmap(lo, hi, u): return float(f"{lo + u * (hi - lo):.4f}")

def _make_config(u):
    return {
        "learning_rate":  _logmap(_LR_LO,     _LR_HI,     u[0]),
        "entropy_coef":   _logmap(_ENT_LO,    _ENT_HI,    u[1]),
        "gamma":          _linmap(_GAMMA_LO,  _GAMMA_HI,  u[2]),
        "hidden_size":    int(round(_logmap(_HIDDEN_LO, _HIDDEN_HI, u[3]))),
        "use_batch_norm": True,
        "use_attention":  bool(u[4] >= 0.5),
        "use_residual":   bool(u[5] >= 0.5),
        "depth":          2 if u[6] >= 0.5 else 1,
    }

ANALYSIS_CONFIGS = [_make_config(u) for u in _pts]

# Model network — trained first, used only for stimulus extraction
MODEL_CONFIG = {
    "learning_rate": 0.0003,
    "entropy_coef":  0.01,
    "gamma":         0.99,
    "hidden_size":   512,
    "depth":         1,
    "use_batch_norm": True,
    "use_attention":  True,
    "use_residual":   True,
}


def load_stimuli():
    import numpy as np
    assert STIMULI_PATH.exists(), (
        f"Stimuli not found: {STIMULI_PATH}\n"
        "Run: python scripts/extract_qbert_stimuli.py --model-run-dir <run_dir>"
    )
    return np.load(STIMULI_PATH)["inputs"]


def update_bo_state(run_dir, run_index):
    """Append this network's result to bo_state.json after training."""
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        print(f"  [bo_state] metadata.json not found in {run_dir}, skipping.")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    observations = []
    if BO_STATE_PATH.exists():
        with open(BO_STATE_PATH) as f:
            observations = json.load(f)

    # Avoid duplicates
    existing_iters = {o["iteration"] for o in observations}
    if run_index in existing_iters:
        print(f"  [bo_state] iteration {run_index} already in bo_state.json, skipping.")
        return

    observations.append({
        "iteration":   run_index,
        "performance": meta["best_metric"],   # raw mean episode score
        "config":      meta["config"],
    })
    observations.sort(key=lambda o: o["iteration"])

    BO_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BO_STATE_PATH, "w") as f:
        json.dump(observations, f, indent=2)
    print(f"  [bo_state] updated {BO_STATE_PATH}  (now {len(observations)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Train one Q*bert analysis network.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-index", type=int,
                       help=f"Analysis network index (0–{len(ANALYSIS_CONFIGS)-1})")
    group.add_argument("--model", action="store_true",
                       help="Train the model network (for stimulus extraction, run first)")
    group.add_argument("--list", action="store_true",
                       help="Print all configs and exit")

    parser.add_argument("--repeat",    type=int, default=0,
                        help="Repeat index (0 = primary run, 1+ = re-runs for variability estimation)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing into a run directory that already has a metadata.json")
    parser.add_argument("--dry-run",   action="store_true", help="Print config and exit")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override total_steps (for smoke tests)")
    parser.add_argument("--n-envs",    type=int, default=16)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: output/production/qbert/)")
    args = parser.parse_args()

    if args.list:
        print(f"Model network config:")
        print(f"  run_model_r0  →  {MODEL_CONFIG}")
        print(f"\nAnalysis network configs ({len(ANALYSIS_CONFIGS)} total):")
        for i, cfg in enumerate(ANALYSIS_CONFIGS):
            print(f"  #{i:02d}  {cfg}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model:
        config    = MODEL_CONFIG
        run_dir   = output_dir / f"run_model_r{args.repeat}"
        run_index = None
        label     = f"model network (repeat {args.repeat})"
    else:
        assert 0 <= args.run_index < len(ANALYSIS_CONFIGS), (
            f"--run-index must be 0–{len(ANALYSIS_CONFIGS)-1}"
        )
        config    = ANALYSIS_CONFIGS[args.run_index]
        run_dir   = output_dir / f"run_{args.run_index:04d}_r{args.repeat}"
        run_index = args.run_index
        label     = f"analysis network #{args.run_index} (repeat {args.repeat})"

    if (run_dir / "metadata.json").exists() and not args.overwrite:
        print(f"ERROR: {run_dir} already has a metadata.json.")
        print("       Use --overwrite to write into this directory anyway.")
        sys.exit(1)

    print(f"Q*bert {label}")
    print(f"  run_dir : {run_dir}")
    print(f"  config  : {config}")
    if args.max_steps:
        print(f"  max_steps (override): {args.max_steps:,}")

    if args.dry_run:
        print("  [dry-run] exiting.")
        return

    # Load stimuli (not needed for model network, but check early for analysis runs)
    if not args.model:
        stimuli = load_stimuli()
        print(f"  stimuli : {stimuli.shape}  ({STIMULI_PATH})")
    else:
        # Model network: create a dummy stimuli placeholder so train_network works.
        # Actual stimuli are extracted from this network's gameplay afterwards.
        import numpy as np
        stimuli = np.zeros((53, 4, 84, 84), dtype=np.float32)
        print("  stimuli : dummy placeholder (extract real stimuli after training)")

    from src.qbert.train import train_network

    total_steps = args.max_steps if args.max_steps else 60_000_000
    best_score = train_network(
        config      = config,
        run_dir     = run_dir,
        stimuli_array = stimuli,
        total_steps = total_steps,
        n_envs      = args.n_envs,
        verbose     = True,
    )

    print(f"\nFinished: best_score = {best_score:.1f}")

    if not args.model and run_index is not None:
        update_bo_state(run_dir, run_index)


if __name__ == "__main__":
    main()
