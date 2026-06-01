"""
Quick GP-phase smoke test using the first 100 Sobol observations as seed.

Takes the first 100 primary observations from experiments_saturating_test/spirals,
which is exactly N_SOBOL, so the run enters GP phase immediately.
Trains each network for max 5 epochs so one iteration takes seconds.

Usage:
    python run_gp_test.py [--n-iter 5] [--h 0.15] [--beta 4.0]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SRC_STATE   = Path("output/experiments_saturating_test/spirals/bo_state.json")
TEST_OUTPUT = Path("output/experiments_gp_test")
N_SEED      = 100  # exactly N_SOBOL primaries to seed with


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-iter",     type=int,   default=5)
    p.add_argument("--h",          type=float, default=0.15)
    p.add_argument("--beta",       type=float, default=4.0)
    p.add_argument("--max-epochs", type=int,   default=None,
                   help="Override max epochs (omit for full training)")
    return p.parse_args()


def seed_test_dir():
    if not SRC_STATE.exists():
        sys.exit(f"Source state not found: {SRC_STATE}")
    obs = json.loads(SRC_STATE.read_text())
    # Keep only the first N_SEED primary observations (no repeats)
    seeded, n_primary = [], 0
    for o in obs:
        if not o.get("is_repeat"):
            if n_primary >= N_SEED:
                break
            n_primary += 1
        seeded.append(o)

    dest = TEST_OUTPUT / "spirals"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "bo_state.json").write_text(json.dumps(seeded, indent=2))
    return n_primary


def main():
    args = parse_args()

    n_primaries = seed_test_dir()
    print(f"Seeded from {SRC_STATE} ({n_primaries} primaries = N_SOBOL — entering GP phase immediately)")
    epochs_str = str(args.max_epochs) if args.max_epochs else "full"
    print(f"Running {args.n_iter} GP iterations  [h={args.h}  beta={args.beta}  max_epochs={epochs_str}]")

    from tasks import TASKS
    from src.bo import (
        get_all_combos, cat_params_for_task, suggest_next,
        save_state, load_state, build_run_counts,
        get_primary_observations,
    )
    from src.train_supervised import train_network

    task       = TASKS["spirals"]()
    output_dir = TEST_OUTPUT / "spirals"
    state_path = output_dir / "bo_state.json"

    all_combos = get_all_combos(task)
    cat_params = cat_params_for_task(task)
    print(f"Categorical combos: {len(all_combos)}")

    ds_train, ds_val = task.get_data(data_dir="data")
    rdm_inputs, _    = task.get_rdm_stimuli(data_dir="data")

    observations = load_state(state_path)

    for i in range(args.n_iter):
        config, combo_idx, mode, cont_unit_vals = suggest_next(
            observations, task, beta=args.beta, h=args.h,
        )
        primary_obs = get_primary_observations(observations)
        counts      = build_run_counts(primary_obs, all_combos, cat_params)
        n_prev      = counts[combo_idx]

        pretty = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in config.items()}
        print(f"\n[iter {i+1}/{args.n_iter}]  combo #{combo_idx}  ({mode}, {n_prev} prior obs)")
        print(f"  config: {json.dumps(pretty)}")

        iteration = len(observations)
        run_dir   = output_dir / f"run_{iteration:04d}_r0"
        val_acc   = train_network(
            task=task, config=config, run_dir=run_dir,
            rdm_inputs=rdm_inputs, ds_train=ds_train, ds_val=ds_val,
            max_epochs_override=args.max_epochs, verbose=True,
        )
        flag = "OK" if val_acc >= task.success_threshold else "FAILED"
        print(f"  val_acc={val_acc:.4f}  [{flag}]")

        observations.append({
            "iteration":      iteration,
            "config":         config,
            "cont_unit_vals": cont_unit_vals,
            "val_accs":       [val_acc],
            "mean_metric":    val_acc,
            "is_repeat":      False,
            "repeat_of":      None,
        })
        save_state(state_path, observations)

    print(f"\nDone. Results in {output_dir}/")


if __name__ == "__main__":
    main()
