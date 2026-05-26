"""
Stratified round-robin Bayesian optimisation over a single task.

Usage:
    python run_bo.py --task spirals [--n-iter 30] [--runs-per-config 1]
                     [--output-dir experiments] [--beta 8.0]
                     [--max-epochs 100]

Every other iteration repeats the most recent primary config so the RF
surrogate can estimate aleatoric noise from (y1-y2)^2/2.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from tasks import TASKS
from src.train_supervised import train_network
from src.bo import (
    get_all_combos, cat_params_for_task, suggest_next,
    save_state, load_state, build_run_counts,
    get_primary_observations, get_repeat_pairs,
)

STATE_FILE = "bo_state.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",            type=str,   required=True,
                   choices=list(TASKS.keys()))
    p.add_argument("--n-iter",          type=int,   default=300)
    p.add_argument("--runs-per-config", type=int,   default=1)
    p.add_argument("--output-dir",      type=str,   default="experiments")
    p.add_argument("--data-dir",        type=str,   default="data")
    p.add_argument("--beta",            type=float, default=8.0)
    p.add_argument("--max-epochs",      type=int,   default=None,
                   help="Override task's max_epochs (e.g. 5 for a quick smoke test)")
    return p.parse_args()


def run_config(task, config, run_id_base, output_dir, rdm_inputs,
               ds_train, ds_val, runs_per_config, max_epochs_override):
    val_accs = []
    for r in range(runs_per_config):
        run_dir = Path(output_dir) / f"{run_id_base}_r{r}"
        print(f"    run {r+1}/{runs_per_config}  ->  {run_dir.name}")

        val_acc = train_network(
            task                = task,
            config              = config,
            run_dir             = run_dir,
            rdm_inputs          = rdm_inputs,
            ds_train            = ds_train,
            ds_val              = ds_val,
            max_epochs_override = max_epochs_override,
            verbose             = True,
        )

        flag = "OK" if val_acc >= task.success_threshold else "FAILED"
        print(f"        val_acc={val_acc:.4f}  [{flag}]")
        val_accs.append(val_acc)

    return val_accs


def _pending_repeat(observations):
    """Return (config, primary_idx) if the last primary obs has no repeat yet, else (None, None)."""
    if not observations:
        return None, None
    for i in range(len(observations) - 1, -1, -1):
        if not observations[i].get("is_repeat", False):
            last_primary_idx = i
            break
    else:
        return None, None
    has_repeat = any(
        o.get("is_repeat") and o.get("repeat_of") == last_primary_idx
        for o in observations
    )
    if has_repeat:
        return None, None
    return observations[last_primary_idx]["config"], last_primary_idx


def main():
    args = parse_args()

    task       = TASKS[args.task]()
    output_dir = Path(args.output_dir) / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / STATE_FILE

    all_combos = get_all_combos(task)
    cat_params = cat_params_for_task(task)

    print(f"Task: {task.name}  ({len(all_combos)} categorical combos)")

    print("Loading data...")
    ds_train, ds_val = task.get_data(data_dir=args.data_dir)
    print(f"  train={len(ds_train)}  val={len(ds_val)}")

    rdm_inputs, _ = task.get_rdm_stimuli(data_dir=args.data_dir)
    print(f"  RDM stimuli: {rdm_inputs.shape}")

    observations  = load_state(state_path)
    n_done        = len(observations)
    primary_obs   = get_primary_observations(observations)
    n_primary     = len(primary_obs)
    n_repeat      = n_done - n_primary
    repeat_pairs  = get_repeat_pairs(observations)
    run_counts    = build_run_counts(primary_obs, all_combos, cat_params)
    coverage      = sum(1 for c in run_counts if c > 0)
    print(f"\nResuming: {n_done} total ({n_primary} primary, {n_repeat} repeats), "
          f"{len(repeat_pairs)} noise pairs, "
          f"{coverage}/{len(all_combos)} categorical combos visited.")

    for iteration in range(n_done, args.n_iter):
        print(f"\n{'='*60}")

        repeat_config, repeat_of_idx = _pending_repeat(observations)

        if repeat_config is not None:
            config    = repeat_config
            mode      = "repeat"
            is_repeat = True
            print(f"[{iteration+1}/{args.n_iter}]  REPEAT of run_{repeat_of_idx:04d}  (noise pair)")
        else:
            config, combo_idx, mode = suggest_next(observations, task, beta=args.beta)
            is_repeat = False
            primary_now = get_primary_observations(observations)
            counts_now  = build_run_counts(primary_now, all_combos, cat_params)
            n_prev      = counts_now[combo_idx]
            print(f"[{iteration+1}/{args.n_iter}]  combo #{combo_idx}  "
                  f"({mode}, {n_prev} prior primary obs for this combo)")

        pretty = {k: (round(v, 6) if isinstance(v, float) else v)
                  for k, v in config.items()}
        print(f"  config: {json.dumps(pretty, separators=(',', ':'))}")

        val_accs = run_config(
            task                = task,
            config              = config,
            run_id_base         = f"run_{iteration:04d}",
            output_dir          = output_dir,
            rdm_inputs          = rdm_inputs,
            ds_train            = ds_train,
            ds_val              = ds_val,
            runs_per_config     = args.runs_per_config,
            max_epochs_override = args.max_epochs,
        )

        mean_acc = float(np.mean(val_accs))
        print(f"  mean_metric = {mean_acc:.4f}")

        observations.append({
            "iteration":  iteration,
            "config":     config,
            "val_accs":   val_accs,
            "mean_metric": mean_acc,
            "is_repeat":  is_repeat,
            "repeat_of":  repeat_of_idx,
        })
        save_state(state_path, observations)

    print(f"\nDone. {len(observations)} total runs.")
    primary_final = get_primary_observations(observations)
    counts_final  = build_run_counts(primary_final, all_combos, cat_params)
    coverage_final = sum(1 for c in counts_final if c > 0)
    print(f"Primary runs: {len(primary_final)}  Repeats: {len(observations) - len(primary_final)}")
    print(f"Categorical coverage: {coverage_final}/{len(all_combos)} combos visited.")
    best = max(primary_final, key=lambda o: o["mean_metric"])
    print(f"Best mean_metric: {best['mean_metric']:.4f}")
    print(f"Best config: {best['config']}")


if __name__ == "__main__":
    main()
