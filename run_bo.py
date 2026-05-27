"""
Stratified round-robin Bayesian optimisation over a single task.

Usage:
    python run_bo.py --task spirals [--n-iter 300] [--output-dir experiments]
                     [--beta 8.0] [--max-epochs 100]

Every 4 primary iterations a repeat of the most recent primary is inserted
(P P P P R pattern), giving a ~20% repeat rate for aleatoric noise estimation.
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
    get_primary_observations,
)

STATE_FILE = "bo_state.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",        type=str,   required=True,
                   choices=list(TASKS.keys()))
    p.add_argument("--n-iter",      type=int,   default=300)
    p.add_argument("--output-dir",  type=str,   default="experiments")
    p.add_argument("--data-dir",    type=str,   default="data")
    p.add_argument("--beta",        type=float, default=8.0)
    p.add_argument("--max-epochs",  type=int,   default=None,
                   help="Override task's max_epochs (e.g. 5 for a quick smoke test)")
    return p.parse_args()


def run_config(task, config, run_id_base, output_dir, rdm_inputs,
               ds_train, ds_val, max_epochs_override):
    run_dir = Path(output_dir) / f"{run_id_base}_r0"
    print(f"    ->  {run_dir.name}")

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
    return val_acc


def _pending_repeat(observations):
    """Return (config, primary_idx) if the most recent 4th primary needs a repeat.

    Pattern: P P P P R P P P P R ...
    A repeat is triggered after every 4th primary (n_primary divisible by 4),
    giving a 20% repeat rate (1 repeat per 4 primaries = 1/5 of total iterations).
    """
    primary_obs = get_primary_observations(observations)
    n_primary = len(primary_obs)

    if n_primary == 0 or n_primary % 4 != 0:
        return None, None

    last_primary_idx = None
    for i in range(len(observations) - 1, -1, -1):
        if not observations[i].get("is_repeat", False):
            last_primary_idx = i
            break

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
    run_counts    = build_run_counts(primary_obs, all_combos, cat_params)
    coverage      = sum(1 for c in run_counts if c > 0)
    print(f"\nResuming: {n_done} total ({n_primary} primary, {n_repeat} repeats), "
          f"{coverage}/{len(all_combos)} categorical combos visited.")

    for iteration in range(n_done, args.n_iter):
        print(f"\n{'='*60}")

        repeat_config, repeat_of_idx = _pending_repeat(observations)

        if repeat_config is not None:
            config    = repeat_config
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

        val_acc = run_config(
            task                = task,
            config              = config,
            run_id_base         = f"run_{iteration:04d}",
            output_dir          = output_dir,
            rdm_inputs          = rdm_inputs,
            ds_train            = ds_train,
            ds_val              = ds_val,
            max_epochs_override = args.max_epochs,
        )

        print(f"  mean_metric = {val_acc:.4f}")

        observations.append({
            "iteration":   iteration,
            "config":      config,
            "val_accs":    [val_acc],
            "mean_metric": val_acc,
            "is_repeat":   is_repeat,
            "repeat_of":   repeat_of_idx if is_repeat else None,
        })
        save_state(state_path, observations)

    print(f"\nDone. {len(observations)} total runs.")
    primary_final  = get_primary_observations(observations)
    counts_final   = build_run_counts(primary_final, all_combos, cat_params)
    coverage_final = sum(1 for c in counts_final if c > 0)
    n_repeats_final = len(observations) - len(primary_final)
    print(f"Primary runs: {len(primary_final)}  Repeats: {n_repeats_final}  "
          f"({100*n_repeats_final/len(observations):.1f}%)")
    print(f"Categorical coverage: {coverage_final}/{len(all_combos)} combos visited.")
    best = max(primary_final, key=lambda o: o["mean_metric"])
    print(f"Best mean_metric: {best['mean_metric']:.4f}")
    print(f"Best config: {best['config']}")


if __name__ == "__main__":
    main()
