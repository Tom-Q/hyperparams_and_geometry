"""
Stratified round-robin Bayesian optimisation over a single task.

Usage:
    python run_bo.py --task spirals [--n-iter 30] [--runs-per-config 1]
                     [--output-dir experiments] [--beta 8.0]
                     [--max-epochs 100]
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
)

STATE_FILE = "bo_state.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",            type=str,   required=True,
                   choices=list(TASKS.keys()))
    p.add_argument("--n-iter",          type=int,   default=300)
    p.add_argument("--runs-per-config", type=int,   default=2)
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
        print(f"    run {r+1}/{runs_per_config}  →  {run_dir.name}")

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

    observations = load_state(state_path)
    n_done       = len(observations)
    run_counts   = build_run_counts(observations, all_combos, cat_params)
    coverage     = sum(1 for c in run_counts if c > 0)
    print(f"\nResuming: {n_done} configs done, "
          f"{coverage}/{len(all_combos)} categorical combos visited.")

    for iteration in range(n_done, args.n_iter):
        print(f"\n{'='*60}")

        config, combo_idx, mode = suggest_next(observations, task, beta=args.beta)

        run_counts_now = build_run_counts(observations, all_combos, cat_params)
        n_prev = run_counts_now[combo_idx]
        print(f"[{iteration+1}/{args.n_iter}]  combo #{combo_idx}  "
              f"({mode}, {n_prev} prior obs for this combo)")

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

        mean_acc  = float(np.mean(val_accs))
        print(f"  mean_metric (scored) = {mean_acc:.4f}")

        observations.append({
            "iteration":   iteration,
            "config":      config,
            "val_accs":    val_accs,
            "mean_metric": mean_acc,
        })
        save_state(state_path, observations)

    print(f"\nDone. {len(observations)} configs trained.")
    run_counts_final = build_run_counts(observations, all_combos, cat_params)
    coverage_final   = sum(1 for c in run_counts_final if c > 0)
    print(f"Categorical coverage: {coverage_final}/{len(all_combos)} combos visited.")
    best = max(observations, key=lambda o: o["mean_metric"])
    print(f"Best mean_metric: {best['mean_metric']:.4f}")
    print(f"Best config: {best['config']}")


if __name__ == "__main__":
    main()
