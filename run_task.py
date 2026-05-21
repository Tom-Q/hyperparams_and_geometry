"""
Task-aware Bayesian optimisation entry point.

Usage:
    python run_task.py --task mnist_dual   [--n-iter 300] [--runs-per-config 2]
    python run_task.py --task mnist_10way  [--n-iter 300]
    python run_task.py --task cartpole     [--n-iter 200] [--runs-per-config 1]
    python run_task.py --task mnist_rnn    [--n-iter 150]

    python run_task.py --task mnist_dual --n-iter 1 --runs-per-config 1 --max-epochs 1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from tasks import TASKS
from src.bo import (
    suggest_next, save_state, load_state, build_run_counts, get_all_combos,
    cat_params_for_task,
)

STATE_FILE = "bo_state.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",            type=str, required=True,
                   help=f"Task name. Available: {sorted(TASKS.keys())}")
    p.add_argument("--n-iter",          type=int,   default=300)
    p.add_argument("--runs-per-config", type=int,   default=2)
    p.add_argument("--output-dir",      type=str,   default="experiments")
    p.add_argument("--data-dir",        type=str,   default="data")
    p.add_argument("--stimuli-path",    type=str,   default=None,
                   help="Path to pre-built RDM stimuli .npz (built if missing)")
    p.add_argument("--beta",            type=float, default=8.0)
    p.add_argument("--max-epochs",      type=int,   default=None,
                   help="Override max epochs / max steps (quick smoke test)")
    return p.parse_args()


def _penalty_for_task(task):
    """Return the 'failed run' penalty value for the BO objective."""
    if task.metric_name == "val_acc":
        return 0.50
    if task.metric_name == "mean_return":
        return 0.0
    if task.metric_name == "val_mse":
        return 1.0   # large MSE penalty; negated in _score → -1.0
    return 0.0


def _score(metric, task, penalty):
    if task.metric_name == "val_mse":
        # For MSE: negate so GP maximises (lower MSE = better = less negative)
        if metric <= task.success_threshold:  # success = MSE below threshold
            return -metric
        return -penalty
    if metric >= task.success_threshold:
        return metric
    return penalty


def _success_flag(metric, task):
    if task.metric_name == "val_mse":
        return "OK" if metric <= task.success_threshold else "FAILED"
    return "OK" if metric >= task.success_threshold else "FAILED"


def run_config_sequential(task, config, run_id_base, output_dir, rdm_inputs,
                          ds_train, ds_val, runs_per_config, max_epochs_override):
    if task.paradigm == "supervised":
        from src.train_supervised import train_network
    else:
        from src.train_rnn import train_network
    metrics = []
    for r in range(runs_per_config):
        run_dir = Path(output_dir) / f"{run_id_base}_r{r}"
        print(f"    run {r+1}/{runs_per_config}  →  {run_dir.name}")
        m = train_network(
            task                = task,
            config              = config,
            run_dir             = run_dir,
            rdm_inputs          = rdm_inputs,
            ds_train            = ds_train,
            ds_val              = ds_val,
            max_epochs_override = max_epochs_override,
        )
        print(f"        {task.metric_name}={m:.4f}  [{_success_flag(m, task)}]")
        metrics.append(m)
    return metrics


def run_config_rl(task, config, run_id_base, output_dir, rdm_inputs,
                  runs_per_config, max_steps_override, env_factory):
    from src.train_rl import train_network
    metrics = []
    for r in range(runs_per_config):
        run_dir = Path(output_dir) / f"{run_id_base}_r{r}"
        print(f"    run {r+1}/{runs_per_config}  →  {run_dir.name}")
        m = train_network(
            task               = task,
            config             = config,
            run_dir            = run_dir,
            rdm_inputs         = rdm_inputs,
            env_factory        = env_factory,
            max_steps_override = max_steps_override,
        )
        print(f"        {task.metric_name}={m:.4f}  [{_success_flag(m, task)}]")
        metrics.append(m)
    return metrics


def main():
    args = parse_args()

    if args.task not in TASKS:
        print(f"Unknown task '{args.task}'. Available: {sorted(TASKS.keys())}")
        sys.exit(1)

    task = TASKS[args.task]()

    # Per-task output directory
    task_output_dir = Path(args.output_dir) / task.name
    task_output_dir.mkdir(parents=True, exist_ok=True)
    state_path = task_output_dir / STATE_FILE

    # RDM stimuli
    stimuli_path = args.stimuli_path or str(task_output_dir / "rdm_stimuli.npz")
    if not Path(stimuli_path).exists():
        print(f"Building RDM stimuli for {task.name}...")
        rdm_inputs, metadata = task.get_rdm_stimuli(data_dir=args.data_dir)
        np.savez(stimuli_path, inputs=rdm_inputs, **metadata)
        print(f"  saved to {stimuli_path}  ({len(rdm_inputs)} stimuli)")
    else:
        print(f"Loading RDM stimuli from {stimuli_path}")
        data = np.load(stimuli_path, allow_pickle=True)
        rdm_inputs = data["inputs"]

    # Data / env
    if task.paradigm in ("supervised", "rnn"):
        print(f"Loading data for {task.name}...")
        ds_train, ds_val = task.get_data(data_dir=args.data_dir)
        print(f"  train={len(ds_train)}  val={len(ds_val)}")
        env_factory = None
    else:  # rl
        ds_train = ds_val = None
        env_factory = task.get_data(data_dir=args.data_dir)

    penalty    = _penalty_for_task(task)
    all_combos = get_all_combos(task)

    # Resume state
    observations = load_state(state_path)
    n_done       = len(observations)
    cat_params   = cat_params_for_task(task)
    run_counts = build_run_counts(observations, all_combos, cat_params)
    coverage   = sum(1 for c in run_counts if c > 0)
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

        if task.paradigm in ("supervised", "rnn"):
            metrics = run_config_sequential(
                task, config, f"run_{iteration:04d}", task_output_dir,
                rdm_inputs, ds_train, ds_val, args.runs_per_config, args.max_epochs,
            )
        else:  # rl
            metrics = run_config_rl(
                task, config, f"run_{iteration:04d}", task_output_dir,
                rdm_inputs, args.runs_per_config, args.max_epochs, env_factory,
            )

        scored      = [_score(m, task, penalty) for m in metrics]
        mean_metric = float(np.mean(scored))
        print(f"  mean_{task.metric_name} (scored) = {mean_metric:.4f}")

        observations.append({
            "iteration":   iteration,
            "config":      config,
            "metrics":     metrics,
            "mean_metric": mean_metric,
        })
        save_state(state_path, observations)

    print(f"\nDone. {len(observations)} configs trained.")
    run_counts_final = build_run_counts(observations, all_combos, cat_params)
    coverage_final   = sum(1 for c in run_counts_final if c > 0)
    print(f"Categorical coverage: {coverage_final}/{len(all_combos)} combos visited.")
    best = max(observations, key=lambda o: o["mean_metric"])
    print(f"Best mean_{task.metric_name}: {best['mean_metric']:.4f}")
    print(f"Best config: {best['config']}")


if __name__ == "__main__":
    main()
