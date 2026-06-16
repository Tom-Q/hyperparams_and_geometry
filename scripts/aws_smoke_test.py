#!/usr/bin/env python3
"""AWS smoke test (part 3 of TESTING.md pre-AWS verification plan).

Runs run_bo.py for all 9 tasks, 2 networks each, through the real entrypoint.
Validates checkpoint structure and activation file content for every run.

Usage (on EC2):
    python scripts/aws_smoke_test.py [--output-dir output/smoke] [--data-dir data]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks import TASKS
from src.utils import log4_checkpoints, epoch_checkpoints, format_epoch_label, PERF_THRESHOLDS

# Small training budgets that still exercise all checkpoint types (step/epoch/perf).
# RL tasks use max_steps; supervised/RNN use max_epochs.
TASK_MAX_EPOCHS = {
    "mnist_dual":    5,
    "mnist_10way":   5,
    "fashion_10way": 5,
    "spirals":       5,
    "parity":        5,
    "adding":        5,
    "mnist_rnn":     5,
    "cartpole":      10_000,
    "fourrooms":     10_000,
}

N_ITER = 2


def mlp_layer_sizes(hidden_size, depth):
    return [hidden_size // (2 ** i) for i in range(depth)]


def check_npz(path, expected_keys, n_stimuli, tag=""):
    data = np.load(path)
    actual = set(data.files)
    missing = expected_keys - actual
    extra   = actual - expected_keys
    errors  = []
    if missing:
        errors.append(f"missing keys: {missing}")
    if extra:
        errors.append(f"unexpected keys: {extra}")
    for key in expected_keys & actual:
        arr = data[key]
        if arr.shape[0] != n_stimuli:
            errors.append(f"{key}: expected {n_stimuli} stimuli, got {arr.shape[0]}")
        if not np.isfinite(arr).all():
            errors.append(f"{key}: contains NaN/Inf")
    if errors:
        return [f"  {path.name}{' ' + tag if tag else ''}: {e}" for e in errors]
    return []


def validate_run(run_dir, task, max_budget, data_dir="data"):
    import math
    errors = []
    run_dir = Path(run_dir)

    if not (run_dir / "metadata.json").exists():
        return [f"  metadata.json missing in {run_dir.name}"]
    if not (run_dir / "history.json").exists():
        errors.append(f"  history.json missing in {run_dir.name}")

    meta       = json.loads((run_dir / "metadata.json").read_text())
    config     = meta["config"]
    final_step = meta["final_step"]
    hidden     = int(config["hidden_size"])

    rdm_inputs, _ = task.get_rdm_stimuli(data_dir=data_dir)
    n_stimuli = rdm_inputs.shape[0]

    if task.paradigm in ("supervised", "rl"):
        layer_sizes    = mlp_layer_sizes(hidden, int(config["depth"]))
        expected_keys  = {f"layer_{i}" for i in range(len(layer_sizes))}
    else:  # rnn
        n_layers = int(config["n_rnn_layers"])
        n_steps  = task.n_steps
        expected_keys = {f"layer_{l}_t_{t}" for l in range(n_layers) for t in range(n_steps)}

    # step checkpoints
    if task.paradigm == "rl":
        total_steps = max_budget  # max_budget is max_steps for RL
    else:
        batch_size  = int(config.get("batch_size", 1))
        ds_train, _ = task.get_data(data_dir=data_dir)
        spe         = math.ceil(len(ds_train) / batch_size)
        total_steps = max_budget * spe

    expected_step = {f"step_{s:07d}" for s in log4_checkpoints(total_steps) if s <= final_step}
    actual_step   = {p.stem for p in run_dir.glob("step_*.npz")}
    if actual_step != expected_step:
        errors.append(f"  step checkpoints mismatch: got {actual_step} expected {expected_step}")
    for p in run_dir.glob("step_*.npz"):
        errors += check_npz(p, expected_keys, n_stimuli)

    # epoch checkpoints (supervised/rnn only)
    if task.paradigm != "rl":
        valid_epoch_labels = {
            f"epoch_{format_epoch_label(e)}"
            for s, e in epoch_checkpoints(spe, max_budget).items()
            if s <= final_step
        }
        actual_epoch = {p.stem for p in run_dir.glob("epoch_*.npz")}
        if actual_epoch != valid_epoch_labels:
            errors.append(f"  epoch checkpoints mismatch: got {actual_epoch} expected {valid_epoch_labels}")
        for p in run_dir.glob("epoch_*.npz"):
            errors += check_npz(p, expected_keys, n_stimuli)

        if not (run_dir / "best.npz").exists():
            errors.append("  best.npz missing")
        else:
            errors += check_npz(run_dir / "best.npz", expected_keys, n_stimuli)
    else:
        if list(run_dir.glob("epoch_*.npz")):
            errors.append("  unexpected epoch_*.npz files in RL run")
        if (run_dir / "best.npz").exists():
            errors.append("  unexpected best.npz in RL run")

    # perf checkpoints
    valid_perf_labels = {f"{t:g}".replace(".", "p") for t in PERF_THRESHOLDS}
    for p in run_dir.glob("perf_*.npz"):
        label = p.stem[len("perf_"):]
        if label not in valid_perf_labels:
            errors.append(f"  unrecognised perf label: {p.name}")
        errors += check_npz(p, expected_keys, n_stimuli)

    # final
    if not (run_dir / "final.npz").exists():
        errors.append("  final.npz missing")
    else:
        errors += check_npz(run_dir / "final.npz", expected_keys, n_stimuli)

    return errors


def run_task(task_name, output_dir, data_dir):
    max_budget = TASK_MAX_EPOCHS[task_name]
    cmd = [
        sys.executable, "run_bo.py",
        "--task",       task_name,
        "--n-iter",     str(N_ITER),
        "--no-repeats",
        "--max-epochs", str(max_budget),
        "--output-dir", str(output_dir),
        "--data-dir",   str(data_dir),
    ]
    print(f"\n{'='*60}")
    print(f"[{task_name}]  running {N_ITER} networks (max_budget={max_budget})")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="output/smoke")
    ap.add_argument("--data-dir",   default="data")
    ap.add_argument("--tasks", nargs="+", choices=list(TASK_MAX_EPOCHS),
                    help="Run only these tasks (default: all)")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    data_dir   = Path(args.data_dir)
    task_order = args.tasks if args.tasks else list(TASK_MAX_EPOCHS)

    train_ok  = {}
    val_errors = {}

    for task_name in task_order:
        ok = run_task(task_name, output_dir, data_dir)
        train_ok[task_name] = ok

        task_dir = output_dir / task_name
        task     = TASKS[task_name]()
        max_bud  = TASK_MAX_EPOCHS[task_name]

        print(f"\n  Validating {task_name} ...")
        errs = []
        for i in range(N_ITER):
            run_dir = task_dir / f"run_{i:04d}_r0"
            if not run_dir.exists():
                errs.append(f"  run_{i:04d}_r0/ directory not found")
                continue
            errs += validate_run(run_dir, task, max_bud, data_dir=str(data_dir))
        val_errors[task_name] = errs

    # --- summary ---
    print(f"\n{'='*60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for task_name in task_order:
        train_status = "OK" if train_ok[task_name] else "FAILED (non-zero exit)"
        errs         = val_errors[task_name]
        val_status   = "OK" if not errs else f"FAILED ({len(errs)} error(s))"
        status       = "PASS" if train_ok[task_name] and not errs else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {status}  {task_name:<16}  train={train_status}  validation={val_status}")
        for e in errs:
            print(f"        {e}")
    print()
    if all_pass:
        print("All tasks passed.")
        sys.exit(0)
    else:
        print("One or more tasks FAILED — see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
