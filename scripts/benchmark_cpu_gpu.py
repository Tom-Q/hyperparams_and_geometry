"""
CPU vs GPU training time benchmark.

Reads the 20-network test run results, picks 6 representative configs per task
(2 fastest, 2 middle, 2 slowest by CPU time), re-trains them on GPU, and prints
a comparison table.

Usage:
    python scripts/benchmark_cpu_gpu.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT      = Path(__file__).parent.parent
CPU_DIR   = ROOT / "output/experiments_supervised_test"
OUT_DIR   = ROOT / "output/benchmark_cpu_gpu"
DATA_DIR  = "data"

sys.path.insert(0, str(ROOT))

from tasks import TASKS
from src.train_supervised import train_network as train_supervised
from src.train_rnn        import train_network as train_rnn

TASKS_TO_BENCH = ["mnist_dual", "mnist_10way", "fashion_10way", "mnist_rnn", "adding"]
N_PICK = 6  # 2 fast + 2 middle + 2 slow


def pick_configs(task_name, n=N_PICK):
    """Return list of (cpu_time_s, run_name, config) sorted by cpu_time_s."""
    task_dir = CPU_DIR / task_name
    entries = []
    for d in sorted(task_dir.glob("run_*")):
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        m = json.loads(meta_path.read_text())
        t = m.get("training_time_s")
        if t is not None:
            entries.append((t, d.name, m["config"]))
    entries.sort(key=lambda x: x[0])
    if len(entries) < n:
        return entries
    # 2 fastest, 2 middle, 2 slowest
    mid = len(entries) // 2
    picked = (
        entries[:2] +
        entries[mid - 1 : mid + 1] +
        entries[-2:]
    )
    return sorted(picked, key=lambda x: x[0])


def train_on_gpu(task, config, run_dir, rdm_inputs, ds_train, ds_val):
    device = torch.device("cuda")
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if task.paradigm == "supervised":
        train_supervised(
            task=task, config=config, run_dir=run_dir,
            rdm_inputs=rdm_inputs, ds_train=ds_train, ds_val=ds_val,
            device=device, verbose=False, save_activations=False,
        )
    elif task.paradigm == "rnn":
        train_rnn(
            task=task, config=config, run_dir=run_dir,
            rdm_inputs=rdm_inputs, ds_train=ds_train, ds_val=ds_val,
            device=device, verbose=False, save_activations=False,
        )
    return time.time() - t0


def main():
    if not torch.cuda.is_available():
        print("No GPU available. Exiting.")
        sys.exit(1)

    results = {}  # task -> list of (cpu_s, gpu_s, config)

    for task_name in TASKS_TO_BENCH:
        print(f"\n{'='*60}")
        print(f"  {task_name}")
        print(f"{'='*60}")

        task = TASKS[task_name]()
        configs = pick_configs(task_name)
        if not configs:
            print("  No CPU results found, skipping.")
            continue

        print(f"  Loading data...")
        ds_train, ds_val = task.get_data(data_dir=DATA_DIR)

        stimuli_path = CPU_DIR / task_name / "rdm_stimuli.npz"
        if stimuli_path.exists():
            rdm_inputs = np.load(stimuli_path)["inputs"]
        else:
            rdm_inputs, _ = task.get_rdm_stimuli(data_dir=DATA_DIR)

        task_results = []
        for i, (cpu_s, run_name, config) in enumerate(configs):
            run_dir = OUT_DIR / task_name / f"{run_name}_gpu"
            print(f"  [{i+1}/{len(configs)}]  cpu={cpu_s/60:.1f}m  ", end="", flush=True)
            gpu_s = train_on_gpu(task, config, run_dir, rdm_inputs, ds_train, ds_val)
            speedup = cpu_s / gpu_s
            print(f"gpu={gpu_s/60:.1f}m  speedup={speedup:.1f}x")
            task_results.append((cpu_s, gpu_s, config))

        results[task_name] = task_results

    # Summary table
    print(f"\n\n{'='*60}")
    print("  Summary: CPU vs GPU training times")
    print(f"{'='*60}\n")
    print(f"  {'Task':<16}  {'CPU':>8}  {'GPU':>8}  {'Speedup':>8}  config snippet")
    print(f"  {'-'*16}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*30}")

    for task_name in TASKS_TO_BENCH:
        if task_name not in results:
            continue
        for cpu_s, gpu_s, config in results[task_name]:
            speedup = cpu_s / gpu_s
            hs  = config.get("hidden_size", "?")
            bs  = config.get("batch_size", "?")
            opt = config.get("optimizer", "?")
            snippet = f"hs={hs} bs={bs} {opt}"
            print(f"  {task_name:<16}  {cpu_s/60:>7.1f}m  {gpu_s/60:>7.1f}m  {speedup:>7.1f}x  {snippet}")
        print()


if __name__ == "__main__":
    main()
