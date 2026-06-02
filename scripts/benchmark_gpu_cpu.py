"""
Quick GPU vs CPU benchmark: train one network on each device, compare wall time.
Tasks: spirals, mnist_10way. Config: batch_size=1 (the slow case on AWS).

Usage:
    python scripts/benchmark_gpu_cpu.py
"""
import sys
import time
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tasks import TASKS
from src.train_supervised import train_network

N_EPOCHS = 5

CONFIG = {
    "batch_size":     1,
    "depth":          2,
    "activation":     "relu",
    "optimizer":      "sgd",
    "init_scale":     0.1,
    "learning_rate":  1e-3,
    "l1_reg":         1e-4,
    "l2_reg":         1e-4,
    "hidden_size":    64,
}


def bench(task_name):
    task = TASKS[task_name]()
    ds_train, ds_val = task.get_data()
    rdm_inputs, _ = task.get_rdm_stimuli()

    results = []
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    else:
        print("  (no CUDA device found — skipping GPU)")

    for device in devices:
        with tempfile.TemporaryDirectory() as tmp:
            t0 = time.time()
            train_network(
                task=task, config=CONFIG,
                run_dir=Path(tmp) / "run",
                rdm_inputs=rdm_inputs,
                ds_train=ds_train, ds_val=ds_val,
                max_epochs_override=N_EPOCHS,
                device=device,
                verbose=False,
            )
            elapsed = time.time() - t0
        results.append((str(device), elapsed))
        print(f"  {str(device):6s}  {elapsed:.1f}s  ({elapsed/N_EPOCHS:.1f}s/epoch)")

    if len(results) == 2:
        ratio = results[0][1] / results[1][1]
        print(f"  → CPU is {ratio:.1f}x slower than GPU")


for task_name in ["spirals", "mnist_10way"]:
    print(f"\n{task_name}  ({N_EPOCHS} epochs, batch_size=1)")
    bench(task_name)
