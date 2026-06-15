"""
Full-task test run: all 9 tasks, 20 Sobol primaries each, no repeats, CPU only.
180 networks total. Goal: flag tasks that are too slow before the AWS run.

Usage:
    python scripts/run_supervised_test.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT       = Path(__file__).parent.parent
OUTPUT_DIR = "output/experiments_supervised_test"
TASKS      = [
    "mnist_dual", "mnist_10way", "fashion_10way", "parity", "spirals",
    "mnist_rnn", "adding",
    "fourrooms", "cartpole",
]

COMMON = [
    sys.executable, str(ROOT / "run_bo.py"),
    "--n-iter",              "20",
    "--n-sobol",             "20",
    "--output-dir",          OUTPUT_DIR,
    "--no-save-activations",
    "--no-repeats",
]

env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}

for task in TASKS:
    print(f"\n{'='*60}")
    print(f"  Task: {task}")
    print(f"{'='*60}\n")
    subprocess.run(COMMON + ["--task", task], cwd=ROOT, check=True, env=env)

# Per-network timing summary from metadata.json
print(f"\n{'='*60}")
print("  Timing summary")
print(f"{'='*60}\n")
print(f"  {'Task':<16}  {'n':>4}  {'mean':>8}  {'min':>8}  {'max':>8}  {'total':>8}")
print(f"  {'-'*16}  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

for task in TASKS:
    run_dirs = sorted((ROOT / OUTPUT_DIR / task).glob("run_*"))
    times = []
    for d in run_dirs:
        meta = d / "metadata.json"
        if meta.exists():
            t = json.loads(meta.read_text()).get("training_time_s")
            if t is not None:
                times.append(t)
    if times:
        mean = sum(times) / len(times)
        print(f"  {task:<16}  {len(times):>4}  {mean/60:>7.1f}m  "
              f"{min(times)/60:>7.1f}m  {max(times)/60:>7.1f}m  {sum(times)/60:>7.1f}m")
    else:
        print(f"  {task:<16}  no timing data")

print()
