"""
RNN test run: mnist_rnn then adding.
50 Sobol + 100 GP primaries = 187 total iterations per task.
h=0.19 per the p90 N_eff = 0.5 criterion at 250 primaries.

Usage:
    python scripts/run_rnn_test.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

COMMON = [
    sys.executable, str(ROOT / "run_bo.py"),
    "--n-iter",    "187",
    "--n-sobol",   "50",
    "--output-dir", "output/experiments_rnn_test",
    "--beta",      "4.0",
    "--h",         "0.19",
    "--no-save-activations",
]

for task in ["adding", "mnist_rnn"]:
    print(f"\n{'='*60}")
    print(f"  Task: {task}")
    print(f"{'='*60}\n")
    subprocess.run(COMMON + ["--task", task], cwd=ROOT, check=True)
