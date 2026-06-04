"""
RL epsilon-decay test: fourrooms then cartpole.
Schedule: ε = 0.5 → 0 linearly over 100,000 steps.
50 Sobol + 100 GP primaries = 187 total iterations per task.
Performance tracked as rolling mean of last 30 training episodes.

Usage:
    python scripts/run_rl_epsilon_test.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

COMMON = [
    sys.executable, str(ROOT / "run_bo.py"),
    "--n-iter",              "187",
    "--n-sobol",             "50",
    "--output-dir",          "output/experiments_rl_epsilon_test",
    "--beta",                "4.0",
    "--h",                   "0.15",
    "--epsilon-start",       "0.5",
    "--epsilon-end",         "0.0",
    "--epsilon-decay-steps", "100000",
    "--no-save-activations",
]

for task in ["fourrooms", "cartpole"]:
    print(f"\n{'='*60}")
    print(f"  Task: {task}")
    print(f"{'='*60}\n")
    subprocess.run(COMMON + ["--task", task], cwd=ROOT, check=True)
