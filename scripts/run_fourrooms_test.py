"""
Local test run: fourrooms, 50 Sobol + 100 GP primaries = 187 total iterations.

Usage:
    python scripts/run_fourrooms_test.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

cmd = [
    sys.executable, str(ROOT / "run_bo.py"),
    "--task",     "fourrooms",
    "--n-iter",   "187",
    "--n-sobol",  "50",
    "--output-dir", "output/experiments",
    "--beta",     "4.0",
    "--h",        "0.15",
]

subprocess.run(cmd, cwd=ROOT, check=True)
