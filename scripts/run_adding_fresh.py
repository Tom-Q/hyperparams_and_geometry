#!/usr/bin/env python3
"""Fresh BO run for adding task. Starts from scratch with 4 BLAS threads."""
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT   = Path(__file__).parent.parent
PYTHON = sys.executable

env = {
    **os.environ,
    "OMP_NUM_THREADS":  "4",
    "MKL_NUM_THREADS":  "4",
    "PYTHONUNBUFFERED": "1",
}

log_path = ROOT / "output/production/adding/run_local.log"
(ROOT / "output/production/adding").mkdir(parents=True, exist_ok=True)

cmd = [
    PYTHON, str(ROOT / "run_bo.py"),
    "--task",       "adding",
    "--n-iter",     "1000",
    "--n-sobol",    "200",
    "--output-dir", str(ROOT / "output/production"),
    "--beta",       "4.0",
    "--h",          "0.147",
]

proc = None

def shutdown(signum, frame):
    print("\nInterrupted — terminating...")
    if proc:
        proc.terminate()
        proc.wait()
    sys.exit(1)

signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

log  = open(log_path, "a")
proc = subprocess.Popen(cmd, stdout=log, stderr=log, env=env, cwd=ROOT)
print(f"Launched adding (fresh run)  PID {proc.pid}")
print(f"  log: {log_path}")
print(f"  monitor: tail -f {log_path}")

code = proc.wait()
print(f"adding: {'done' if code == 0 else f'FAILED (exit {code})'}")
