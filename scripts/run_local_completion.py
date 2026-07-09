#!/usr/bin/env python3
"""Launch mnist_dual and adding simultaneously to completion, one thread each."""
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PYTHON = sys.executable

env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}

tasks = [
    {
        "task":       "mnist_dual",
        "n_iter":     1000,
        "n_sobol":    200,
        "h":          0.162,
        "remaining":  360,
    },
    {
        "task":       "adding",
        "n_iter":     1000,
        "n_sobol":    200,
        "h":          0.147,
        "remaining":  948,
    },
]

procs = []

def shutdown(signum, frame):
    print("\nInterrupted — terminating child processes...")
    for _, proc, _ in procs:
        proc.terminate()
    for _, proc, _ in procs:
        proc.wait()
    sys.exit(1)

signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

for t in tasks:
    log_path = ROOT / "output/production" / t["task"] / "run_local.log"
    cmd = [
        PYTHON, str(ROOT / "run_bo.py"),
        "--task",       t["task"],
        "--n-iter",     str(t["n_iter"]),
        "--n-sobol",    str(t["n_sobol"]),
        "--output-dir", str(ROOT / "output/production"),
        "--beta",       "4.0",
        "--h",          str(t["h"]),
    ]
    log = open(log_path, "a")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log, env=env, cwd=ROOT)
    procs.append((t["task"], proc, log_path))
    print(f"Launched {t['task']} ({t['remaining']} iterations remaining)  PID {proc.pid}")
    print(f"  log: {log_path}")

print("\nMonitor with:")
for task, _, log_path in procs:
    print(f"  tail -f {log_path}")

for task, proc, _ in procs:
    code = proc.wait()
    status = "done" if code == 0 else f"FAILED (exit {code})"
    print(f"{task}: {status}")
