"""
Run BO on the four supervised tasks, up to --parallel tasks at a time.
Each task's full output goes to experiments_supervised_test/<task>.log.
Console shows one summary line per completed network.

Usage:
    python run_supervised_tests.py
    python run_supervised_tests.py --parallel 2        # default
    python run_supervised_tests.py --tasks mnist_dual parity
    python run_supervised_tests.py --n-iter 300 --h 0.15 --beta 4.0
"""
import argparse
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


TASKS = ["mnist_dual", "mnist_10way", "fashion_10way", "parity"]

# Patterns to parse run_bo.py output
RE_HEADER     = re.compile(r"\[(\d+)/(\d+)\].*combo #(\d+)\s+\(([^)]+)\)")
RE_REPEAT     = re.compile(r"\[(\d+)/(\d+)\].*REPEAT")
RE_MEAN       = re.compile(r"performance\s*=\s*([0-9.]+)")
RE_THRESHOLD  = re.compile(r"\[(OK|FAILED)\]")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks",    nargs="+", default=TASKS)
    p.add_argument("--n-iter",   type=int,   default=300)
    p.add_argument("--h",        type=float, default=0.15)
    p.add_argument("--beta",     type=float, default=4.0)
    p.add_argument("--out-dir",  default="output/experiments_supervised_test")
    p.add_argument("--parallel", type=int,   default=2,
                   help="Max tasks running simultaneously")
    return p.parse_args()


def run_task(task, args, log_dir):
    log_path = log_dir / f"{task}.log"
    cmd = [
        sys.executable, str(Path(__file__).parent.parent / "run_bo.py"),
        "--task",       task,
        "--n-iter",     str(args.n_iter),
        "--h",          str(args.h),
        "--beta",       str(args.beta),
        "--output-dir", args.out_dir,
    ]

    t0 = time.time()
    current_iter = None
    current_total = None
    current_label = None  # "combo #7 (sobol, 0 prior)" or "REPEAT"
    flag = None

    with open(log_path, "w") as log, \
         subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as proc:
        for line in proc.stdout:
            log.write(line)
            log.flush()

            m = RE_HEADER.search(line)
            if m:
                current_iter  = int(m.group(1))
                current_total = int(m.group(2))
                current_label = f"combo #{m.group(3):>2s}  {m.group(4)}"
                flag = None
                continue

            m = RE_REPEAT.search(line)
            if m:
                current_iter  = int(m.group(1))
                current_total = int(m.group(2))
                current_label = "repeat"
                flag = None
                continue

            m = RE_THRESHOLD.search(line)
            if m:
                flag = m.group(1)
                continue

            m = RE_MEAN.search(line)
            if m and current_iter is not None:
                acc = float(m.group(1))
                status = flag if flag else ("OK" if acc >= 0.85 else "FAILED")
                print(f"  [{task}]  {current_iter:3d}/{current_total}"
                      f"  {current_label}  acc={acc:.4f}  [{status}]",
                      flush=True)

    elapsed = time.time() - t0
    hours, rem = divmod(int(elapsed), 3600)
    mins = rem // 60
    rc = proc.returncode
    status = "OK" if rc == 0 else f"FAILED (exit {rc})"
    print(f"\n  [{task}] finished in {hours}h{mins:02d}m — {status}  (log: {log_path})\n",
          flush=True)
    return task, status


def main():
    args = parse_args()
    log_dir = Path(args.out_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tasks:    {args.tasks}")
    print(f"Parallel: {args.parallel}")
    print(f"n_iter={args.n_iter}  h={args.h}  beta={args.beta}")
    print(f"Output:   {args.out_dir}/   logs: {args.out_dir}/<task>.log\n")

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_task, task, args, log_dir): task
                   for task in args.tasks}
        for future in as_completed(futures):
            future.result()

    print("All tasks complete.")


if __name__ == "__main__":
    main()
