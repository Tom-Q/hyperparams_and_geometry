#!/usr/bin/env python3
"""
Crystallisation time: for each network, find the earliest step checkpoint from
which Spearman r with the final checkpoint stays >= 0.99 continuously through
to the end of training. Report this as a proportion of total training steps.

Plots an ECDF per task, split by successful vs. failed networks.

Output:
  output/analysis/tables/crystallisation_time.csv
  output/analysis/figures/crystallisation_time.pdf
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ANALYSIS  = Path(__file__).parent
REPO_ROOT = ANALYSIS.parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import TABLES_DIR, FIGURES_DIR, RL_TASKS, TASK_NAMES, get_depth_from_config

PROD_DIR      = REPO_ROOT / "output" / "production"
RNN_TASKS     = {"adding", "mnist_rnn"}
THRESHOLD     = 0.90


def load_last_hidden(npz_path, task, depth):
    try:
        data = np.load(npz_path)
    except Exception:
        return None
    if task in RNN_TASKS:
        keys = sorted(k for k in data.files if k.startswith("layer_0_t_"))
        if not keys:
            return None
        return np.concatenate([data[k] for k in keys], axis=1).astype(np.float64)
    else:
        arr = data.get(f"layer_{depth - 1}")
        return arr.astype(np.float64) if arr is not None else None


def spearman_finite(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return float("nan")
    return float(spearmanr(a[mask], b[mask]).statistic)


def load_bo_perf(task):
    """Return {run_id: performance} from bo_state.json (canonical source)."""
    state_path = PROD_DIR / task / "bo_state.json"
    observations = json.load(open(state_path))
    return {f"run_{obs['iteration']:04d}_r0": obs["performance"]
            for obs in observations}


def process_network(run_dir, task, bo_perf):
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return None
    meta       = json.load(open(meta_path))
    depth      = get_depth_from_config(meta.get("config", {}))
    final_step = int(meta.get("final_step", 0))
    if final_step == 0:
        return None

    final_npz  = run_dir / "final.npz"
    if not final_npz.exists():
        return None
    final_acts = load_last_hidden(final_npz, task, depth)
    if final_acts is None:
        return None

    # Collect all step checkpoints, sorted ascending
    step_ckpts = {}
    for npz in run_dir.glob("step_*.npz"):
        step = int(npz.stem.replace("step_", ""))
        step_ckpts[step] = npz

    if not step_ckpts:
        return None

    # Compute r at each step checkpoint against final
    rs = {}
    for step, npz_path in sorted(step_ckpts.items()):
        acts = load_last_hidden(npz_path, task, depth)
        if acts is None or acts.shape != final_acts.shape:
            continue
        r = spearman_finite(acts.flatten(), final_acts.flatten())
        if np.isfinite(r):
            rs[step] = r

    if not rs:
        return None

    # Include final checkpoint itself (r = 1.0 by definition)
    rs[final_step] = 1.0

    sorted_steps = sorted(rs.keys())

    # Find the latest step where r < threshold; crystallisation is the next step
    last_below = None
    for step in sorted_steps:
        if rs[step] < THRESHOLD:
            last_below = step

    if last_below is None:
        # All checkpoints including step 1 are >= threshold
        crystallisation_step = sorted_steps[0]
    else:
        idx = sorted_steps.index(last_below)
        crystallisation_step = sorted_steps[idx + 1]  # always exists since final = 1.0

    crystallisation_fraction = crystallisation_step / final_step

    return {
        "task":                    task,
        "run_id":                  run_dir.name,
        "final_step":              final_step,
        "crystallisation_step":    crystallisation_step,
        "crystallisation_fraction": round(crystallisation_fraction, 6),
        "performance":             bo_perf.get(run_dir.name, float("nan")),
    }


def load_thresholds():
    data = json.load(open(TABLES_DIR / "success_thresholds.json"))
    return {k: float(v["upper"]) for k, v in data.items() if isinstance(v, dict)}


def ecdf(values):
    xs = np.sort(values)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    # Prepend 0 so the curve starts from the left
    xs = np.concatenate([[0], xs])
    ys = np.concatenate([[0], ys])
    return xs, ys


def make_figure(df, thresholds):
    tasks   = [t for t in TASK_NAMES if t in df["task"].values]
    n_tasks = len(tasks)
    n_cols  = 3
    n_rows  = int(np.ceil(n_tasks / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                              squeeze=False)

    for idx, task in enumerate(tasks):
        ax     = axes[idx // n_cols][idx % n_cols]
        tdf    = df[df["task"] == task]
        thresh = thresholds.get(task)

        if thresh is not None:
            success = tdf[tdf["performance"] >= thresh]["crystallisation_fraction"].values
            failed  = tdf[tdf["performance"] <  thresh]["crystallisation_fraction"].values
        else:
            success = tdf["crystallisation_fraction"].values
            failed  = np.array([])

        if len(success) > 0:
            xs, ys = ecdf(success)
            ax.step(xs, ys, where="post", color="#2166ac", linewidth=1.2,
                    label=f"successful (n={len(success)})")

        if len(failed) > 0:
            xs, ys = ecdf(failed)
            ax.step(xs, ys, where="post", color="#d73027", linewidth=1.2,
                    label=f"failed (n={len(failed)})")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Crystallisation time (fraction of total training)", fontsize=7)
        ax.set_ylabel("Proportion of networks", fontsize=7)
        ax.set_title(task, fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
        ax.axvline(1.0, color="grey", lw=0.5, ls="--")

    for idx in range(n_tasks, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(
        f"Crystallisation time — earliest point at which Spearman r with final "
        f"checkpoint stays ≥ {THRESHOLD}\n"
        f"(as proportion of total training steps)",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Only recompute these tasks, upserting into existing CSV.")
    args = parser.parse_args()
    update_tasks = set(args.tasks) if args.tasks else None

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds()
    rows = []

    for task in (t for t in TASK_NAMES if update_tasks is None or t in update_tasks):
        task_dir = PROD_DIR / task
        if not task_dir.exists():
            print(f"  {task}: not found, skipping")
            continue
        run_dirs = sorted(p for p in task_dir.iterdir()
                          if p.is_dir() and p.name.endswith("_r0"))
        bo_perf = load_bo_perf(task)
        print(f"{task}: {len(run_dirs)} networks ...", flush=True)
        for run_dir in run_dirs:
            result = process_network(run_dir, task, bo_perf)
            if result is not None:
                rows.append(result)

    df = pd.DataFrame(rows)

    csv_out = TABLES_DIR / "crystallisation_time.csv"
    if update_tasks and csv_out.exists():
        old = pd.read_csv(csv_out)
        old = old[~old["task"].isin(update_tasks)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")

    fig = make_figure(df, thresholds)
    fig_out = FIGURES_DIR / "crystallisation_time.pdf"
    fig.savefig(fig_out, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"Saved: {fig_out}")


if __name__ == "__main__":
    main()
