#!/usr/bin/env python3
"""
Spearman stability: for each network, compute the minimum Spearman r between
the last hidden layer activations at each step checkpoint in the last N% of
training and the final checkpoint.

A high minimum r indicates the representational geometry was stable (crystallised)
well before the end of training. Distributions are shown separately for successful
and failed networks.

Output:
  output/analysis/tables/spearman_stability_{window}.csv
  output/analysis/figures/spearman_stability_{window}.pdf
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

ANALYSIS = Path(__file__).parent
REPO_ROOT = ANALYSIS.parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import TABLES_DIR, FIGURES_DIR, FINAL_DIR, RL_TASKS, TASK_NAMES, get_depth_from_config

PROD_DIR  = REPO_ROOT / "output" / "production"
RNN_TASKS = {"adding", "mnist_rnn"}

# (window_label, cutoff_fraction) — cutoff_fraction is where the window starts
WINDOWS = [
    ("99pct", 0.01),
    ("95pct", 0.05),
    ("90pct", 0.10),
    ("25pct", 0.75),
]


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


def process_network(run_dir, task, cutoff_fraction, bo_perf):
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return None
    meta       = json.load(open(meta_path))
    depth      = get_depth_from_config(meta.get("config", {}))
    final_step = int(meta.get("final_step", 0))
    if final_step == 0:
        return None

    cutoff     = final_step * cutoff_fraction
    final_npz  = run_dir / "final.npz"
    if not final_npz.exists():
        return None
    final_acts = load_last_hidden(final_npz, task, depth)
    if final_acts is None:
        return None

    window_ckpts = {}
    for npz in run_dir.glob("step_*.npz"):
        step = int(npz.stem.replace("step_", ""))
        if step >= cutoff:
            window_ckpts[step] = npz
    if not window_ckpts:
        return None

    rs = []
    for step, npz_path in sorted(window_ckpts.items()):
        acts = load_last_hidden(npz_path, task, depth)
        if acts is None or acts.shape != final_acts.shape:
            continue
        r = spearman_finite(acts.flatten(), final_acts.flatten())
        if np.isfinite(r):
            rs.append(r)
    if not rs:
        return None

    return {
        "task":           task,
        "run_id":         run_dir.name,
        "final_step":     final_step,
        "n_ckpts_window": len(rs),
        "min_spearman_r": round(min(rs), 6),
        "performance":    bo_perf.get(run_dir.name, float("nan")),
    }


def load_thresholds():
    path = TABLES_DIR / "success_thresholds.json"
    data = json.load(open(path))
    return {k: float(v["upper"]) for k, v in data.items() if isinstance(v, dict)}


def make_figure(df, thresholds, window_pct):
    tasks     = df["task"].unique()
    n_tasks   = len(tasks)
    n_cols    = 3
    n_rows    = int(np.ceil(n_tasks / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                              squeeze=False)

    bins = np.linspace(-1, 1, 201)

    for idx, task in enumerate(tasks):
        ax      = axes[idx // n_cols][idx % n_cols]
        tdf     = df[df["task"] == task]
        thresh  = thresholds.get(task)

        if thresh is not None:
            success = tdf[tdf["performance"] >= thresh]["min_spearman_r"]
            failed  = tdf[tdf["performance"] <  thresh]["min_spearman_r"]
        else:
            success = tdf["min_spearman_r"]
            failed  = pd.Series([], dtype=float)

        if len(failed) > 0:
            ax.hist(failed,  bins=bins, color="#d73027", alpha=0.7,
                    label=f"failed (n={len(failed)})",  density=True)
        if len(success) > 0:
            ax.hist(success, bins=bins, color="#2166ac", alpha=0.7,
                    label=f"success (n={len(success)})", density=True)

        avg_ckpts = tdf["n_ckpts_window"].mean()
        ax.set_title(f"{task}\n(avg {avg_ckpts:.1f} ckpts in window)",
                     fontsize=8, fontweight="bold")
        ax.set_xlabel("Min Spearman r (checkpoint vs. final)", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.set_xlim(-1, 1)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
        ax.axvline(0, color="black", lw=0.5, ls="--")

    for idx in range(n_tasks, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(
        f"Spearman stability — last {window_pct}% of training "
        f"(min r across step checkpoints vs. final)",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Only recompute these tasks, upserting into existing CSVs.")
    args = parser.parse_args()
    update_tasks = set(args.tasks) if args.tasks else None

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds()

    # Load run dirs for tasks to process
    tasks_to_run = [t for t in TASK_NAMES if update_tasks is None or t in update_tasks]
    task_run_dirs = {}
    for task in tasks_to_run:
        task_dir = PROD_DIR / task
        if not task_dir.exists():
            print(f"  {task}: not found, skipping")
            continue
        task_run_dirs[task] = sorted(p for p in task_dir.iterdir()
                                     if p.is_dir() and p.name.endswith("_r0"))

    for window_label, cutoff_fraction in WINDOWS:
        window_pct = int(round((1 - cutoff_fraction) * 100))
        print(f"\n=== Window: last {window_pct}% (cutoff at {int(cutoff_fraction*100)}%) ===",
              flush=True)
        rows = []
        for task, run_dirs in task_run_dirs.items():
            print(f"  {task} ...", flush=True)
            bo_perf = load_bo_perf(task)
            for run_dir in run_dirs:
                result = process_network(run_dir, task, cutoff_fraction, bo_perf)
                if result is not None:
                    rows.append(result)

        csv_out = TABLES_DIR / f"spearman_stability_{window_label}.csv"
        df = pd.DataFrame(rows)
        if update_tasks and csv_out.exists():
            old = pd.read_csv(csv_out)
            old = old[~old["task"].isin(update_tasks)]
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(csv_out, index=False)
        print(f"  Saved: {csv_out.name}")

        fig = make_figure(df, thresholds, window_pct)
        fig_out = FIGURES_DIR / f"spearman_stability_{window_label}.pdf"
        fig.savefig(fig_out, bbox_inches="tight", dpi=130)
        ss_final = FINAL_DIR / "learning_dynamics/figures/spearman_stability"
        ss_final.mkdir(parents=True, exist_ok=True)
        fig.savefig(ss_final / f"spearman_stability_{window_label}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fig_out.name}")


if __name__ == "__main__":
    main()
