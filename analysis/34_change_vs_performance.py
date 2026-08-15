#!/usr/bin/env python3
"""
Representational change vs. performance (Finding #3.3).

For each consecutive pair of activation checkpoints, computes
(1 - Spearman_r) / Δperf, where Δperf is the difference in normalised
performance between the two checkpoints.

Checkpoint sources (in order of performance):
  perf_*.npz  — snapshots saved during training at specific performance
                thresholds; performance read from filename / history.json
  best.npz    — peak-validation checkpoint (supervised/RNN tasks only);
                performance = max normalised val perf from history.json
  final.npz   — end-of-training checkpoint (RL tasks, no best.npz);
                performance = normalised BO-observed performance

X-axis: normalised performance at end of interval (p1).
Y-axis: (1 − Spearman r) / Δperf (representational change per unit
        performance gain).

Output:
  output/analysis/tables/change_vs_performance.csv
  output/analysis/figures/change_vs_performance.pdf
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
from analysis_utils import TABLES_DIR, FIGURES_DIR, FINAL_DIR, RL_TASKS, TASK_NAMES, task_meta, get_depth_from_config

PROD_DIR  = REPO_ROOT / "output" / "production"
RNN_TASKS = {"adding", "mnist_rnn"}


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
    state_path = PROD_DIR / task / "bo_state.json"
    observations = json.load(open(state_path))
    return {f"run_{obs['iteration']:04d}_r0": obs["performance"]
            for obs in observations}


def perf_threshold(npz_path):
    return float(npz_path.stem.replace("perf_", "").replace("p", "."))


def _norm_perf_from_entry(entry, tm):
    use_mse = tm["metric_name"] == "val_mse"
    raw = -entry["val_loss"] if use_mse else entry["val_acc"]
    return (raw - tm["chance_perf"]) / (tm["max_metric"] - tm["chance_perf"])


def actual_perf_for_supervised(run_dir, task, perf_npzs):
    """
    Return {npz_path: actual_norm_perf} by finding the first epoch in
    history.json where normalised performance >= threshold.
    Falls back to the filename threshold if history is missing or the
    epoch cannot be found.
    """
    h_path = run_dir / "history.json"
    if not h_path.exists():
        return {npz: perf_threshold(npz) for npz in perf_npzs}

    history = json.load(open(h_path))
    if not history:
        return {npz: perf_threshold(npz) for npz in perf_npzs}

    tm = task_meta()[task]

    result = {}
    for npz in perf_npzs:
        threshold = perf_threshold(npz)
        actual    = threshold  # fallback
        for entry in history:
            if _norm_perf_from_entry(entry, tm) >= threshold:
                actual = _norm_perf_from_entry(entry, tm)
                break
        result[npz] = actual
    return result


def best_norm_perf_from_history(run_dir, task):
    """Return max normalised val performance seen in history.json."""
    h_path = run_dir / "history.json"
    if not h_path.exists():
        return float("nan")
    history = json.load(open(h_path))
    if not history:
        return float("nan")
    tm = task_meta()[task]
    perfs = [_norm_perf_from_entry(e, tm) for e in history
             if np.isfinite(_norm_perf_from_entry(e, tm))]
    return max(perfs) if perfs else float("nan")


def normalise_bo_perf(raw_perf, task):
    tm = task_meta()[task]
    return (raw_perf - tm["chance_perf"]) / (tm["max_metric"] - tm["chance_perf"])


def process_network(run_dir, task, bo_perf):
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return None
    meta  = json.load(open(meta_path))
    depth = get_depth_from_config(meta.get("config", {}))

    perf_npzs = sorted(run_dir.glob("perf_*.npz"), key=perf_threshold)
    if len(perf_npzs) < 1:
        return None

    # Resolve actual normalised performance at each checkpoint
    if task in RL_TASKS:
        actual_perf = {npz: perf_threshold(npz) for npz in perf_npzs}
    else:
        actual_perf = actual_perf_for_supervised(run_dir, task, perf_npzs)

    acts = {}
    for npz in perf_npzs:
        a = load_last_hidden(npz, task, depth)
        if a is not None:
            acts[npz] = (actual_perf[npz], a)

    # Add best checkpoint (supervised/RNN) or final checkpoint (RL) as endpoint
    best_path  = run_dir / "best.npz"
    final_path = run_dir / "final.npz"
    if best_path.exists() and task not in RL_TASKS:
        a_best = load_last_hidden(best_path, task, depth)
        if a_best is not None:
            p_best = best_norm_perf_from_history(run_dir, task)
            if np.isfinite(p_best):
                acts[best_path] = (p_best, a_best)
    elif final_path.exists() and task in RL_TASKS:
        a_final = load_last_hidden(final_path, task, depth)
        if a_final is not None:
            raw_perf = bo_perf.get(run_dir.name, float("nan"))
            p_final  = normalise_bo_perf(raw_perf, task)
            if np.isfinite(p_final):
                acts[final_path] = (p_final, a_final)

    # Sort by actual performance
    sorted_npzs = sorted(acts.keys(), key=lambda p: acts[p][0])
    if len(sorted_npzs) < 2:
        return None

    perf = bo_perf.get(run_dir.name, float("nan"))
    rows = []
    for i in range(len(sorted_npzs) - 1):
        n0, n1       = sorted_npzs[i], sorted_npzs[i + 1]
        p0, a0       = acts[n0]
        p1, a1       = acts[n1]
        if a0.shape != a1.shape:
            continue
        r = spearman_finite(a0.flatten(), a1.flatten())
        if not np.isfinite(r):
            continue
        delta_p = p1 - p0
        if delta_p < 0.01:
            continue
        rows.append({
            "task":        task,
            "run_id":      run_dir.name,
            "p_start":     round(p0, 6),
            "p_end":       round(p1, 6),
            "one_minus_r": round((1 - r) / delta_p, 6),
            "performance": perf,
        })
    return rows


def load_thresholds():
    data = json.load(open(TABLES_DIR / "success_thresholds.json"))
    return {k: float(v["upper"]) for k, v in data.items() if isinstance(v, dict)}


def make_figure(df, thresholds):
    tasks  = [t for t in TASK_NAMES if t in df["task"].values]
    n_cols = 3
    n_rows = int(np.ceil(len(tasks) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    bin_edges = np.linspace(0, 1, 11)  # 10 uniform bins
    rng = np.random.default_rng(42)

    for idx, task in enumerate(tasks):
        ax     = axes[idx // n_cols][idx % n_cols]
        tdf    = df[df["task"] == task]
        thresh = thresholds.get(task)

        groups = {}
        if thresh is not None:
            groups["successful"] = ("#2166ac", tdf[tdf["performance"] >= thresh])
            groups["failed"]     = ("#d73027", tdf[tdf["performance"] <  thresh])
        else:
            groups["all"] = ("#2166ac", tdf)

        for label, (color, gdf) in groups.items():
            if gdf.empty:
                continue

            all_run_ids = gdf["run_id"].unique()
            sample_ids  = rng.choice(all_run_ids,
                                     size=min(100, len(all_run_ids)),
                                     replace=False)
            for rid in sample_ids:
                net = gdf[gdf["run_id"] == rid].sort_values("p_end")
                ax.plot(net["p_end"], net["one_minus_r"],
                        color=color, linewidth=0.6, alpha=0.06, zorder=1)

            gdf = gdf.copy()
            gdf["bin"] = pd.cut(gdf["p_end"], bins=bin_edges, labels=False,
                                include_lowest=True)
            n_bins = len(bin_edges) - 1
            means, xs = [], []
            for b, bdf in gdf.groupby("bin"):
                vals = bdf["one_minus_r"].values
                means.append(np.mean(vals))
                x = bin_edges[int(b) + 1] if int(b) == n_bins - 1 else bdf["p_end"].mean()
                xs.append(x)

            n_nets = gdf["run_id"].nunique()
            ax.plot(xs, means, color=color, linewidth=2.25, zorder=3,
                    label=f"{label} (n={n_nets})")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 10)
        ax.set_xlabel("Normalised performance", fontsize=7)
        ax.set_ylabel("(1 − Spearman r) / Δperf", fontsize=7)
        ax.set_title(task, fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

    for idx in range(len(tasks), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(
        "Representational change per unit performance gain\n"
        "Mean across networks (100 individual curves shown per group)",
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
            if result:
                rows.extend(result)

    df = pd.DataFrame(rows)

    csv_out = TABLES_DIR / "change_vs_performance.csv"
    if update_tasks and csv_out.exists():
        old = pd.read_csv(csv_out)
        old = old[~old["task"].isin(update_tasks)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")

    fig = make_figure(df, thresholds)
    fig_out = FIGURES_DIR / "change_vs_performance.pdf"
    fig.savefig(fig_out, bbox_inches="tight", dpi=130)
    cv_final = FINAL_DIR / "learning_dynamics/figures/change_vs_performance"
    cv_final.mkdir(parents=True, exist_ok=True)
    fig.savefig(cv_final / "change_vs_performance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_out}")


if __name__ == "__main__":
    main()
