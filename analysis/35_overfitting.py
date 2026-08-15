#!/usr/bin/env python3
"""
Overfitting and representational change (Finding #3.x).

For each network, compares the best-checkpoint RDM to the final-checkpoint RDM:
  - rdm_dissimilarity: 1 - Spearman_r(best activations, final activations)
  - overfit_delta:     norm_perf(best) - norm_perf(final)
                       (positive = validation performance degraded after best)

For supervised tasks: overfit_delta is derived from history.json.
For RL tasks:         overfit_delta is derived from best_metric / final_metric
                      in metadata.json.

Networks where best_step == final_step are excluded (training stopped at peak).

Output:
  output/analysis/tables/overfitting.csv
  output/analysis/figures/overfitting.pdf
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


def overfit_delta_supervised(run_dir, task, best_step):
    h_path = run_dir / "history.json"
    if not h_path.exists():
        return float("nan")
    history = json.load(open(h_path))
    if not history:
        return float("nan")

    tm      = task_meta()[task]
    chance  = tm["chance_perf"]
    max_m   = tm["max_metric"]
    use_mse = tm["metric_name"] == "val_mse"

    def norm(entry):
        raw = -entry["val_loss"] if use_mse else entry["val_acc"]
        return (raw - chance) / (max_m - chance)

    best_entry  = min(history, key=lambda e: abs(e["step"] - best_step))
    final_entry = history[-1]
    return norm(best_entry) - norm(final_entry)


def overfit_delta_rl(meta, task):
    best_metric  = meta.get("best_metric")
    final_metric = meta.get("final_metric")
    if best_metric is None or final_metric is None:
        return float("nan")
    tm     = task_meta()[task]
    chance = tm["chance_perf"]
    max_m  = tm["max_metric"]
    norm   = lambda v: (v - chance) / (max_m - chance)
    return norm(best_metric) - norm(final_metric)


def process_network(run_dir, task, bo_perf):
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return None
    meta       = json.load(open(meta_path))
    depth      = get_depth_from_config(meta.get("config", {}))
    best_step  = meta.get("best_step")
    final_step = meta.get("final_step")
    if not best_step or not final_step:
        return None
    best_step  = int(best_step)
    final_step = int(final_step)

    if best_step == final_step:
        return None

    best_npz  = run_dir / "best.npz"
    final_npz = run_dir / "final.npz"
    if not best_npz.exists() or not final_npz.exists():
        return None

    a_best  = load_last_hidden(best_npz,  task, depth)
    a_final = load_last_hidden(final_npz, task, depth)
    if a_best is None or a_final is None or a_best.shape != a_final.shape:
        return None

    r = spearman_finite(a_best.flatten(), a_final.flatten())
    if not np.isfinite(r):
        return None

    delta = (overfit_delta_rl(meta, task) if task in RL_TASKS
             else overfit_delta_supervised(run_dir, task, best_step))

    return {
        "task":             task,
        "run_id":           run_dir.name,
        "rdm_dissimilarity": round(1 - r, 6),
        "overfit_delta":    round(float(delta), 6) if np.isfinite(delta) else float("nan"),
        "performance":      bo_perf.get(run_dir.name, float("nan")),
        "best_step":        best_step,
        "final_step":       final_step,
    }


def load_thresholds():
    data = json.load(open(TABLES_DIR / "success_thresholds.json"))
    return {k: float(v["upper"]) for k, v in data.items() if isinstance(v, dict)}


def make_figure(df, thresholds):
    tasks  = [t for t in TASK_NAMES if t in df["task"].values]
    n_cols = 3
    n_rows = int(np.ceil(len(tasks) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for idx, task in enumerate(tasks):
        ax     = axes[idx // n_cols][idx % n_cols]
        tdf    = df[df["task"] == task].dropna(subset=["overfit_delta", "rdm_dissimilarity"])
        thresh = thresholds.get(task)

        gdf = tdf[tdf["performance"] >= thresh] if thresh is not None else tdf
        if gdf.empty:
            ax.set_visible(False)
            continue

        ax.scatter(gdf["overfit_delta"], gdf["rdm_dissimilarity"],
                   color="#2166ac", alpha=0.3, s=4, zorder=2,
                   label=f"successful (n={len(gdf)})")

        # Spearman correlation + linear fit
        r, p = spearmanr(gdf["overfit_delta"], gdf["rdm_dissimilarity"])
        m, b = np.polyfit(gdf["overfit_delta"], gdf["rdm_dissimilarity"], 1)
        x_line = np.array([gdf["overfit_delta"].min(), gdf["overfit_delta"].max()])
        ax.plot(x_line, m * x_line + b, color="#2166ac", linewidth=1.2, zorder=3)
        p_str = f"p<0.001" if p < 0.001 else f"p={p:.3f}"
        ax.text(0.97, 0.97, f"Spearman r={r:.2f}\n{p_str}",
                transform=ax.transAxes, fontsize=6,
                ha="right", va="top")

        ax.axvline(0, color="black", linewidth=0.5, linestyle="--", zorder=1)
        ax.set_xlabel("norm_perf(best) − norm_perf(final)", fontsize=7)
        ax.set_ylabel("1 − Spearman r (best vs final)", fontsize=7)
        ax.set_title(task, fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

    for idx in range(len(tasks), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(
        "Overfitting vs. representational change (best → final checkpoint)\n"
        "Positive x-axis = validation performance degraded after best checkpoint",
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
                rows.append(result)

    df = pd.DataFrame(rows)
    print(f"\nNetworks included: {len(df)}")

    csv_out = TABLES_DIR / "overfitting.csv"
    if update_tasks and csv_out.exists():
        old = pd.read_csv(csv_out)
        old = old[~old["task"].isin(update_tasks)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")

    fig = make_figure(df, thresholds)
    fig_out = FIGURES_DIR / "overfitting.pdf"
    fig.savefig(fig_out, bbox_inches="tight", dpi=130)
    ov_final = FINAL_DIR / "learning_dynamics/figures/overfitting"
    ov_final.mkdir(parents=True, exist_ok=True)
    fig.savefig(ov_final / "overfitting.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_out}")


if __name__ == "__main__":
    main()
