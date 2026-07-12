#!/usr/bin/env python3
"""
Step 15: Layer comparison — Finding #1.4.

For depth=2 networks, compares layer_0 (first hidden, closer to input) and
layer_1 (second hidden, closer to output) on three metrics:

  1. Category structure: Spearman r with each category model per layer.
     Key question: does layer_1 reflect output/task categories more than layer_0,
     which should be closer to raw input structure?

  2. Noise ceiling: LOO inter-network agreement per layer.
     Is layer_1 more reliable (higher noise ceiling)?

  3. Within-network layer correlation: Spearman(layer_0, layer_1) for each network.
     How much do the two layers' RDMs agree within the same network?

Focal tasks: mnist_dual, spirals, fourrooms  (as specified)
All non-RNN tasks: computed and shown in a summary panel.

Outputs:
    output/analysis/figures/f1_layer_comparison.pdf
    output/analysis/tables/rdm_layer_comparison.csv
"""

import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ANALYSIS = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import (
    CACHE_DIR, FIGURES_DIR, RDM_DIR, TABLES_DIR, TASK_NAMES, RL_TASKS, task_meta,
)

MODELS_DIR = CACHE_DIR / "category_models"
RNN_TASKS  = {"adding", "mnist_rnn"}
FOCAL_TASKS = ["mnist_dual", "spirals", "fourrooms"]

# Ordered category models per task (input-like → output-like, left to right)
TASK_MODELS_ORDERED = {
    "mnist_dual":    ["digit", "mixed", "output"],
    "mnist_10way":   ["digit"],
    "fashion_10way": ["class"],
    "spirals":       ["spatial", "arm"],
    "parity":        ["hamming_diff", "parity_label"],
    "cartpole":      ["euclidean", "angle_diff"],
    "fourrooms":     ["euclidean", "goal_dist", "room"],
}

# Colour per model (consistent with earlier figures)
MODEL_COLORS = {
    "digit":        "#888888",
    "output":       "#2166ac",
    "mixed":        "#4dac26",
    "arm":          "#2166ac",
    "spatial":      "#d6604d",
    "parity_label": "#2166ac",
    "hamming_diff": "#d6604d",
    "class":        "#2166ac",
    "euclidean":    "#d6604d",
    "angle_diff":   "#888888",
    "room":         "#2166ac",
    "goal_dist":    "#4dac26",
}

LAYER_COLORS = {0: "#4393c3", 1: "#d6604d"}
LAYER_LABELS = {0: "layer 0 (input-side)", 1: "layer 1 (output-side)"}

TASK_LABELS = {
    "mnist_dual":    "MNIST dual",
    "mnist_10way":   "MNIST 10-way",
    "fashion_10way": "Fashion 10-way",
    "spirals":       "Spirals",
    "parity":        "Parity",
    "cartpole":      "CartPole",
    "fourrooms":     "FourRooms",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_thresholds():
    path = TABLES_DIR / "success_thresholds.json"
    if not path.exists():
        return {}
    data = json.load(open(path))
    return {k: (float(v["upper"]) if isinstance(v, dict) else None)
            for k, v in data.items() if k != "_alpha"}


def noise_ceiling_loo(rdm_matrix):
    N = len(rdm_matrix)
    if N < 3:
        return np.full(N, np.nan)
    rf = rdm_matrix.astype(np.float64)
    total = rf.sum(axis=0)
    out = np.empty(N)
    for i in range(N):
        loo = (total - rf[i]) / (N - 1)
        out[i] = spearmanr(rf[i], loo)[0]
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_depth2_pairs(task, threshold):
    """
    Load primary depth=2 networks with both layer_0 and layer_1 RDMs.
    Returns list of dicts with run_id, perf, l0_vec, l1_vec.
    """
    h5_path = RDM_DIR / f"{task}_rdms.h5"
    ckpt    = "final" if task in RL_TASKS else "best"
    results = []

    with h5py.File(h5_path, "r") as h5:
        runs_grp = h5.get("runs", {})
        for run_id in sorted(runs_grp.keys()):
            rg = runs_grp[run_id]
            if bool(rg.attrs.get("is_repeat", False)):
                continue
            if int(rg.attrs.get("hp_depth", 1)) != 2:
                continue
            perf     = float(rg.attrs.get("performance", float("nan")))
            ckpt_grp = rg.get(ckpt)
            if ckpt_grp is None:
                continue
            ds0 = ckpt_grp.get("layer_0")
            ds1 = ckpt_grp.get("layer_1")
            if ds0 is None or ds1 is None:
                continue
            if (ds0.attrs.get("degenerate", False) or len(ds0) == 0 or
                    ds1.attrs.get("degenerate", False) or len(ds1) == 0):
                continue
            results.append({
                "run_id":    run_id,
                "perf":      perf,
                "l0_vec":    ds0[:].astype(np.float32),
                "l1_vec":    ds1[:].astype(np.float32),
                "successful": (threshold is None or perf >= threshold),
            })

    return results


def load_cat_vecs(task):
    """Load category model RDMs as upper-triangle vectors."""
    npz_path = MODELS_DIR / f"{task}.npz"
    if not npz_path.exists():
        return {}
    data = np.load(npz_path)
    cat_vecs = {}
    for name in data.files:
        D = data[name]
        n = D.shape[0]
        ri, ci = np.triu_indices(n, k=1)
        cat_vecs[name] = D[ri, ci].astype(np.float32)
    return cat_vecs


# ---------------------------------------------------------------------------
# Per-task analysis
# ---------------------------------------------------------------------------

def analyse_task(task, pairs, cat_vecs, ordered_models):
    """
    Compute per-network metrics and noise ceiling for one task.

    Returns:
        rows        : list of per-network metric dicts
        nc_l0       : LOO correlations for successful L0
        nc_l1       : LOO correlations for successful L1
        within_corrs: Spearman(l0, l1) per network (all depth=2 primaries)
    """
    rows = []
    within_corrs = []

    for p in pairs:
        within_r = spearmanr(p["l0_vec"], p["l1_vec"])[0]
        within_corrs.append(float(within_r))

        row = {
            "task":        task,
            "run_id":      p["run_id"],
            "performance": p["perf"],
            "successful":  p["successful"],
            "within_corr": float(within_r),
        }

        for model_name, cat_vec in cat_vecs.items():
            r_l0 = spearmanr(p["l0_vec"], cat_vec)[0] if np.isfinite(p["l0_vec"]).all() else np.nan
            r_l1 = spearmanr(p["l1_vec"], cat_vec)[0] if np.isfinite(p["l1_vec"]).all() else np.nan
            row[f"l0_{model_name}"] = float(r_l0) if np.isfinite(r_l0) else float("nan")
            row[f"l1_{model_name}"] = float(r_l1) if np.isfinite(r_l1) else float("nan")

        rows.append(row)

    # Noise ceiling among successful networks only
    succ = [p for p in pairs if p["successful"]]
    if len(succ) >= 3:
        nc_l0 = noise_ceiling_loo(np.array([p["l0_vec"] for p in succ], dtype=np.float32))
        nc_l1 = noise_ceiling_loo(np.array([p["l1_vec"] for p in succ], dtype=np.float32))
    else:
        nc_l0 = nc_l1 = np.array([])

    return rows, nc_l0, nc_l1, np.array(within_corrs)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _bar_pairs(ax, models, l0_medians, l0_q25, l0_q75, l1_medians, l1_q25, l1_q75, n_nets):
    """Grouped bar chart: layer_0 vs layer_1 per model."""
    x = np.arange(len(models))
    w = 0.35
    for i, (mname, m0, lo0, hi0, m1, lo1, hi1) in enumerate(
            zip(models, l0_medians, l0_q25, l0_q75, l1_medians, l1_q25, l1_q75)):
        col0 = LAYER_COLORS[0]
        col1 = LAYER_COLORS[1]
        ax.bar(i - w/2, m0, w, yerr=[[m0-lo0], [hi0-m0]], color=col0,
               capsize=3, alpha=0.85,
               label="layer 0" if i == 0 else "")
        ax.bar(i + w/2, m1, w, yerr=[[m1-lo1], [hi1-m1]], color=col1,
               capsize=3, alpha=0.85,
               label="layer 1" if i == 0 else "")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=8)
    ax.set_ylabel("Spearman r (cat. model)", fontsize=8)
    ax.set_ylim(-0.1, 0.85)
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.legend(fontsize=7)
    ax.text(0.98, 0.97, f"n={n_nets}", transform=ax.transAxes,
            fontsize=7, ha="right", va="top", color="grey")


def _nc_violin(ax, nc_l0, nc_l1):
    """Paired violin: noise ceiling for layer 0 vs layer 1."""
    data = [d for d in [nc_l0, nc_l1] if len(d) > 0]
    pos  = [i for i, d in enumerate([nc_l0, nc_l1]) if len(d) > 0]
    if not data:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    parts = ax.violinplot(data, positions=pos, showmedians=True, showextrema=True)
    for body, L in zip(parts["bodies"], pos):
        body.set_facecolor(LAYER_COLORS[L])
        body.set_alpha(0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["layer 0", "layer 1"], fontsize=8)
    ax.set_ylabel("LOO Spearman r", fontsize=8)
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0, color="grey", lw=0.6, ls="--")


def _within_violin(ax, within_corrs):
    """Violin of within-network Spearman(L0, L1) correlation."""
    valid = within_corrs[np.isfinite(within_corrs)]
    if len(valid) < 3:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    parts = ax.violinplot([valid], positions=[0], showmedians=True, showextrema=True)
    parts["bodies"][0].set_facecolor("#4dac26")
    parts["bodies"][0].set_alpha(0.6)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xticks([])
    ax.set_ylabel("Spearman r (L0 ↔ L1)", fontsize=8)
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.text(0.5, 0.03, f"med={np.median(valid):.3f}  n={len(valid)}",
            ha="center", va="bottom", transform=ax.transAxes, fontsize=7, color="grey")


def make_focal_figure(results_by_task):
    """3×3 grid: focal tasks × (category structure | noise ceiling | within-network corr)."""
    fig, axes = plt.subplots(len(FOCAL_TASKS), 3, figsize=(14, 4.5 * len(FOCAL_TASKS)))

    for row, task in enumerate(FOCAL_TASKS):
        if task not in results_by_task:
            for ax in axes[row]:
                ax.set_visible(False)
            continue

        res = results_by_task[task]
        df  = pd.DataFrame(res["rows"])
        nc_l0 = res["nc_l0"]
        nc_l1 = res["nc_l1"]
        within = res["within_corrs"]
        ordered = res["ordered_models"]

        ax_cat, ax_nc, ax_wc = axes[row]

        # Panel 1: category structure bars (successful networks)
        succ_df = df[df["successful"]] if "successful" in df.columns else df
        n_nets  = len(succ_df)
        l0_meds, l0_q25, l0_q75 = [], [], []
        l1_meds, l1_q25, l1_q75 = [], [], []
        for m in ordered:
            c0 = succ_df.get(f"l0_{m}", pd.Series(dtype=float)).dropna()
            c1 = succ_df.get(f"l1_{m}", pd.Series(dtype=float)).dropna()
            l0_meds.append(c0.median() if len(c0) else np.nan)
            l0_q25.append(np.percentile(c0, 25) if len(c0) else np.nan)
            l0_q75.append(np.percentile(c0, 75) if len(c0) else np.nan)
            l1_meds.append(c1.median() if len(c1) else np.nan)
            l1_q25.append(np.percentile(c1, 25) if len(c1) else np.nan)
            l1_q75.append(np.percentile(c1, 75) if len(c1) else np.nan)

        _bar_pairs(ax_cat, ordered,
                   l0_meds, l0_q25, l0_q75,
                   l1_meds, l1_q25, l1_q75,
                   n_nets)
        ax_cat.set_title(f"{TASK_LABELS.get(task, task)} — category structure",
                         fontsize=9, fontweight="bold")

        # Panel 2: noise ceiling
        _nc_violin(ax_nc, nc_l0, nc_l1)
        ax_nc.set_title(f"Noise ceiling (n={len(nc_l0)}+{len(nc_l1)})", fontsize=9)

        # Panel 3: within-network layer correlation
        _within_violin(ax_wc, within)
        ax_wc.set_title("Within-network layer corr.", fontsize=9)

    fig.suptitle(
        "Layer comparison (depth=2 networks) — layer 0 vs. layer 1\n"
        "Category structure (successful only) | Noise ceiling | Within-network correlation",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def make_summary_figure(results_by_task, all_tasks):
    """Compact summary: noise ceiling L0 vs L1 for all non-RNN tasks."""
    tasks_with_data = [t for t in all_tasks if t in results_by_task and
                       len(results_by_task[t]["nc_l0"]) > 0]
    n = len(tasks_with_data)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks_with_data):
        res   = results_by_task[task]
        nc_l0 = res["nc_l0"]
        nc_l1 = res["nc_l1"]
        _nc_violin(ax, nc_l0, nc_l1)
        ax.set_title(TASK_LABELS.get(task, task), fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel("LOO Spearman r", fontsize=9)
    fig.suptitle("Noise ceiling: layer 0 vs. layer 1 (depth=2 successful networks)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds()
    all_tasks  = [t for t in TASK_NAMES if t not in RNN_TASKS]

    results_by_task = {}
    all_rows = []

    for task in all_tasks:
        threshold = thresholds.get(task)
        print(f"  {task} ...", end="", flush=True)

        pairs = load_depth2_pairs(task, threshold)
        if not pairs:
            print(" [no depth=2 networks]")
            continue

        cat_vecs = load_cat_vecs(task)
        ordered  = [m for m in TASK_MODELS_ORDERED.get(task, []) if m in cat_vecs]

        rows, nc_l0, nc_l1, within = analyse_task(task, pairs, cat_vecs, ordered)

        print(f" {len(pairs)} depth=2 nets, {sum(p['successful'] for p in pairs)} successful, "
              f"NC L0={np.nanmedian(nc_l0):.3f} L1={np.nanmedian(nc_l1):.3f} "
              f"within={np.nanmedian(within):.3f}")

        results_by_task[task] = {
            "rows":           rows,
            "nc_l0":          nc_l0,
            "nc_l1":          nc_l1,
            "within_corrs":   within,
            "ordered_models": ordered,
        }
        all_rows.extend(rows)

    # Save CSV
    csv_path = TABLES_DIR / "rdm_layer_comparison.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Focal figure
    fig_focal = make_focal_figure(results_by_task)
    out_focal = FIGURES_DIR / "f1_layer_comparison.pdf"
    fig_focal.savefig(out_focal, bbox_inches="tight", dpi=150)
    plt.close(fig_focal)
    print(f"Saved: {out_focal}")

    # Summary figure
    fig_sum = make_summary_figure(results_by_task, all_tasks)
    if fig_sum is not None:
        out_sum = FIGURES_DIR / "f1_layer_comparison_summary.pdf"
        fig_sum.savefig(out_sum, bbox_inches="tight", dpi=150)
        plt.close(fig_sum)
        print(f"Saved: {out_sum}")


if __name__ == "__main__":
    main()
