#!/usr/bin/env python3
"""
Step 14: Category structure — Finding #1.3.

For each task, computes the Spearman correlation between each primary network's
RDM and the task's category model RDM(s). Plots correlation vs. normalised
performance, colour-coded by success category.

Normalised performance: (perf − chance) / (upper_threshold − chance)
  0 = chance level, 1 = success threshold, >1 = above threshold.

Outputs:
    output/analysis/figures/f1_category_structure.pdf
    output/analysis/figures/f1_category_adding_phases.pdf    (adding phase analysis)
    output/analysis/figures/f1_category_mnist_rnn_temporal.pdf  (mnist_rnn per-timestep)
    output/analysis/tables/rdm_category_structure.csv   (per-network, reused in Finding #2)
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
    CACHE_DIR, DATASET_DIR, FIGURES_DIR, RDM_DIR, TABLES_DIR,
    TASK_NAMES, RL_TASKS, task_meta,
)

MODELS_DIR         = CACHE_DIR / "category_models"
TASK_DIR_OVERRIDES = {"adding": "adding_failed_run"}
RNN_TASKS          = {"adding", "mnist_rnn"}

# Tasks that show multiple models as overlapping series.
# Each entry: list of (model_name, marker, color, legend_label) plotted back-to-front.
MULTI_MODEL = {
    "mnist_dual": [
        ("digit",  "s", "#d6604d", "digit"),
        ("mixed",  "^", "#4dac26", "mixed"),
        ("output", "o", "#2166ac", "output"),
    ],
    "spirals": [
        ("spatial", "s", "#d6604d", "spatial dist"),
        ("arm",     "o", "#2166ac", "arm"),
    ],
    "fourrooms": [
        ("euclidean", "s", "#d6604d", "euclidean"),
        ("room",      "o", "#2166ac", "room"),
    ],
}

# Single primary model for tasks not in MULTI_MODEL
PRIMARY_MODEL = {
    "mnist_10way":   "digit",
    "fashion_10way": "class",
    "mnist_rnn":     "digit",
    "parity":        "parity_label",
    "cartpole":      "euclidean",
}

PERF_COLORS = {
    "successful":  "#2166ac",
    "partial":     "#f4a261",
    "near_chance": "#d6604d",
}

TASK_LABELS = {
    "mnist_dual":    "MNIST dual",
    "mnist_10way":   "MNIST 10-way",
    "fashion_10way": "Fashion 10-way",
    "mnist_rnn":     "MNIST RNN",
    "spirals":       "Spirals",
    "parity":        "Parity",
    "cartpole":      "CartPole",
    "fourrooms":     "FourRooms",
}


# ---------------------------------------------------------------------------
# Thresholds and performance normalisation
# ---------------------------------------------------------------------------

def load_thresholds():
    path = TABLES_DIR / "success_thresholds.json"
    data = json.load(open(path))
    result = {}
    for task, vals in data.items():
        if task != "_alpha" and isinstance(vals, dict):
            result[task] = {"upper": vals.get("upper"), "lower": vals.get("lower")}
    return result


def norm_perf(perf, chance, upper):
    if upper == chance:
        return float("nan")
    return (perf - chance) / (upper - chance)


def perf_category(perf, upper, lower):
    if perf >= upper:
        return "successful"
    if perf <= lower:
        return "near_chance"
    return "partial"


# ---------------------------------------------------------------------------
# RDM loading (last hidden layer, best/final checkpoint, all primary networks)
# ---------------------------------------------------------------------------

def get_ckpt_name(task):
    return "final" if task in RL_TASKS else "best"


def get_last_layer_key(ckpt_grp, task, depth):
    if task in RNN_TASKS:
        parsed = []
        for k in ckpt_grp.keys():
            if "_t_" not in k:
                continue
            parts = k.split("_")
            try:
                parsed.append((int(parts[1]), int(parts[3])))
            except (IndexError, ValueError):
                continue
        if not parsed:
            return None
        max_l = max(p[0] for p in parsed)
        max_t = max(p[1] for p in parsed if p[0] == max_l)
        return f"layer_{max_l}_t_{max_t}"
    return f"layer_{max(0, int(depth) - 1)}"


def load_all_primary_rdms(task):
    """Load all primary network RDMs regardless of performance. Returns list of (run_id, perf, vec)."""
    h5_path = RDM_DIR / f"{task}_rdms.h5"
    if not h5_path.exists():
        return []
    ckpt = get_ckpt_name(task)
    results = []
    with h5py.File(h5_path, "r") as h5:
        runs_grp = h5.get("runs")
        if runs_grp is None:
            return []
        for run_id in sorted(runs_grp.keys()):
            rg = runs_grp[run_id]
            if bool(rg.attrs.get("is_repeat", False)):
                continue
            perf = float(rg.attrs.get("performance", float("nan")))
            depth = int(rg.attrs.get("hp_depth", 1))
            ckpt_grp = rg.get(ckpt)
            if ckpt_grp is None:
                continue
            key = get_last_layer_key(ckpt_grp, task, depth)
            if key is None:
                continue
            ds = ckpt_grp.get(key)
            if ds is None or ds.attrs.get("degenerate", False) or len(ds) == 0:
                continue
            results.append((run_id, perf, ds[:].astype(np.float32)))
    return results


# ---------------------------------------------------------------------------
# Category model loading — upper-triangle vector
# ---------------------------------------------------------------------------

def load_cat_vec(task, model_name):
    npz_path = MODELS_DIR / f"{task}.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    if model_name not in data:
        return None
    D = data[model_name]           # N×N
    n = D.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    return D[rows, cols].astype(np.float32)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_task(ax, task, rows_df, thresholds, meta):
    if rows_df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        return

    chance = meta["chance_perf"]
    upper  = thresholds[task]["upper"]
    lower  = thresholds[task]["lower"]

    if task in MULTI_MODEL:
        for mname, marker, color, label in MULTI_MODEL[task]:
            sub = rows_df[rows_df["model_name"] == mname]
            if sub.empty:
                continue
            ax.scatter(sub["norm_perf"], sub["spearman_r"],
                       s=2, alpha=0.2, marker=marker,
                       color=color, label=label, rasterized=True)
        ax.legend(fontsize=7, markerscale=2.5, loc="upper left")
    else:
        primary = PRIMARY_MODEL.get(task)
        sub = rows_df[rows_df["model_name"] == primary]
        for cat, color in PERF_COLORS.items():
            pts = sub[sub["perf_cat"] == cat]
            ax.scatter(pts["norm_perf"], pts["spearman_r"],
                       s=2, alpha=0.2, color=color, rasterized=True)

    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.axvline(1, color="grey", lw=0.8, ls=":")   # success threshold
    ax.set_xlim(left=min(rows_df["norm_perf"].min() - 0.05, -0.1))
    ax.set_ylim(-0.15, 1.05)
    ax.set_xlabel("norm. performance", fontsize=7)
    ax.set_ylabel("Spearman r (cat. model)", fontsize=7)
    ax.set_title(TASK_LABELS.get(task, task), fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# Adding: phase-aligned category structure
# ---------------------------------------------------------------------------

ADDING_PHASE_NAMES  = ["phase_1", "phase_2", "phase_3", "phase_4", "phase_5", "phase_6"]
ADDING_PHASE_LABELS = [
    "before\nflag₁", "at\nflag₁", "between\nflags",
    "at\nflag₂", "after\nflag₂", "final",
]
_ADDING_T = 25


def _adding_phase_masks():
    """Compute (6, 100) bool mask: True where stimulus has ≥1 step in that phase."""
    sys.path.insert(0, str(ANALYSIS.parent))
    from tasks import TASKS
    task = TASKS["adding"]()
    inputs, _ = task.get_rdm_stimuli()   # (100, 25, 2)
    N = 100
    flag_pos = np.array([
        sorted(np.where(inputs[i, :, 1] > 0.5)[0].tolist())
        for i in range(N)
    ])
    f1, f2 = flag_pos[:, 0], flag_pos[:, 1]
    masks = np.zeros((6, N), dtype=bool)
    masks[0] = f1 > 0
    masks[1] = True
    masks[2] = f2 > f1 + 1
    masks[3] = True
    masks[4] = f2 < _ADDING_T - 2
    masks[5] = True
    return masks


def _load_adding_phase_rdms(success_threshold):
    """
    Load successful primary network phase RDMs from adding_rdms.h5.
    Returns dict phase_name -> list of float32 vectors, or None if none exist.
    """
    h5_path = RDM_DIR / "adding_rdms.h5"
    result  = {p: [] for p in ADDING_PHASE_NAMES}
    found   = False

    with h5py.File(h5_path, "r") as h5:
        runs_grp = h5.get("runs")
        if runs_grp is None:
            return None
        for run_id in sorted(runs_grp.keys()):
            rg = runs_grp[run_id]
            if bool(rg.attrs.get("is_repeat", False)):
                continue
            perf = float(rg.attrs.get("performance", float("nan")))
            if success_threshold is not None and perf < success_threshold:
                continue
            ckpt_grp = rg.get("best")
            if ckpt_grp is None:
                continue
            for pname in ADDING_PHASE_NAMES:
                key = f"layer_0_{pname}"
                if key not in ckpt_grp:
                    continue
                found = True
                ds = ckpt_grp[key]
                if ds.attrs.get("degenerate", False) or len(ds) == 0:
                    continue
                result[pname].append(ds[:].astype(np.float32))

    return result if found else None


def plot_adding_phases(thresholds):
    """Phase-aligned category structure figure for the Adding task."""
    npz_path = MODELS_DIR / "adding.npz"
    if not npz_path.exists():
        print("  [skip adding phases] no category models — run 12 first")
        return

    threshold = (thresholds.get("adding") or {}).get("upper")
    try:
        phase_data = _load_adding_phase_rdms(threshold)
    except BlockingIOError:
        print("  [skip adding phases] adding_rdms.h5 is locked (10b still running?)")
        return
    if phase_data is None:
        print("  [skip adding phases] phase RDMs not computed — run 10b first")
        return

    phase_masks = _adding_phase_masks()
    cat_models  = dict(np.load(npz_path))  # "value1": (100,100), "sum": (100,100)

    # Compute per-model, per-phase correlation distributions
    series = {
        "value1": ("#d6604d", "value₁"),
        "sum":    ("#2166ac", "sum"),
    }

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(ADDING_PHASE_NAMES))
    offsets = [-0.15, 0.15]

    for offset, (model_name, (color, label)) in zip(offsets, series.items()):
        full_model = cat_models[model_name]  # (100, 100)
        medians, q25, q75 = [], [], []

        for k, pname in enumerate(ADDING_PHASE_NAMES):
            mask    = phase_masks[k]
            n_valid = mask.sum()
            sub     = full_model[np.ix_(mask, mask)]
            ri, ci  = np.triu_indices(n_valid, k=1)
            model_v = sub[ri, ci].astype(np.float32)

            vecs = phase_data[pname]
            corrs = []
            for v in vecs:
                if len(v) != len(model_v):
                    continue
                r, _ = spearmanr(v, model_v)
                if np.isfinite(r):
                    corrs.append(r)

            if corrs:
                medians.append(np.median(corrs))
                q25.append(np.percentile(corrs, 25))
                q75.append(np.percentile(corrs, 75))
            else:
                medians.append(np.nan)
                q25.append(np.nan)
                q75.append(np.nan)

        xp = x + offset
        med = np.array(medians)
        lo  = np.array(q25)
        hi  = np.array(q75)
        ax.errorbar(xp, med, yerr=[med - lo, hi - med],
                    fmt="o-", color=color, lw=1.4, ms=5, capsize=3,
                    label=label)

    n_nets = len(phase_data["phase_2"])
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(ADDING_PHASE_LABELS, fontsize=9)
    ax.set_ylabel("Spearman r (cat. model)", fontsize=10)
    ax.set_ylim(-0.25, 1.0)
    ax.set_title(
        f"Adding — phase-aligned category structure\n"
        f"(median ± IQR, n≈{n_nets} successful networks)",
        fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    out = FIGURES_DIR / "f1_category_adding_phases.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# MNIST RNN: temporal category structure (all timesteps × layers)
# ---------------------------------------------------------------------------

def _load_mnist_rnn_temporal(success_threshold):
    """
    Load RDMs at every (layer, timestep) for successful primary networks.
    Returns:
      data : dict (layer, t) -> list of float32 RDM vectors
      n_layers : max layer index + 1 (1 or 2 for this dataset)
      n_t : number of timesteps
    """
    h5_path = RDM_DIR / "mnist_rnn_rdms.h5"
    data = {}
    n_t = 0
    n_layers = 0

    with h5py.File(h5_path, "r") as h5:
        runs_grp = h5.get("runs")
        if runs_grp is None:
            return {}, 0, 0
        for run_id in sorted(runs_grp.keys()):
            rg = runs_grp[run_id]
            if bool(rg.attrs.get("is_repeat", False)):
                continue
            perf = float(rg.attrs.get("performance", float("nan")))
            if success_threshold is not None and perf < success_threshold:
                continue
            ckpt_grp = rg.get("best")
            if ckpt_grp is None:
                continue
            for key in ckpt_grp.keys():
                if "_t_" not in key:
                    continue
                parts = key.split("_")
                try:
                    L = int(parts[1])
                    T = int(parts[3])
                except (IndexError, ValueError):
                    continue
                n_t     = max(n_t, T + 1)
                n_layers = max(n_layers, L + 1)
                ds = ckpt_grp[key]
                if ds.attrs.get("degenerate", False) or len(ds) == 0:
                    continue
                data.setdefault((L, T), []).append(ds[:].astype(np.float32))

    return data, n_layers, n_t


def plot_mnist_rnn_temporal(thresholds):
    """Per-timestep category (digit) correlation figure for MNIST RNN."""
    npz_path = MODELS_DIR / "mnist_rnn.npz"
    if not npz_path.exists():
        print("  [skip mnist_rnn temporal] no category models — run 12 first")
        return

    threshold = (thresholds.get("mnist_rnn") or {}).get("upper")
    temporal_data, n_layers, n_t = _load_mnist_rnn_temporal(threshold)
    if not temporal_data:
        print("  [skip mnist_rnn temporal] no RDMs found")
        return

    cat_data    = np.load(npz_path)
    digit_model = cat_data["digit"]                      # (100, 100)
    n           = digit_model.shape[0]
    ri, ci      = np.triu_indices(n, k=1)
    digit_vec   = digit_model[ri, ci].astype(np.float32)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ts = np.arange(n_t)

    layer_styles = {0: ("-",  "#2166ac", "layer 0 (all)"),
                    1: ("--", "#d6604d", "layer 1 (depth=2)")}

    for L in range(n_layers):
        style, color, label = layer_styles.get(L, ("-", "grey", f"layer {L}"))
        medians, q25, q75 = [], [], []
        for T in ts:
            vecs = temporal_data.get((L, T), [])
            corrs = []
            for v in vecs:
                if len(v) != len(digit_vec):
                    continue
                r, _ = spearmanr(v, digit_vec)
                if np.isfinite(r):
                    corrs.append(r)
            if corrs:
                medians.append(np.median(corrs))
                q25.append(np.percentile(corrs, 25))
                q75.append(np.percentile(corrs, 75))
            else:
                medians.append(np.nan)
                q25.append(np.nan)
                q75.append(np.nan)

        med = np.array(medians)
        lo  = np.array(q25)
        hi  = np.array(q75)
        n_nets = max(len(temporal_data.get((L, T), [])) for T in ts)
        ax.fill_between(ts, lo, hi, alpha=0.18, color=color)
        ax.plot(ts, med, style, color=color, lw=1.5, ms=4, label=f"{label} (n≈{n_nets})")

    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.set_xlabel("timestep", fontsize=10)
    ax.set_ylabel("Spearman r (digit model)", fontsize=10)
    ax.set_title("MNIST RNN — temporal category structure\n"
                 "(median ± IQR across successful networks)",
                 fontsize=10, fontweight="bold")
    ax.set_xlim(-0.3, n_t - 0.7)
    ax.set_xticks(ts)
    ax.set_ylim(-0.15, 1.0)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    out = FIGURES_DIR / "f1_category_mnist_rnn_temporal.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds()
    meta       = task_meta()

    tasks_with_models = [t for t in TASK_NAMES if t in PRIMARY_MODEL or t in MULTI_MODEL]
    all_rows = []

    for task in tasks_with_models:
        print(f"  {task} ...", end="", flush=True)

        npz_path = MODELS_DIR / f"{task}.npz"
        if not npz_path.exists():
            print(" [no category model]")
            continue

        cat_models = dict(np.load(npz_path))   # name → N×N
        # Convert to upper-triangle vectors
        cat_vecs = {}
        for name, D in cat_models.items():
            n = D.shape[0]
            rows_idx, cols_idx = np.triu_indices(n, k=1)
            cat_vecs[name] = D[rows_idx, cols_idx].astype(np.float32)

        rdm_entries = load_all_primary_rdms(task)
        if not rdm_entries:
            print(" [no RDMs]")
            continue

        th    = thresholds[task]
        upper = th["upper"]
        lower = th["lower"]
        m     = meta[task]
        chance = m["chance_perf"]

        for run_id, perf, net_vec in rdm_entries:
            np_val = norm_perf(perf, chance, upper)
            cat    = perf_category(perf, upper, lower)

            if net_vec.std() < 1e-8:
                if cat == "successful":
                    raise ValueError(
                        f"{task}/{run_id} (perf={perf:.4f}) is successful but has "
                        "a constant RDM — check RDM computation"
                    )
                continue   # failed network with collapsed representation — skip

            for model_name, cat_vec in cat_vecs.items():
                assert len(net_vec) == len(cat_vec), (
                    f"{task}/{run_id}: net_vec length {len(net_vec)} != "
                    f"cat_vec length {len(cat_vec)} for model '{model_name}'"
                )
                r, _ = spearmanr(net_vec, cat_vec)
                assert np.isfinite(r), (
                    f"{task}/{run_id} model={model_name}: spearmanr returned {r}"
                )
                all_rows.append({
                    "task":       task,
                    "run_id":     run_id,
                    "performance": perf,
                    "norm_perf":  np_val,
                    "perf_cat":   cat,
                    "model_name": model_name,
                    "spearman_r": float(r),
                })

        n_nets = len(rdm_entries)
        print(f" {n_nets} networks, models={list(cat_vecs.keys())}")

    df = pd.DataFrame(all_rows)

    # Save CSV
    csv_path = TABLES_DIR / "rdm_category_structure.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Figure — 3×3 grid, one subplot per task (adding absent → one empty cell)
    fig, axes = plt.subplots(3, 3, figsize=(20, 17))
    axes = axes.flatten()

    for ax, task in zip(axes, tasks_with_models):
        sub = df[df["task"] == task]
        plot_task(ax, task, sub, thresholds, meta[task])

    for ax in axes[len(tasks_with_models):]:
        ax.set_visible(False)

    # Shared legend for performance categories
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=7, label=lbl)
        for lbl, c in PERF_COLORS.items()
    ]
    fig.legend(handles=handles, title="performance category",
               loc="lower right", fontsize=8, title_fontsize=8,
               framealpha=0.9)

    fig.suptitle("Category structure — Spearman r with category model RDM\n"
                 "(last hidden layer, best/final checkpoint, all primary networks)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = FIGURES_DIR / "f1_category_structure.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Temporal analyses
    print("\nAdding phase analysis ...")
    plot_adding_phases(thresholds)

    print("\nMNIST RNN temporal analysis ...")
    plot_mnist_rnn_temporal(thresholds)


if __name__ == "__main__":
    main()
