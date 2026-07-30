#!/usr/bin/env python3
"""
Performance vs. representational geometry (Finding #1.7).

For every pair of networks within the same performance bin, computes:
  - RDM dissimilarity (1 - Spearman r between best/final RDM vectors)
  - HP distance (Euclidean distance in normalised mixed HP space)

Two outputs per task:
  1. Within-bin mean RDM similarity across 10 normalised-performance bins
     (plus the top 1%). Tests whether high-performing networks converge
     representationally.
  2. Within-bin Spearman r between HP distance and RDM dissimilarity.
     Tests whether HP differences still predict representational differences
     at high performance, or whether convergence is HP-independent.

Outputs:
  tables/perf_rdm_bins.csv
  figures/perf_rdm_within_bin.pdf
  figures/perf_rdm_hp_correlation.pdf
"""

import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ANALYSIS  = Path(__file__).parent
REPO_ROOT = ANALYSIS.parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import (RDM_DIR, TABLES_DIR, FIGURES_DIR,
                             RL_TASKS, TASK_NAMES, task_meta)

RNN_TASKS     = {"adding", "mnist_rnn"}
SUBSAMPLE_MAX = 10_000
N_BINS        = 10
TOP_FRAC      = 0.01   # top 1%
MIN_PAIRS     = 10     # min pairs required to report a bin statistic
METRIC        = "cosine"
LOG_HPS       = {"hp_learning_rate", "hp_l1_reg", "hp_l2_reg"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def last_layer_key(task, rg):
    if task in RNN_TASKS:
        return f"temporal_{METRIC}"
    depth = int(rg.attrs.get("hp_depth", 1))
    return f"layer_{depth - 1}_{METRIC}"


def ref_ckpt_name(task):
    return "final" if task in RL_TASKS else "best"


def normalise_perf(perf, task):
    tm = task_meta()[task]
    return (perf - tm["chance_perf"]) / (tm["max_metric"] - tm["chance_perf"])


# ---------------------------------------------------------------------------
# HP feature matrix
# ---------------------------------------------------------------------------

def build_hp_matrix(hp_dicts):
    """
    Convert a list of HP dicts to a normalised N×P feature matrix.
    String-valued HPs → one-hot (drop first level).
    Numeric HPs → log-transform if in LOG_HPS, then z-score.
    Returns (matrix, feature_names).
    """
    all_keys = sorted(set().union(*[d.keys() for d in hp_dicts]))
    features, names = [], []

    for key in all_keys:
        vals = [d.get(key) for d in hp_dicts]
        if any(isinstance(v, str) for v in vals if v is not None):
            levels = sorted(set(v for v in vals if v is not None))
            for level in levels[1:]:   # drop reference level
                col = np.array([float(v == level) if v is not None else 0.0
                                for v in vals])
                features.append(col)
                names.append(f"{key}={level}")
        else:
            col = np.array([float(v) if v is not None else np.nan for v in vals])
            if key in LOG_HPS:
                col = np.log(np.maximum(col, 1e-12))
            std = np.nanstd(col)
            if std > 0:
                col = (col - np.nanmean(col)) / std
            np.nan_to_num(col, nan=0.0, copy=False)
            features.append(col)
            names.append(key)

    if not features:
        return np.zeros((len(hp_dicts), 0)), []
    return np.column_stack(features), names


# ---------------------------------------------------------------------------
# Pairwise distance matrices
# ---------------------------------------------------------------------------

def pairwise_spearman_dissim(X):
    """
    N×N pairwise Spearman dissimilarity via rank-transform + Pearson.
    Strips non-finite columns first.
    """
    X = X[:, np.all(np.isfinite(X), axis=0)]
    if X.shape[1] == 0:
        return None
    X_r = np.apply_along_axis(rankdata, 1, X).astype(np.float64)
    X_r -= X_r.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X_r, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_r /= norms
    R = np.clip(X_r @ X_r.T, -1.0, 1.0)
    D = (1.0 - R) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


def pairwise_euclidean(M):
    """N×N pairwise Euclidean distance."""
    sq = (M ** 2).sum(axis=1)
    D  = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * (M @ M.T), 0.0))
    np.fill_diagonal(D, 0.0)
    return D


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_task_data(task):
    """
    Returns dict with run_ids, norm_perfs, rdm_matrix (N×D), hp_matrix (N×P).
    Returns None if the task has no usable networks.
    """
    h5_path  = RDM_DIR / f"{task}_rdms.h5"
    ref_name = ref_ckpt_name(task)

    # Pass 1: collect valid run_ids, perfs, HPs — metadata only, no vector reads
    run_ids, perfs, hp_dicts = [], [], []
    with h5py.File(h5_path, "r") as f:
        for run_id, rg in f["runs"].items():
            if rg.attrs.get("is_repeat", False):
                continue
            perf = float(rg.attrs.get("performance", float("nan")))
            if not np.isfinite(perf):
                continue
            lkey    = last_layer_key(task, rg)
            ref_cg  = rg.get(ref_name)
            if ref_cg is None:
                continue
            ds = ref_cg.get(lkey)
            if ds is None or ds.attrs.get("degenerate", False) or len(ds) == 0:
                continue
            hp_dict = {k: v for k, v in rg.attrs.items() if k.startswith("hp_")}
            run_ids.append(run_id)
            perfs.append(perf)
            hp_dicts.append(hp_dict)

    if len(run_ids) < 10:
        return None

    norm_perfs = np.clip(
        np.array([normalise_perf(p, task) for p in perfs]), 0.0, 1.0
    )

    # Pass 2: probe vector size from one network, compute subsample index
    subsample_idx = None
    with h5py.File(h5_path, "r") as f:
        rg      = f["runs"][run_ids[0]]
        lkey    = last_layer_key(task, rg)
        vec_size = len(rg[ref_name][lkey])
    if vec_size > SUBSAMPLE_MAX:
        rng = np.random.default_rng(42)
        subsample_idx = rng.choice(vec_size, size=SUBSAMPLE_MAX, replace=False)
        print(f"  vector size {vec_size:,} → subsampling to {SUBSAMPLE_MAX:,} dims")

    # Pass 3: load RDM vectors, subsample at load time
    rdm_vecs = []
    with h5py.File(h5_path, "r") as f:
        for run_id in run_ids:
            rg   = f["runs"][run_id]
            lkey = last_layer_key(task, rg)
            vec  = rg[ref_name][lkey][:].astype(np.float64)
            if subsample_idx is not None:
                vec = vec[subsample_idx]
            rdm_vecs.append(vec)

    hp_matrix, hp_names = build_hp_matrix(hp_dicts)

    return {
        "run_ids":    run_ids,
        "norm_perfs": norm_perfs,
        "rdm_matrix": np.array(rdm_vecs, dtype=np.float64),
        "hp_matrix":  hp_matrix,
        "hp_names":   hp_names,
    }


# ---------------------------------------------------------------------------
# Per-task analysis
# ---------------------------------------------------------------------------

def analyze_task(task, data):
    """
    Returns list of row dicts (one per bin + one for top 1%).
    """
    norm_perfs = data["norm_perfs"]
    rdm_D = pairwise_spearman_dissim(data["rdm_matrix"])
    if rdm_D is None:
        return []
    hp_D = pairwise_euclidean(data["hp_matrix"])

    N = len(norm_perfs)
    ii, jj   = np.triu_indices(N, k=1)
    rdm_all  = rdm_D[ii, jj]
    hp_all   = hp_D[ii, jj]
    bin_edges = np.linspace(0.0, 1.0, N_BINS + 1)
    bin_of   = np.clip(np.digitize(norm_perfs, bin_edges) - 1, 0, N_BINS - 1)

    rows = []

    for b in range(N_BINS):
        in_bin    = np.where(bin_of == b)[0]
        pair_mask = (bin_of[ii] == b) & (bin_of[jj] == b)
        n_pairs   = int(pair_mask.sum())
        if n_pairs < MIN_PAIRS:
            continue

        rdm_bin = rdm_all[pair_mask]
        hp_bin  = hp_all[pair_mask]

        hp_r = float("nan")
        if hp_bin.std() > 0:
            hp_r, _ = spearmanr(hp_bin, rdm_bin)

        rows.append({
            "task":       task,
            "bin":        b,
            "bin_center": (bin_edges[b] + bin_edges[b + 1]) / 2.0,
            "bin_lo":     bin_edges[b],
            "bin_hi":     bin_edges[b + 1],
            "n_nets":     int(len(in_bin)),
            "n_pairs":    n_pairs,
            "mean_sim":   float(1.0 - rdm_bin.mean()),
            "std_sim":    float(rdm_bin.std()),
            "hp_rdm_r":   float(hp_r),
            "is_top1pct": False,
        })

    # Top 1%
    n_top  = max(2, int(np.ceil(N * TOP_FRAC)))
    top_idx = np.argsort(norm_perfs)[-n_top:]
    top_set = set(top_idx.tolist())
    top_mask = np.array([i in top_set and j in top_set for i, j in zip(ii, jj)])
    if top_mask.sum() >= MIN_PAIRS:
        rdm_top = rdm_all[top_mask]
        hp_top  = hp_all[top_mask]
        hp_r_top = float("nan")
        if hp_top.std() > 0:
            hp_r_top, _ = spearmanr(hp_top, rdm_top)
        rows.append({
            "task":       task,
            "bin":        N_BINS,       # plot past the last regular bin
            "bin_center": 1.0,
            "bin_lo":     1.0 - TOP_FRAC,
            "bin_hi":     1.0,
            "n_nets":     n_top,
            "n_pairs":    int(top_mask.sum()),
            "mean_sim":   float(1.0 - rdm_top.mean()),
            "std_sim":    float(rdm_top.std()),
            "hp_rdm_r":   float(hp_r_top),
            "is_top1pct": True,
        })

    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

TASK_LABELS = {
    "mnist_dual":    "MNIST dual",
    "mnist_10way":   "MNIST 10-way",
    "fashion_10way": "Fashion 10-way",
    "spirals":       "Spirals",
    "parity":        "Parity",
    "adding":        "Adding",
    "mnist_rnn":     "MNIST RNN",
    "cartpole":      "CartPole",
    "fourrooms":     "FourRooms",
}


def make_figures(df):
    tasks = [t for t in TASK_NAMES if t in df["task"].values]
    ncols = 3
    nrows = int(np.ceil(len(tasks) / ncols))

    # --- Figure A: within-bin RDM similarity ---
    fig_a, axes_a = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 4.0, nrows * 3.2),
                                 squeeze=False)
    for ax_idx, task in enumerate(tasks):
        ax  = axes_a[ax_idx // ncols][ax_idx % ncols]
        tdf = df[df["task"] == task]
        reg = tdf[~tdf["is_top1pct"]].sort_values("bin_center")
        top = tdf[tdf["is_top1pct"]]

        ax.plot(reg["bin_center"], reg["mean_sim"],
                marker="o", markersize=5, linewidth=1.4, color="#2166ac")
        ax.fill_between(reg["bin_center"],
                        reg["mean_sim"] - reg["std_sim"],
                        reg["mean_sim"] + reg["std_sim"],
                        alpha=0.15, color="#2166ac")
        if not top.empty:
            ax.scatter([1.0], top["mean_sim"].values,
                       marker="*", s=120, color="#d6604d", zorder=5,
                       label=f"top 1% (n={int(top['n_nets'].iloc[0])})")
            ax.legend(fontsize=6, loc="lower right")

        ax.set_xlim(-0.05, 1.08)
        ax.set_ylim(0, 1.05)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Normalised performance bin", fontsize=8)
        ax.set_ylabel("Mean within-bin RDM similarity", fontsize=8)
        ax.set_title(TASK_LABELS.get(task, task), fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

        # Annotate n per bin
        for _, row in reg.iterrows():
            ax.text(row["bin_center"], -0.06, f"n={int(row['n_nets'])}",
                    ha="center", va="top", fontsize=4.5, color="#555555",
                    transform=ax.get_xaxis_transform())

    for ax_idx in range(len(tasks), nrows * ncols):
        axes_a[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    fig_a.suptitle(
        "Within-performance-bin RDM similarity\n"
        "Do networks at similar performance levels converge representationally?",
        fontsize=9,
    )
    fig_a.tight_layout()

    # --- Figure B: HP→RDM Spearman r within bin ---
    fig_b, axes_b = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 4.0, nrows * 3.2),
                                 squeeze=False)
    for ax_idx, task in enumerate(tasks):
        ax  = axes_b[ax_idx // ncols][ax_idx % ncols]
        tdf = df[df["task"] == task]
        reg = tdf[~tdf["is_top1pct"]].sort_values("bin_center")
        top = tdf[tdf["is_top1pct"]]

        valid = reg["hp_rdm_r"].notna()
        ax.plot(reg["bin_center"][valid], reg["hp_rdm_r"][valid],
                marker="o", markersize=5, linewidth=1.4, color="#4dac26")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        if not top.empty and np.isfinite(top["hp_rdm_r"].iloc[0]):
            ax.scatter([1.0], top["hp_rdm_r"].values,
                       marker="*", s=120, color="#d6604d", zorder=5,
                       label=f"top 1% (n={int(top['n_nets'].iloc[0])})")
            ax.legend(fontsize=6, loc="upper left")

        ax.set_xlim(-0.05, 1.08)
        ax.set_ylim(-0.2, 0.8)
        ax.set_xlabel("Normalised performance bin", fontsize=8)
        ax.set_ylabel("Spearman r (HP dist vs. RDM dissim)", fontsize=8)
        ax.set_title(TASK_LABELS.get(task, task), fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

    for ax_idx in range(len(tasks), nrows * ncols):
        axes_b[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    fig_b.suptitle(
        "Does HP distance predict RDM dissimilarity within each performance bin?\n"
        "Declining slope → representations converge independent of hyperparameters",
        fontsize=9,
    )
    fig_b.tight_layout()

    return fig_a, fig_b


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for task in TASK_NAMES:
        h5_path = RDM_DIR / f"{task}_rdms.h5"
        if not h5_path.exists():
            print(f"{task}: no HDF5, skipping")
            continue
        print(f"{task} ...", flush=True)

        data = load_task_data(task)
        if data is None:
            print(f"  insufficient data, skipping")
            continue
        print(f"  {len(data['run_ids'])} networks loaded")

        rows = analyze_task(task, data)
        all_rows.extend(rows)
        n_bins_filled = sum(1 for r in rows if not r["is_top1pct"])
        print(f"  {n_bins_filled} non-empty bins")

    if not all_rows:
        print("No data produced.")
        return

    df = pd.DataFrame(all_rows)
    out_csv = TABLES_DIR / "perf_rdm_bins.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv.name}")

    fig_a, fig_b = make_figures(df)

    out_a = FIGURES_DIR / "perf_rdm_within_bin.pdf"
    fig_a.savefig(out_a, bbox_inches="tight", dpi=130)
    plt.close(fig_a)
    print(f"Saved: {out_a.name}")

    out_b = FIGURES_DIR / "perf_rdm_hp_correlation.pdf"
    fig_b.savefig(out_b, bbox_inches="tight", dpi=130)
    plt.close(fig_b)
    print(f"Saved: {out_b.name}")


if __name__ == "__main__":
    main()
