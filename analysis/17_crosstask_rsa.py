#!/usr/bin/env python3
"""
Step 17: Cross-task RSA — Finding #1.6.

For the MNIST family (mnist_dual, mnist_10way, mnist_rnn), computes
cross-task RDM correlations: one network from task A correlated with
one network from task B, over their shared stimulus set.

Shared stimulus set:
  mnist_10way ↔ mnist_rnn : identical 100 stimuli, same order.
  mnist_dual ↔ others     : mnist_dual has 200 stimuli = same 100 images
                            each presented twice (task=0 and task=1,
                            interleaved). Sub-RDM at task=0 stimuli
                            (even indices) maps exactly to the 10way/rnn
                            100 stimuli in the same order.

For each task pair, N_PAIRS random network pairs are drawn from the
successful primary networks of each task and their Spearman r computed.
Within-task pairs (same task, different networks) serve as the diagonal
reference.

Outputs:
    output/analysis/figures/f1_crosstask_rsa.pdf
    output/analysis/tables/rdm_crosstask.csv
"""

import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ANALYSIS = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import FIGURES_DIR, RDM_DIR, TABLES_DIR

MNIST_TASKS = ["mnist_dual", "mnist_10way", "mnist_rnn"]

TASK_LABELS = {
    "mnist_dual":  "MNIST dual\n(200 stim)",
    "mnist_10way": "MNIST 10-way\n(100 stim)",
    "mnist_rnn":   "MNIST RNN\n(100 stim)",
}

N_PAIRS  = 3000
RNG_SEED = 42


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


def vec_to_rdm(vec):
    """Reconstruct N×N RDM from upper-triangle vector."""
    n_pairs = len(vec)
    n = int(round((1 + np.sqrt(1 + 8 * n_pairs)) / 2))
    assert n * (n - 1) // 2 == n_pairs
    D = np.zeros((n, n), dtype=np.float32)
    rows, cols = np.triu_indices(n, k=1)
    D[rows, cols] = vec
    D += D.T
    return D


def dual_sub_vec(dual_vec):
    """
    Extract the 100-stimulus sub-RDM from a mnist_dual upper-triangle vector.

    mnist_dual has 200 stimuli interleaved: even indices are task=0,
    odd indices are task=1. The task=0 stimuli (indices 0,2,4,...,198)
    are the same 100 images used by mnist_10way and mnist_rnn.

    Returns a 4950-length upper-triangle vector (100-stim RDM).
    """
    D    = vec_to_rdm(dual_vec)           # (200, 200)
    idx  = np.arange(0, 200, 2)           # even indices = task=0
    sub  = D[np.ix_(idx, idx)]            # (100, 100)
    ri, ci = np.triu_indices(100, k=1)
    return sub[ri, ci].astype(np.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_successful_rdms(task, threshold):
    """
    Load upper-triangle RDM vectors for all successful primary networks.
    For mnist_dual: returns both the full vec (4950 = 100-stim sub) and full (19900).
    Returns: dict run_id -> vec
    """
    h5_path = RDM_DIR / f"{task}_rdms.h5"
    ckpt    = "best"   # all three MNIST tasks use 'best'
    rdms    = {}

    with h5py.File(h5_path, "r") as h5:
        runs_grp = h5.get("runs", {})
        for run_id in sorted(runs_grp.keys()):
            rg = runs_grp[run_id]
            if bool(rg.attrs.get("is_repeat", False)):
                continue
            perf = float(rg.attrs.get("performance", float("nan")))
            if threshold is not None and perf < threshold:
                continue
            depth    = int(rg.attrs.get("hp_depth", 1))
            ckpt_grp = rg.get(ckpt)
            if ckpt_grp is None:
                continue

            # Last hidden layer
            if task == "mnist_rnn":
                parsed = []
                for k in ckpt_grp.keys():
                    if "_t_" not in k:
                        continue
                    parts = k.split("_")
                    try:
                        parsed.append((int(parts[1]), int(parts[3])))
                    except (IndexError, ValueError):
                        pass
                if not parsed:
                    continue
                max_l = max(p[0] for p in parsed)
                max_t = max(p[1] for p in parsed if p[0] == max_l)
                key = f"layer_{max_l}_t_{max_t}"
            else:
                key = f"layer_{max(0, depth - 1)}"

            ds = ckpt_grp.get(key)
            if ds is None or ds.attrs.get("degenerate", False) or len(ds) == 0:
                continue

            vec = ds[:].astype(np.float32)

            if task == "mnist_dual":
                # Extract the task=0 sub-RDM (100 stimuli) for cross-task comparisons
                rdms[run_id] = {"full": vec, "sub": dual_sub_vec(vec)}
            else:
                rdms[run_id] = {"full": vec, "sub": vec}   # already 100 stimuli

    return rdms


# ---------------------------------------------------------------------------
# Cross-task correlation computation
# ---------------------------------------------------------------------------

def sample_cross_corrs(rdms_a, rdms_b, n_pairs, rng, key_a="sub", key_b="sub"):
    """
    Sample n_pairs random (network_a, network_b) pairs and compute Spearman r.
    If a is b (within-task), enforce a != b.
    """
    ids_a = list(rdms_a.keys())
    ids_b = list(rdms_b.keys())
    same_task = rdms_a is rdms_b

    corrs = []
    attempts = 0
    while len(corrs) < n_pairs and attempts < n_pairs * 5:
        ia = rng.integers(0, len(ids_a))
        ib = rng.integers(0, len(ids_b))
        if same_task and ids_a[ia] == ids_b[ib]:
            attempts += 1
            continue
        va = rdms_a[ids_a[ia]][key_a]
        vb = rdms_b[ids_b[ib]][key_b]
        if len(va) != len(vb):
            attempts += 1
            continue
        r = spearmanr(va, vb)[0]
        if np.isfinite(r):
            corrs.append(float(r))
        attempts += 1

    return np.array(corrs)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(corr_data, median_matrix, task_list):
    """
    Two-panel figure:
      Left:  heatmap of median cross-task Spearman r
      Right: violin distributions for each task pair (within on diagonal)
    """
    n = len(task_list)
    labels = [TASK_LABELS[t] for t in task_list]

    # Collect all pair labels and data
    pair_labels = []
    pair_data   = []
    pair_colors = []
    DIAG_COLOR  = "#2166ac"
    OFF_COLOR   = "#d6604d"

    for i, ta in enumerate(task_list):
        for j, tb in enumerate(task_list):
            if j < i:
                continue
            key = (ta, tb)
            if key not in corr_data:
                continue
            if i == j:
                pair_labels.append(f"{ta.replace('mnist_','')}\n(within)")
                pair_colors.append(DIAG_COLOR)
            else:
                pair_labels.append(f"{ta.replace('mnist_','')} ↔\n{tb.replace('mnist_','')}")
                pair_colors.append(OFF_COLOR)
            pair_data.append(corr_data[key])

    fig = plt.figure(figsize=(13, 5))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.2, 2.5], wspace=0.35)
    ax_heat = fig.add_subplot(gs[0])
    ax_viol = fig.add_subplot(gs[1])

    # --- Heatmap ---
    mat = median_matrix.copy()
    # Mask upper triangle strictly (keep diagonal and lower)
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
    mat_masked = np.where(mask, np.nan, mat)

    vmin = max(0, np.nanmin(mat_masked) - 0.05)
    vmax = min(1, np.nanmax(mat_masked) + 0.05)
    cmap = plt.cm.RdYlGn

    im = ax_heat.imshow(mat_masked, cmap=cmap, vmin=vmin, vmax=vmax,
                        aspect="equal", origin="upper")
    ax_heat.set_xticks(range(n))
    ax_heat.set_yticks(range(n))
    ax_heat.set_xticklabels(labels, fontsize=8)
    ax_heat.set_yticklabels(labels, fontsize=8)
    plt.colorbar(im, ax=ax_heat, shrink=0.8, label="median Spearman r")
    for i in range(n):
        for j in range(n):
            if j <= i and not np.isnan(mat_masked[i, j]):
                ax_heat.text(j, i, f"{mat_masked[i,j]:.3f}",
                             ha="center", va="center", fontsize=8, fontweight="bold")
    ax_heat.set_title("Median cross-task Spearman r\n(lower triangle)", fontsize=9)

    # --- Violin distributions ---
    pos = np.arange(len(pair_data))
    for k, (vals, color) in enumerate(zip(pair_data, pair_colors)):
        if len(vals) == 0:
            continue
        parts = ax_viol.violinplot([vals], positions=[k], showmedians=True, showextrema=True)
        parts["bodies"][0].set_facecolor(color)
        parts["bodies"][0].set_alpha(0.65)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

    ax_viol.set_xticks(pos)
    ax_viol.set_xticklabels(pair_labels, fontsize=8)
    ax_viol.set_ylabel("Spearman r", fontsize=9)
    ax_viol.set_ylim(-0.1, 1.05)
    ax_viol.axhline(0, color="grey", lw=0.6, ls="--")
    ax_viol.set_title("Cross-task RDM correlation distributions\n"
                      "(blue = within-task, orange = cross-task; "
                      "successful primary networks)", fontsize=9)

    # Legend patches
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=DIAG_COLOR, alpha=0.65, label="within-task"),
               Patch(facecolor=OFF_COLOR,  alpha=0.65, label="cross-task")]
    ax_viol.legend(handles=handles, fontsize=8, loc="lower right")

    fig.suptitle(
        "Cross-task RSA — MNIST family\n"
        "(mnist_dual sub-RDM at task=0 stimuli used for cross-task comparison)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds()
    rng        = np.random.default_rng(RNG_SEED)

    # Load RDMs
    rdms_by_task = {}
    for task in MNIST_TASKS:
        th = thresholds.get(task)
        print(f"  Loading {task} (threshold={th}) ...", end="", flush=True)
        rdms = load_successful_rdms(task, th)
        rdms_by_task[task] = rdms
        print(f" {len(rdms)} successful networks")

    n = len(MNIST_TASKS)
    median_matrix = np.full((n, n), np.nan)
    corr_data     = {}
    all_rows      = []

    for i, ta in enumerate(MNIST_TASKS):
        for j, tb in enumerate(MNIST_TASKS):
            if j < i:
                continue
            ra = rdms_by_task[ta]
            rb = rdms_by_task[tb]
            same = (ta == tb)

            print(f"  {ta} ↔ {tb} ...", end="", flush=True)

            # Always use "sub": for dual that's the 100-stim task=0 sub-RDM;
            # for 10way/rnn it's identical to the full RDM.
            # This ensures all comparisons are over the same 4950 stimulus pairs.
            corrs = sample_cross_corrs(ra, rb, N_PAIRS, rng, key_a="sub", key_b="sub")
            print(f" n={len(corrs)}, med={np.median(corrs):.3f}")

            median_matrix[i, j] = np.median(corrs)
            corr_data[(ta, tb)] = corrs

            pair_type = "within" if same else "cross"
            for r in corrs:
                all_rows.append({
                    "task_a":    ta,
                    "task_b":    tb,
                    "pair_type": pair_type,
                    "spearman_r": float(r),
                })

    # Save CSV
    csv_path = TABLES_DIR / "rdm_crosstask.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Figure
    fig = make_figure(corr_data, median_matrix, MNIST_TASKS)
    out = FIGURES_DIR / "f1_crosstask_rsa.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
