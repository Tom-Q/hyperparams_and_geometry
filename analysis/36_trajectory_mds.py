#!/usr/bin/env python3
"""
Representational trajectory MDS (Finding #3).

For each task: selects up to 100 successful networks at random, embeds all their
step-checkpoint RDM vectors jointly in 2D via MDS (Spearman dissimilarity), and
draws each network's trajectory as a line colored from light blue (start of
training) to dark red (end of training).

Output:
  output/analysis/figures/trajectory_mds.pdf
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import gc
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.collections import LineCollection
from scipy.stats import spearmanr
from sklearn.manifold import MDS

ANALYSIS  = Path(__file__).parent
REPO_ROOT = ANALYSIS.parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import RDM_DIR, TABLES_DIR, FIGURES_DIR, FINAL_DIR, RL_TASKS, TASK_NAMES, get_depth, is_run_successful, metric_suffix

RNN_TASKS = {"adding", "mnist_rnn"}
N_NETS        = 500          # upper bound on networks; actual count also limited by MAX_POINTS
MAX_POINTS    = 4500        # hard cap on total vectors entering MDS (networks × checkpoints)
SUBSAMPLE_MAX = 10_000      # max RDM vector dims for large-vector tasks
METRIC        = "pearson"  # default; overridden by --metric
CMAP      = mcolors.LinearSegmentedColormap.from_list(
    "learning", ["#90ee90", "#4393c3", "#d6604d", "#8b0000", "#000000"]
)


# ---------------------------------------------------------------------------
# Helpers (reused from 30_explore_trajectories.py)
# ---------------------------------------------------------------------------

def last_layer_key(task, rg):
    if task in RNN_TASKS:
        return f"temporal_{METRIC}"
    depth = get_depth(rg)
    return f"layer_{depth - 1}_{METRIC}"


def load_rdm_vec(cg, key):
    ds = cg.get(key)
    if ds is None or ds.attrs.get("degenerate", False) or len(ds) == 0:
        return None
    return ds[:].astype(np.float64)


def strip_nan_cols(X):
    valid = np.all(np.isfinite(X), axis=0)
    return X[:, valid]


def spearman_dissimilarity_matrix(X):
    n = X.shape[0]
    result = spearmanr(X.T)
    if n == 2:
        r_mat = np.array([[1.0, result.statistic],
                          [result.statistic, 1.0]])
    else:
        r_mat = np.array(result.statistic)
    D = (1.0 - r_mat) / 2.0
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0.0, 1.0)


def load_thresholds():
    data = json.load(open(TABLES_DIR / "success_thresholds.json"))
    return {k: float(v["upper"]) for k, v in data.items() if isinstance(v, dict)}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _rdm_exists(cg, key):
    """Lightweight check: does a non-degenerate, non-empty RDM exist? No data loaded."""
    ds = cg.get(key)
    return ds is not None and not ds.attrs.get("degenerate", False) and len(ds) > 0


def select_networks(task, thresholds, rng, max_nets=N_NETS):
    h5_path   = RDM_DIR / f"{task}_rdms.h5"
    if not h5_path.exists():
        return []
    nets = []
    with h5py.File(h5_path, "r") as f:
        for run_id, rg in f["runs"].items():
            if rg.attrs.get("is_repeat", False):
                continue
            if not is_run_successful(task, rg, thresholds):
                continue
            lkey = last_layer_key(task, rg)
            valid = sum(1 for n in rg.keys()
                        if n.startswith("step_") and _rdm_exists(rg[n], lkey))
            if valid >= 2:
                nets.append(run_id)
    if not nets:
        return []
    chosen = rng.choice(nets, size=min(max_nets, len(nets)), replace=False)
    return list(chosen)


def load_step_trajectories(task, run_ids, subsample_idx=None):
    """
    Returns list of dicts: {run_id, steps (sorted list), vecs (list of arrays)}.
    subsample_idx, if given, is applied immediately per-vector to cap memory.
    """
    h5_path = RDM_DIR / f"{task}_rdms.h5"
    result  = []
    with h5py.File(h5_path, "r") as f:
        for run_id in run_ids:
            rg   = f["runs"][run_id]
            lkey = last_layer_key(task, rg)
            ckpts = {}
            for name, cg in rg.items():
                if not name.startswith("step_"):
                    continue
                vec = load_rdm_vec(cg, lkey)
                if vec is not None:
                    if subsample_idx is not None:
                        vec = vec[subsample_idx]
                    step = int(name[5:])
                    ckpts[step] = vec
            if len(ckpts) < 2:
                continue
            sorted_steps = sorted(ckpts)
            result.append({
                "run_id": run_id,
                "steps":  sorted_steps,
                "vecs":   [ckpts[s] for s in sorted_steps],
            })
    return result


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def probe_vec_size(task, run_ids):
    """Return the length of one step-checkpoint RDM vector, or None."""
    h5_path = RDM_DIR / f"{task}_rdms.h5"
    with h5py.File(h5_path, "r") as f:
        for run_id in run_ids:
            rg = f["runs"][run_id]
            lkey = last_layer_key(task, rg)
            for name, cg in rg.items():
                if name.startswith("step_"):
                    ds = cg.get(lkey)
                    if ds is not None and len(ds) > 0:
                        return len(ds)
    return None


MAX_NETS_PREFILTER = N_NETS   # post-load trim enforces MAX_POINTS; this just caps HDF5 reads


def make_task_figure(task, thresholds, rng):
    run_ids = select_networks(task, thresholds, rng, max_nets=MAX_NETS_PREFILTER)
    if not run_ids:
        print(f"  no usable networks, skipping")
        return None

    # Subsample large RDM vectors — index computed once, applied per-vector during load
    vec_size = probe_vec_size(task, run_ids)
    subsample_idx = None
    if vec_size and vec_size > SUBSAMPLE_MAX:
        subsample_idx = rng.choice(vec_size, size=SUBSAMPLE_MAX, replace=False)
        print(f"  vector size {vec_size:,} → subsampling to {SUBSAMPLE_MAX:,} dims")

    nets = load_step_trajectories(task, run_ids, subsample_idx=subsample_idx)
    if not nets:
        return None

    # Post-load trim: enforce MAX_POINTS on actual total vector count
    total_pts = sum(len(nd["steps"]) for nd in nets)
    if total_pts > MAX_POINTS:
        rng.shuffle(nets)
        kept, count = [], 0
        for nd in nets:
            if count + len(nd["steps"]) <= MAX_POINTS:
                kept.append(nd)
                count += len(nd["steps"])
        nets = kept
        print(f"  trimmed to {len(nets)} networks ({count} total vectors)")
    if not nets:
        return None

    all_vecs, net_slices = [], []
    for nd in nets:
        start = len(all_vecs)
        all_vecs.extend(nd["vecs"])
        net_slices.append((start, len(all_vecs), nd["steps"]))

    X = np.array(all_vecs, dtype=np.float64)
    X = strip_nan_cols(X)
    if X.shape[1] == 0:
        return None

    D   = spearman_dissimilarity_matrix(X)
    del X
    gc.collect()
    np.nan_to_num(D, nan=1.0, copy=False)
    emb = MDS(n_components=2, metric="precomputed",
              n_init=1, init="classical_mds",
              random_state=42, normalized_stress="auto").fit_transform(D)
    del D
    gc.collect()

    fig, ax = plt.subplots(figsize=(6, 5.5))

    for (start, end, steps) in net_slices:
        coords    = emb[start:end]
        log_steps = np.log(np.maximum(steps, 1))
        t_vals    = log_steps / log_steps[-1]

        segments, colors = [], []
        for i in range(len(coords) - 1):
            segments.append([coords[i], coords[i + 1]])
            colors.append((t_vals[i] + t_vals[i + 1]) / 2.0)

        lc = LineCollection(segments, cmap=CMAP, norm=plt.Normalize(0, 1),
                            linewidth=0.5, alpha=0.25, zorder=2)
        lc.set_array(np.array(colors))
        ax.add_collection(lc)

    ax.autoscale_view()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("MDS dim 1", fontsize=8)
    ax.set_ylabel("MDS dim 2", fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Training progress (log scale)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        f"{task} — representational trajectories (MDS)\n"
        f"{len(nets)} successful networks  |  step checkpoints",
        fontsize=9,
    )
    fig.tight_layout()

    del emb, nets, all_vecs, net_slices
    gc.collect()
    return fig


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["cosine", "pearson"], default="pearson")
    parser.add_argument("--task", nargs="+", metavar="TASK",
                        help="Only generate trajectory MDS for these tasks.")
    args = parser.parse_args()

    global METRIC
    METRIC = args.metric

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds()
    rng = np.random.default_rng(42)

    tasks_to_run = set(args.task) if args.task else set(TASK_NAMES)

    for task in TASK_NAMES:
        if task not in tasks_to_run:
            continue
        if not (RDM_DIR / f"{task}_rdms.h5").exists():
            print(f"{task}: no HDF5, skipping")
            continue
        print(f"{task} ...", flush=True)
        fig = make_task_figure(task, thresholds, rng)
        if fig is None:
            continue
        out = FIGURES_DIR / f"trajectory_mds_{task}.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=130)

        suf = metric_suffix(METRIC)
        final_dir = FINAL_DIR / "learning_dynamics/figures/trajectory_mds"
        final_dir.mkdir(parents=True, exist_ok=True)
        out_png = final_dir / f"trajectory_mds_{task}{suf}.png"
        fig.savefig(out_png, dpi=200, bbox_inches="tight")

        plt.close(fig)
        gc.collect()
        print(f"  Saved: {out.name}, {out_png.name}")


if __name__ == "__main__":
    main()
