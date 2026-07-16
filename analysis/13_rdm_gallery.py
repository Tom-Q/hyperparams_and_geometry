#!/usr/bin/env python3
"""
Step 13: RDM gallery — mean RDM + individual examples for each task.

For each task, loads successful primary network RDMs (last hidden layer,
best/final checkpoint), computes the group mean, and shows it alongside
6 randomly sampled individual RDMs. Stimuli are sorted by primary category.

Outputs:
    output/analysis/figures/rdm_gallery_{task}.pdf
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANALYSIS = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import (
    CACHE_DIR, FIGURES_DIR, RDM_DIR, TABLES_DIR, TASK_NAMES, RL_TASKS,
    metric_output_dirs,
)
from plot_utils import plot_rdm, vec_to_rdm

MODELS_DIR  = CACHE_DIR / "category_models"
TASK_DIR_OVERRIDES = {}
RNN_TASKS   = {"adding", "mnist_rnn"}
N_EXAMPLES  = 6
RNG_SEED    = 0

TASK_LABELS = {
    "mnist_dual":    "MNIST dual (200 stimuli)",
    "mnist_10way":   "MNIST 10-way (100 stimuli)",
    "fashion_10way": "Fashion 10-way (100 stimuli)",
    "mnist_rnn":     "MNIST RNN (100 stimuli)",
    "spirals":       "Spirals (198 stimuli)",
    "parity":        "Parity (118 stimuli)",
    "adding":        "Adding (100 stimuli)",
    "cartpole":      "CartPole (196 stimuli)",
    "fourrooms":     "FourRooms (68 stimuli)",
}

# Primary sort key per task: (category_model_name_or_None, metadata_key)
# None model_name → load directly from task metadata
SORT_CONFIG = {
    "mnist_dual":    ("digit",        "digit"),
    "mnist_10way":   ("digit",        "digit"),
    "fashion_10way": ("class",        "class"),
    "mnist_rnn":     ("digit",        "digit"),
    "spirals":       ("arm",          "arm"),
    "parity":        ("hamming_diff", "hamming"),
    "adding":        (None,           "target"),
    "cartpole":      ("angle_diff",   "angle"),
    "fourrooms":     ("room",         "room"),
}


# ---------------------------------------------------------------------------
# Sort order helpers
# ---------------------------------------------------------------------------

def get_sort_info(task):
    """
    Returns (sort_idx, line_vals) where sort_idx is the permutation that
    orders stimuli by primary category, and line_vals is the label array
    in sorted order (integer, for drawing category boundary lines).
    """
    from tasks import TASKS

    model_name, _ = SORT_CONFIG.get(task, (None, None))
    task_obj = TASKS[task]()
    _, metadata = task_obj.get_rdm_stimuli()

    if task == "mnist_dual":
        digits = metadata["digits"]
        sort_idx = np.argsort(digits, kind="stable")
        line_vals = digits[sort_idx]

    elif task in ("mnist_10way", "mnist_rnn"):
        digits = metadata["digits"]
        sort_idx = np.argsort(digits, kind="stable")
        line_vals = digits[sort_idx]

    elif task == "fashion_10way":
        classes = metadata["classes"]
        sort_idx = np.argsort(classes, kind="stable")
        line_vals = classes[sort_idx]

    elif task == "spirals":
        classes = metadata["classes"]
        sort_idx = np.argsort(classes, kind="stable")
        line_vals = classes[sort_idx]

    elif task == "parity":
        n_ones = metadata["n_ones"]
        sort_idx = np.argsort(n_ones, kind="stable")
        line_vals = n_ones[sort_idx]

    elif task == "adding":
        targets = metadata["targets"]
        sort_idx = np.argsort(targets, kind="stable")
        line_vals = None   # continuous — no boundary lines

    elif task == "cartpole":
        angles = metadata["pole_angles"]
        sort_idx = np.argsort(angles, kind="stable")
        line_vals = None   # continuous

    elif task == "fourrooms":
        rows = metadata["rows"].astype(float)
        cols = metadata["cols"].astype(float)
        N = len(rows)
        room = np.full(N, 4, dtype=int)
        for i in range(N):
            r, c = int(rows[i]), int(cols[i])
            if r <= 4 and c <= 4:
                room[i] = 0
            elif r <= 4 and c >= 6:
                room[i] = 1
            elif r >= 6 and c <= 4:
                room[i] = 2
            elif r >= 6 and c >= 6:
                room[i] = 3
        sort_idx = np.argsort(room, kind="stable")
        line_vals = room[sort_idx]

    else:
        N = list(metadata.values())[0].shape[0]
        sort_idx = np.arange(N)
        line_vals = None

    return sort_idx, line_vals


# ---------------------------------------------------------------------------
# RDM loading (same logic as 11_rsa_validity.py)
# ---------------------------------------------------------------------------

def get_ckpt_name(task):
    return "final" if task in RL_TASKS else "best"


def get_last_layer_key(ckpt_grp, task, depth, metric="cosine"):
    if task in RNN_TASKS:
        parsed = []
        for k in ckpt_grp.keys():
            if "_t_" not in k:
                continue
            parts = k.split("_")
            try:
                l_idx = int(parts[1])
                t_idx = int(parts[3])
                parsed.append((l_idx, t_idx))
            except (IndexError, ValueError):
                continue
        if not parsed:
            return None
        max_l = max(p[0] for p in parsed)
        max_t = max(p[1] for p in parsed if p[0] == max_l)
        return f"layer_{max_l}_t_{max_t}_{metric}"
    else:
        return f"layer_{max(0, int(depth) - 1)}_{metric}"


def load_task_rdms(task, success_threshold=None, metric="cosine"):
    """Load RDM vectors for successful primary networks. Returns list of (run_id, perf, vec)."""
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
            if success_threshold is not None and perf < success_threshold:
                continue
            depth = int(rg.attrs.get("hp_depth", 1))
            ckpt_grp = rg.get(ckpt)
            if ckpt_grp is None:
                continue
            key = get_last_layer_key(ckpt_grp, task, depth, metric=metric)
            if key is None:
                continue
            ds = ckpt_grp.get(key)
            if ds is None or ds.attrs.get("degenerate", False) or len(ds) == 0:
                continue
            results.append((run_id, perf, ds[:].astype(np.float32)))

    return results


def load_thresholds():
    path = TABLES_DIR / "success_thresholds.json"
    if not path.exists():
        return {}
    data = json.load(open(path))
    return {k: float(v["upper"]) for k, v in data.items()
            if k != "_alpha" and isinstance(v, dict) and v.get("upper") is not None}


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def make_gallery(task, rdm_entries, sort_idx, line_vals, display_label, metric="cosine"):
    """
    rdm_entries : list of (run_id, perf, vec) — all successful primary networks
    """
    rng = np.random.default_rng(RNG_SEED)
    n_total = len(rdm_entries)

    # Mean RDM
    all_vecs = np.stack([e[2] for e in rdm_entries])
    mean_vec = all_vecs.mean(axis=0)

    # Sample N_EXAMPLES, evenly spread by performance rank
    if n_total <= N_EXAMPLES:
        sample = list(range(n_total))
    else:
        perfs = np.array([e[1] for e in rdm_entries])
        order = np.argsort(perfs)
        # Pick N_EXAMPLES evenly spaced indices from sorted list
        picks = np.round(np.linspace(0, n_total - 1, N_EXAMPLES)).astype(int)
        sample = [order[p] for p in picks]

    ncols = 1 + len(sample)
    fig, axes = plt.subplots(1, ncols, figsize=(2.8 * ncols, 3.2),
                             constrained_layout=True)

    # Mean RDM (slightly larger title)
    im = plot_rdm(axes[0], mean_vec, title=f"mean\n(N={n_total})",
                  sort_idx=sort_idx, line_vals=line_vals)

    # Individual examples
    for ax, idx in zip(axes[1:], sample):
        run_id, perf, vec = rdm_entries[idx]
        plot_rdm(ax, vec, title=f"{run_id}\nperf={perf:.3f}",
                 sort_idx=sort_idx, line_vals=line_vals)

    dist_label = "cosine dist" if metric == "cosine" else "Pearson dist"
    fig.colorbar(im, ax=axes[-1], shrink=0.8, label=dist_label)
    fig.suptitle(display_label, fontsize=10, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RDM gallery.")
    parser.add_argument("--metric", choices=["cosine", "pearson"], default="cosine",
                        help="RDM metric to use (default: cosine).")
    args = parser.parse_args()

    out_figures, _ = metric_output_dirs(args.metric)
    out_figures.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds()

    for task in TASK_NAMES:
        threshold = thresholds.get(task)
        print(f"  {task} ...", end="", flush=True)

        rdm_entries = load_task_rdms(task, success_threshold=threshold, metric=args.metric)
        if not rdm_entries:
            print(" [no RDMs found]")
            continue

        print(f" {len(rdm_entries)} networks", end="", flush=True)

        sort_idx, line_vals = get_sort_info(task)
        display_label = TASK_LABELS.get(task, task)

        fig = make_gallery(task, rdm_entries, sort_idx, line_vals, display_label, metric=args.metric)
        out_path = out_figures / f"rdm_gallery_{task}.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f" → {out_path.name}")


if __name__ == "__main__":
    main()
