#!/usr/bin/env python3
"""
RDM gallery through learning (Finding #3).

For each task: 15 successful + 5 failed networks.
Columns: step checkpoints in training order, then best (if available) and final,
         with actual step numbers shown in column headers.
Rows: networks sorted by performance (best at top, failed at bottom).

Output: output/analysis/figures/rdm_gallery_{task}.pdf
"""

import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANALYSIS  = Path(__file__).parent
REPO_ROOT = ANALYSIS.parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import RDM_DIR, TABLES_DIR, FIGURES_DIR, RL_TASKS, TASK_NAMES, get_depth, is_run_successful, task_run_dir

RNN_TASKS    = {"adding", "mnist_rnn"}
N_SUCCESS    = 15
N_FAILED     = 5
METRIC       = "pearson"  # default; overridden by --metric
CELL_W       = 0.80    # inches per column
CELL_H       = 0.85    # inches per row
LABEL_W      = 1.6     # left margin for run_id label
MAX_MAT_DIM  = 150     # downsample RDM matrix to this size for display


# ---------------------------------------------------------------------------
# Helpers
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
    return ds[:].astype(np.float32)


def vec_to_matrix(vec, max_dim=None):
    n = round((1 + np.sqrt(1 + 8 * len(vec))) / 2)
    mat = np.zeros((n, n), dtype=np.float32)
    ri, ci = np.triu_indices(n, k=1)
    mat[ri, ci] = vec
    mat += mat.T
    if max_dim and n > max_dim:
        idx = np.linspace(0, n - 1, max_dim).astype(int)
        mat = mat[np.ix_(idx, idx)]
    return mat


def _rdm_step_exists(rg, lkey):
    count = 0
    for name in rg.keys():
        if not name.startswith("step_"):
            continue
        ds = rg[name].get(lkey)
        if ds is not None and not ds.attrs.get("degenerate", False) and len(ds) > 0:
            count += 1
            if count >= 2:
                return True
    return False


def load_thresholds():
    data = json.load(open(TABLES_DIR / "success_thresholds.json"))
    return {k: float(v["upper"]) for k, v in data.items() if isinstance(v, dict)}


def load_metadata(task, run_id):
    """Return metadata dict from metadata.json, or {} if not found."""
    path = task_run_dir(task) / run_id / "metadata.json"
    if path.exists():
        return json.load(open(path))
    return {}


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_networks_gallery(task, thresholds, rng):
    h5_path   = RDM_DIR / f"{task}_rdms.h5"
    successful, failed = [], []

    with h5py.File(h5_path, "r") as f:
        for run_id, rg in f["runs"].items():
            if rg.attrs.get("is_repeat", False):
                continue
            perf = float(rg.attrs.get("performance", float("nan")))
            if not np.isfinite(perf):
                continue
            lkey = last_layer_key(task, rg)
            if not _rdm_step_exists(rg, lkey):
                continue
            if is_run_successful(task, rg, thresholds):
                successful.append((perf, run_id))
            else:
                failed.append((perf, run_id))

    successful.sort(key=lambda x: x[0])
    failed.sort(key=lambda x: x[0], reverse=True)   # best-failing first

    n_s = len(successful)
    if n_s == 0:
        print(f"  WARNING: no successful networks for {task}")
        success_ids = []
    elif n_s <= N_SUCCESS:
        success_ids = [rid for _, rid in successful]
    else:
        idx = np.round(np.linspace(0, n_s - 1, N_SUCCESS)).astype(int)
        success_ids = [successful[i][1] for i in idx]

    failed_ids = [rid for _, rid in failed[:N_FAILED]]

    print(f"  {len(success_ids)} successful, {len(failed_ids)} failed selected "
          f"(pool: {len(successful)} / {len(failed)})")
    return success_ids, failed_ids


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gallery_data(task, success_ids, failed_ids):
    """
    Returns list of dicts sorted by perf (best at top when reversed):
      {run_id, perf, is_successful,
       step_ckpts:  {step_int: matrix},
       best_mat:    matrix or None,
       best_step:   int or None,
       final_mat:   matrix or None,
       final_step:  int or None}
    """
    h5_path     = RDM_DIR / f"{task}_rdms.h5"
    all_ids     = list(success_ids) + list(failed_ids)
    success_set = set(success_ids)
    nets        = []

    with h5py.File(h5_path, "r") as f:
        for run_id in all_ids:
            if run_id not in f["runs"]:
                continue
            rg   = f["runs"][run_id]
            lkey = last_layer_key(task, rg)
            perf = float(rg.attrs.get("performance", float("nan")))

            # Step checkpoints
            step_ckpts = {}
            for name, cg in rg.items():
                if not name.startswith("step_"):
                    continue
                vec = load_rdm_vec(cg, lkey)
                if vec is not None:
                    step_ckpts[int(name[5:])] = vec_to_matrix(vec, max_dim=MAX_MAT_DIM)
            if len(step_ckpts) < 2:
                continue

            # best / final
            meta       = load_metadata(task, run_id)
            best_mat   = None
            best_step  = meta.get("best_step")
            final_mat  = None
            final_step = meta.get("final_step")

            best_cg = rg.get("best")
            if best_cg is not None:
                vec = load_rdm_vec(best_cg, lkey)
                if vec is not None:
                    best_mat = vec_to_matrix(vec, max_dim=MAX_MAT_DIM)

            final_cg = rg.get("final")
            if final_cg is not None:
                vec = load_rdm_vec(final_cg, lkey)
                if vec is not None:
                    final_mat = vec_to_matrix(vec, max_dim=MAX_MAT_DIM)

            nets.append({
                "run_id":        run_id,
                "perf":          perf,
                "is_successful": run_id in success_set,
                "step_ckpts":    step_ckpts,
                "best_mat":      best_mat,
                "best_step":     best_step,
                "final_mat":     final_mat,
                "final_step":    final_step,
            })

    nets.sort(key=lambda nd: nd["perf"])
    return nets


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_gallery_figure(task, nets):
    if not nets:
        return None

    all_steps  = sorted(set().union(*[nd["step_ckpts"].keys() for nd in nets]))
    has_best   = any(nd["best_mat"] is not None for nd in nets)
    has_final  = any(nd["final_mat"] is not None for nd in nets)
    n_nets     = len(nets)
    n_step_cols = len(all_steps)
    n_cols     = n_step_cols + int(has_best) + int(has_final)

    fig_w = LABEL_W + CELL_W * n_cols + 0.2
    fig_h = CELL_H * n_nets + 0.8

    fig   = plt.figure(figsize=(fig_w, fig_h))
    left0 = LABEL_W / fig_w
    col_w = CELL_W  / fig_w
    row_h = CELL_H  / fig_h
    top0  = 1.0 - 0.5 / fig_h

    # --- Column headers ---
    header_y = top0 - 0.22 / fig_h
    for col_idx, step in enumerate(all_steps):
        cx = left0 + col_idx * col_w + col_w * 0.5
        fig.text(cx, header_y, str(step), ha="center", va="bottom",
                 fontsize=4, transform=fig.transFigure)

    extra_col = n_step_cols
    if has_best:
        cx = left0 + extra_col * col_w + col_w * 0.5
        fig.text(cx, header_y, "BEST", ha="center", va="bottom",
                 fontsize=4.5, fontweight="bold", color="darkred",
                 transform=fig.transFigure)
        extra_col += 1
    if has_final:
        cx = left0 + extra_col * col_w + col_w * 0.5
        fig.text(cx, header_y, "FINAL", ha="center", va="bottom",
                 fontsize=4.5, fontweight="bold", color="darkblue",
                 transform=fig.transFigure)

    # "step →" label
    fig.text(left0 + n_step_cols * col_w * 0.5, top0,
             "Training step →", ha="center", va="top",
             fontsize=5.5, fontweight="bold", transform=fig.transFigure)

    # --- Rows ---
    for row_idx, nd in enumerate(reversed(nets)):
        row_top = top0 - 0.45 / fig_h - row_idx * row_h
        is_s    = nd["is_successful"]
        lbl_col = "#1a6600" if is_s else "#cc0000"
        marker  = "✓" if is_s else "✗"

        fig.text(LABEL_W * 0.98 / fig_w, row_top - row_h * 0.5,
                 f"{marker} {nd['run_id']}\nperf={nd['perf']:.3f}",
                 ha="right", va="center", fontsize=4.5,
                 color=lbl_col, transform=fig.transFigure)

        def add_cell(col_idx, mat, border_color=None, extra_lbl=None):
            left   = left0 + col_idx * col_w
            bottom = row_top - row_h
            ax = fig.add_axes([left + col_w * 0.04,
                               bottom + row_h * 0.06,
                               col_w * 0.90,
                               row_h * 0.86])
            ax.set_xticks([])
            ax.set_yticks([])
            if mat is not None:
                ax.imshow(mat, cmap="viridis", vmin=0, vmax=1,
                          aspect="equal", interpolation="nearest")
            else:
                ax.set_facecolor("#e0e0e0")
            ec = border_color or lbl_col
            lw = 1.2 if border_color else 0.5
            for spine in ax.spines.values():
                spine.set_edgecolor(ec)
                spine.set_linewidth(lw)
            if extra_lbl and mat is not None:
                ax.set_xlabel(extra_lbl, fontsize=3.5, labelpad=1, color=ec)

        # Step columns
        for col_idx, step in enumerate(all_steps):
            add_cell(col_idx, nd["step_ckpts"].get(step))

        # Best column
        extra_col = n_step_cols
        if has_best:
            step_lbl = f"s={nd['best_step']}" if nd.get("best_step") else ""
            add_cell(extra_col, nd["best_mat"], border_color="darkred", extra_lbl=step_lbl)
            extra_col += 1

        # Final column
        if has_final:
            step_lbl = f"s={nd['final_step']}" if nd.get("final_step") else ""
            add_cell(extra_col, nd["final_mat"], border_color="darkblue", extra_lbl=step_lbl)

    fig.suptitle(
        f"{task}  —  RDM gallery through learning  (last hidden layer, {METRIC} distance)\n"
        f"Rows: best→worst performance  |  ✓ green = successful  |  ✗ red = failed  "
        f"|  BEST = peak perf checkpoint  |  FINAL = end of training",
        fontsize=6.5, y=1.0,
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["cosine", "pearson"], default="pearson")
    parser.add_argument("--task", nargs="+", metavar="TASK",
                        help="Only generate gallery figures for these tasks.")
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
        h5_path = RDM_DIR / f"{task}_rdms.h5"
        if not h5_path.exists():
            print(f"{task}: no HDF5, skipping")
            continue
        print(f"{task} ...", flush=True)

        success_ids, failed_ids = select_networks_gallery(task, thresholds, rng)
        if not success_ids and not failed_ids:
            print(f"  no usable networks, skipping")
            continue

        nets = load_gallery_data(task, success_ids, failed_ids)
        fig  = make_gallery_figure(task, nets)
        if fig is None:
            print(f"  no figure produced")
            continue

        out = FIGURES_DIR / f"rdm_gallery_{task}.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=130)
        plt.close(fig)
        print(f"  Saved: {out.name}")


if __name__ == "__main__":
    main()
