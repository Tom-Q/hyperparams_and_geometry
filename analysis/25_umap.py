#!/usr/bin/env python3
"""
Step 25: UMAP of networks by RDM similarity — Finding #2.6.

For each task, compute pairwise RDM-to-RDM distances (1 − Spearman r between
upper-triangle vectors) across all successful primary networks, then embed in
2D with UMAP (metric="precomputed").

RDM selection:
  - Supervised / RL : last hidden layer  (layer_{depth-1}_{metric})
  - RNN tasks       : temporal RDM       (temporal_{metric})
  - Adding          : fixed NaN pairs stripped before computing distances

Scatter plots are coloured by each HP (viridis / discrete palette) and by
each of the four RDM summary statistics.

Outputs:
    output/analysis/{metric}/figures/f2_umap.pdf
    output/analysis/{metric}/tables/rdm_umap_coords.csv
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import umap

ANALYSIS = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import RDM_DIR, TASK_NAMES, RL_TASKS, metric_output_dirs

RNN_TASKS = {"adding", "mnist_rnn"}
NAN_TASKS = {"adding"}

UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1
RANDOM_STATE     = 42

TASK_SHORT = {
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

CONT_HPS = [
    ("hp_learning_rate", True),
    ("hp_l1_reg",        True),
    ("hp_l2_reg",        True),
    ("hp_hidden_size",   False),
    ("hp_batch_size",    False),
]
CONT_LABELS = {
    "hp_learning_rate": "learning rate",
    "hp_l1_reg":        "l1 reg",
    "hp_l2_reg":        "l2 reg",
    "hp_hidden_size":   "hidden size",
    "hp_batch_size":    "batch size",
}

CAT_HPS_SUPERVISED = ["hp_optimizer", "hp_activation", "hp_depth", "hp_init_scale"]
CAT_HPS_RNN        = ["hp_optimizer", "hp_cell_type",  "hp_n_rnn_layers", "hp_init_scale"]
CAT_HPS_RL         = ["hp_optimizer", "hp_activation", "hp_depth", "hp_init_scale"]
CAT_LABELS = {
    "hp_optimizer":    "optimizer",
    "hp_activation":   "activation",
    "hp_depth":        "depth",
    "hp_init_scale":   "init scale",
    "hp_cell_type":    "cell type",
    "hp_n_rnn_layers": "n rnn layers",
}
CAT_COLORS = {
    "hp_optimizer":    {"adam": "#2271b2", "sgd": "#e05c00"},
    "hp_activation":   {"relu": "#2271b2", "sigmoid": "#e05c00", "tanh": "#2ba02b"},
    "hp_depth":        {"1": "#2271b2", "2": "#e05c00"},
    "hp_init_scale":   {"0.1": "#2271b2", "1.0": "#e05c00"},
    "hp_cell_type":    {"gru": "#2271b2", "rnn": "#e05c00"},
    "hp_n_rnn_layers": {"1": "#2271b2", "2": "#e05c00"},
}

RDM_PROPS = ["reliability", "category_corr", "dimensionality", "mean_dissimilarity"]
RDM_LABELS = {
    "reliability":        "reliability",
    "category_corr":      "category corr",
    "dimensionality":     "dimensionality",
    "mean_dissimilarity": "mean dissim.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _val_to_str(v):
    try:
        fv = float(v)
        return str(int(fv)) if fv == round(fv) else str(fv)
    except (TypeError, ValueError):
        return str(v)


def _rdm_key(task, depth, metric):
    if task in RNN_TASKS:
        return f"temporal_{metric}"
    return f"layer_{max(0, depth - 1)}_{metric}"


def _cat_hps_for_task(task):
    if task in RNN_TASKS: return CAT_HPS_RNN
    if task in RL_TASKS:  return CAT_HPS_RL
    return CAT_HPS_SUPERVISED


# ---------------------------------------------------------------------------
# Data loading  (same pattern as script 22)
# ---------------------------------------------------------------------------

def load_rdm_vectors(task, run_ids, metric="cosine"):
    h5_path = RDM_DIR / f"{task}_rdms.h5"
    if not h5_path.exists():
        return {}

    ckpt = "final" if task in RL_TASKS else "best"
    vectors = {}

    with h5py.File(h5_path, "r") as h5:
        runs_grp = h5.get("runs", {})
        for run_id in run_ids:
            rg = runs_grp.get(run_id)
            if rg is None:
                continue
            cg = rg.get(ckpt)
            if cg is None:
                continue
            depth = int(rg.attrs.get("hp_depth", 1))
            key = _rdm_key(task, depth, metric)
            if key not in cg:
                continue
            ds = cg[key]
            if ds.attrs.get("degenerate", False) or len(ds) == 0:
                continue
            vectors[run_id] = ds[:].astype(np.float32)

    if not vectors:
        return {}

    if task in NAN_TASKS:
        sample = next(iter(vectors.values()))
        valid = np.isfinite(sample)
        vectors = {rid: v[valid] for rid, v in vectors.items()}
    else:
        for rid, v in vectors.items():
            if not np.all(np.isfinite(v)):
                raise ValueError(f"{task}/{rid}: unexpected NaN in RDM")

    return vectors


# ---------------------------------------------------------------------------
# Pairwise Spearman distance
# ---------------------------------------------------------------------------

def pairwise_spearman_dist(mat):
    """
    mat: (N, D) float32 RDM vectors.
    Returns (N, N) float32 distance matrix: 1 − Spearman r.
    Uses rank-transform then normalised dot product (= Pearson on ranks = Spearman r).
    """
    N, D = mat.shape
    # Rank-transform each row via double argsort
    order = mat.argsort(axis=1)
    ranks = order.argsort(axis=1).astype(np.float32)

    # Centre and unit-normalise each row
    ranks -= ranks.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(ranks, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    ranks /= norms

    # Pairwise dot product = Spearman r
    corr = (ranks @ ranks.T).astype(np.float32)
    np.clip(corr, -1.0, 1.0, out=corr)
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    return dist


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _hp_list(hp_df, task):
    cont = [(h, log, "cont") for h, log in CONT_HPS
            if h in hp_df.columns and not (h == "hp_batch_size" and task in RL_TASKS)]
    cats = [(h, None, "cat") for h in _cat_hps_for_task(task) if h in hp_df.columns]
    return cont + cats


def _prop_list(hp_df):
    return [(p, RDM_LABELS[p], "prop") for p in RDM_PROPS if p in hp_df.columns]


def make_task_page(embedding, hp_df, task):
    """Return a figure with UMAP scatter coloured by each HP and RDM property."""
    hp_list   = _hp_list(hp_df, task)
    prop_list = _prop_list(hp_df)
    all_panels = hp_list + prop_list

    n_panels = len(all_panels)
    n_cols   = 3
    n_rows   = max(1, (n_panels + n_cols - 1) // n_cols)

    fig_h = max(6, 3.0 * n_rows + 0.8)
    fig = plt.figure(figsize=(13, fig_h))
    gs  = fig.add_gridspec(n_rows, n_cols, hspace=0.55, wspace=0.35,
                           top=0.92, bottom=0.04, left=0.05, right=0.97)

    u1 = embedding[:, 0]
    u2 = embedding[:, 1]

    n = len(hp_df)
    n_neighbors = min(UMAP_N_NEIGHBORS, n - 1)

    fig.suptitle(
        f"{TASK_SHORT.get(task, task)}  |  N={n:,}  |  "
        f"UMAP n_neighbors={n_neighbors}, min_dist={UMAP_MIN_DIST}",
        fontsize=9, fontweight="bold",
    )

    for idx, panel in enumerate(all_panels):
        row = idx // n_cols
        col = idx % n_cols
        ax  = fig.add_subplot(gs[row, col])

        col_name, log_or_label, kind = panel

        if kind == "cont":
            log_flag = log_or_label
            vals = hp_df[col_name].values.astype(float)
            mask = np.isfinite(vals)
            if mask.sum() < 3:
                ax.set_visible(False)
                continue
            vmin, vmax = vals[mask].min(), vals[mask].max()
            if log_flag and vmin > 0 and vmin < vmax:
                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            sc = ax.scatter(u1, u2, c=vals, cmap="viridis", norm=norm,
                            s=6, alpha=0.5, linewidths=0)
            cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.03)
            cb.ax.tick_params(labelsize=6)
            ax.set_title(CONT_LABELS.get(col_name, col_name), fontsize=8)

        elif kind == "cat":
            vals_raw  = hp_df[col_name]
            colors_map = CAT_COLORS.get(col_name, {})
            tab10 = plt.cm.tab10.colors
            unique_cats = sorted(vals_raw.dropna().unique(), key=_val_to_str)
            for i, cat in enumerate(unique_cats):
                mask = vals_raw == cat
                key  = _val_to_str(cat)
                c    = colors_map.get(key, tab10[i % len(tab10)])
                ax.scatter(u1[mask], u2[mask], c=[c], s=6, alpha=0.5,
                           linewidths=0, label=key)
            ax.legend(fontsize=6, markerscale=2.5, framealpha=0.7,
                      loc="best", handlelength=1.0)
            ax.set_title(CAT_LABELS.get(col_name, col_name), fontsize=8)

        else:  # RDM property
            label = log_or_label
            vals = hp_df[col_name].values.astype(float)
            mask = np.isfinite(vals)
            if mask.sum() < 3:
                ax.set_visible(False)
                continue
            sc = ax.scatter(u1, u2, c=vals, cmap="plasma",
                            s=6, alpha=0.5, linewidths=0)
            cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.03)
            cb.ax.tick_params(labelsize=6)
            ax.set_title(label, fontsize=8)

        ax.set_xlabel("UMAP 1", fontsize=7)
        ax.set_ylabel("UMAP 2", fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused slots
    for idx in range(n_panels, n_rows * n_cols):
        try:
            ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
            ax.set_visible(False)
        except Exception:
            pass

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UMAP of networks by RDM similarity.")
    parser.add_argument("--metric", choices=["cosine", "pearson"], default="cosine")
    args = parser.parse_args()

    out_figures, out_tables = metric_output_dirs(args.metric)
    out_figures.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    stats_path = out_tables / "rdm_per_network_stats.csv"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"{stats_path} not found — run 20_hp_effects.py --metric {args.metric} first.")

    print("Loading per-network stats ...")
    all_df = pd.read_csv(stats_path)

    all_coord_rows = []

    pdf_path = out_figures / "f2_umap.pdf"
    with PdfPages(pdf_path) as pdf:
        for task in TASK_NAMES:
            task_df = all_df[all_df["task"] == task].copy()
            if len(task_df) < 10:
                print(f"  [skip] {task}: only {len(task_df)} networks")
                continue

            run_ids = task_df["run_id"].tolist()
            print(f"  {task}: loading {len(run_ids)} RDM vectors ...", flush=True)

            vectors = load_rdm_vectors(task, run_ids, metric=args.metric)
            if len(vectors) < 10:
                print(f"    [skip] only {len(vectors)} vectors loaded")
                continue

            print(f"    {len(vectors)} loaded — computing pairwise distances ...",
                  flush=True)
            ordered_ids = sorted(vectors.keys())
            mat = np.vstack([vectors[rid] for rid in ordered_ids])
            dist_mat = pairwise_spearman_dist(mat)

            n_neighbors = min(UMAP_N_NEIGHBORS, len(ordered_ids) - 1)
            print(f"    running UMAP (n_neighbors={n_neighbors}) ...", flush=True)
            reducer = umap.UMAP(
                n_components=2,
                metric="precomputed",
                n_neighbors=n_neighbors,
                min_dist=UMAP_MIN_DIST,
                random_state=RANDOM_STATE,
            )
            embedding = reducer.fit_transform(dist_mat)

            # Align HP/property DataFrame to UMAP ordering
            id_set = set(ordered_ids)
            hp_df = (task_df[task_df["run_id"].isin(id_set)]
                     .set_index("run_id")
                     .loc[ordered_ids]
                     .reset_index())

            for i, run_id in enumerate(ordered_ids):
                all_coord_rows.append({
                    "task": task, "run_id": run_id,
                    "umap1": float(embedding[i, 0]),
                    "umap2": float(embedding[i, 1]),
                })

            print(f"    making figure ...", flush=True)
            fig = make_task_page(embedding, hp_df, task)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"    done.")

    print(f"\nSaved: {pdf_path}")

    coords_df = pd.DataFrame(all_coord_rows)
    coords_path = out_tables / "rdm_umap_coords.csv"
    coords_df.to_csv(coords_path, index=False)
    print(f"Saved: {coords_path}")


if __name__ == "__main__":
    main()
