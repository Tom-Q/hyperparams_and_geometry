#!/usr/bin/env python3
"""
Step 8 (sanity check): HP-space UMAP.

Embed all primary networks in 2D using UMAP on their HP configurations.
Distance metric: Gower distance, which gives each HP equal weight regardless
of type:
  - Continuous HP : |x_i - y_i|  (unit_* columns are already in [0,1])
  - Categorical HP: 0 if same, 1 if different

With p HPs total, each contributes at most 1/p to the total distance.
This avoids the bias that Euclidean + one-hot encoding introduces, where each
categorical HP contributes up to √2 while continuous HPs contribute at most 1.

The N×N Gower distance matrix is passed to UMAP as metric="precomputed".
For N=800, this is ~2.5 MB — trivial.

Includes all primary networks (successful and failed), so the embedding shows
where BO explored and where successful networks cluster.

Scatter panels per task:
  - Each continuous HP (viridis, log scale for lr/l1/l2)
  - Each categorical HP (discrete colours)
  - Performance (plasma, continuous)
  - Success vs. failure (binary; threshold from task_meta)

Outputs:
    output/analysis/figures/sc_hp_umap.pdf
    output/analysis/tables/sc_hp_umap_coords.csv
"""

import sys
from pathlib import Path

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
from analysis_utils import (
    FIGURES_DIR,
    TABLES_DIR,
    TASK_NAMES,
    load_task_df,
    primary_df,
    task_meta,
)

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

HP_LABELS = {
    "learning_rate": "learning rate",
    "l1_reg":        "l1 reg",
    "l2_reg":        "l2 reg",
    "hidden_size":   "hidden size",
    "batch_size":    "batch size",
    "depth":         "depth",
    "activation":    "activation",
    "optimizer":     "optimizer",
    "init_scale":    "init scale",
    "cell_type":     "cell type",
    "n_rnn_layers":  "n rnn layers",
}

CONT_HP_LOG = {
    "learning_rate": True,
    "l1_reg":        True,
    "l2_reg":        True,
    "hidden_size":   False,
    "batch_size":    False,
}

# Keys match the actual stored values (int/float, not str) in the DataFrame
CAT_COLORS = {
    "optimizer":    {"adam": "#2271b2", "sgd": "#e05c00"},
    "activation":   {"relu": "#2271b2", "sigmoid": "#e05c00", "tanh": "#2ba02b"},
    "depth":        {1: "#2271b2", 2: "#e05c00"},
    "init_scale":   {0.1: "#2271b2", 1.0: "#e05c00"},
    "cell_type":    {"gru": "#2271b2", "rnn": "#e05c00"},
    "n_rnn_layers": {1: "#2271b2", 2: "#e05c00"},
}


# ---------------------------------------------------------------------------
# Feature / distance matrix builders
# ---------------------------------------------------------------------------

def cont_feature_matrix(df: pd.DataFrame, meta: dict) -> np.ndarray:
    """
    Continuous-only feature matrix for UMAP (Euclidean metric).
    Uses unit_{hp} columns (log-normalised [0,1]).
    Returns float32 array of shape (N, n_cont).
    """
    cont_names = meta["cont_param_names"]
    unit_cols  = [f"unit_{c}" for c in cont_names if f"unit_{c}" in df.columns]
    return df[unit_cols].values.astype(np.float32)


def gower_distance_matrix(df: pd.DataFrame, meta: dict) -> np.ndarray:
    """
    Compute pairwise Gower distance matrix for mixed continuous/categorical HPs.

    Each HP contributes equally (weight 1/n_HPs) to the total distance:
      - Continuous: |x_i - y_i|  (unit_* columns are in [0,1], so max = 1)
      - Categorical: 0 if same, 1 if different

    Returns float32 (N, N) distance matrix with zeros on the diagonal.
    """
    cont_names = meta["cont_param_names"]
    cat_names  = meta["cat_param_names"]

    unit_cols = [f"unit_{c}" for c in cont_names if f"unit_{c}" in df.columns]
    cont_mat  = df[unit_cols].values.astype(np.float32)       # (N, n_cont)
    cat_cols  = [c for c in cat_names if c in df.columns]

    N       = len(df)
    n_total = len(unit_cols) + len(cat_cols)
    dist    = np.zeros((N, N), dtype=np.float32)

    for j in range(cont_mat.shape[1]):
        col   = cont_mat[:, j]
        dist += np.abs(col[:, None] - col[None, :])

    for cat in cat_cols:
        vals  = np.asarray(df[cat])
        dist += (vals[:, None] != vals[None, :]).astype(np.float32)

    dist /= n_total
    np.fill_diagonal(dist, 0.0)
    return dist


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_task_page(embedding: np.ndarray, df: pd.DataFrame,
                   task: str, meta: dict, embed_label: str) -> plt.Figure:
    """Scatter plots coloured by each HP, performance, and success/failure."""
    cont_names = meta["cont_param_names"]
    cat_names  = meta["cat_param_names"]
    threshold  = meta["success_threshold"]

    panels = []
    for c in cont_names:
        if c in df.columns:
            panels.append(("cont", c))
    for c in cat_names:
        if c in df.columns:
            panels.append(("cat", c))
    panels.append(("perf",    "performance"))
    panels.append(("success", "success"))

    n_panels = len(panels)
    n_cols   = 3
    n_rows   = (n_panels + n_cols - 1) // n_cols

    fig_h = max(6, 3.2 * n_rows + 0.8)
    fig   = plt.figure(figsize=(13, fig_h))

    n_neighbors = min(UMAP_N_NEIGHBORS, len(df) - 1)
    fig.suptitle(
        f"{TASK_SHORT.get(task, task)}  |  N={len(df):,}  |  "
        f"{embed_label}  |  "
        f"n_neighbors={n_neighbors}, min_dist={UMAP_MIN_DIST}",
        fontsize=9, fontweight="bold",
    )
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.6, wspace=0.38,
                          top=0.92, bottom=0.04, left=0.05, right=0.97)

    u1, u2 = embedding[:, 0], embedding[:, 1]
    is_success = df["performance"].values >= threshold

    for idx, (kind, col) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])

        if kind == "cont":
            vals = df[col].values.astype(float)
            mask = np.isfinite(vals)
            if mask.sum() < 3:
                ax.set_visible(False)
                continue
            vmin, vmax = vals[mask].min(), vals[mask].max()
            log_flag   = CONT_HP_LOG.get(col, False)
            if log_flag and vmin > 0 and vmin < vmax:
                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            sc = ax.scatter(u1, u2, c=vals, cmap="viridis", norm=norm,
                            s=5, alpha=0.5, linewidths=0)
            cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.03)
            cb.ax.tick_params(labelsize=6)
            ax.set_title(HP_LABELS.get(col, col), fontsize=8)

        elif kind == "cat":
            colors_map  = CAT_COLORS.get(col, {})
            tab10       = plt.cm.tab10.colors
            unique_vals = sorted(df[col].dropna().unique(), key=str)
            for i, val in enumerate(unique_vals):
                mask = df[col] == val
                c    = colors_map.get(val, tab10[i % len(tab10)])
                ax.scatter(u1[mask], u2[mask], c=[c], s=5, alpha=0.5,
                           linewidths=0, label=str(val))
            ax.legend(fontsize=6, markerscale=2.5, framealpha=0.7,
                      loc="best", handlelength=1.0)
            ax.set_title(HP_LABELS.get(col, col), fontsize=8)

        elif kind == "perf":
            vals = df["performance"].values.astype(float)
            sc   = ax.scatter(u1, u2, c=vals, cmap="plasma",
                              s=5, alpha=0.5, linewidths=0)
            cb   = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.03)
            cb.ax.tick_params(labelsize=6)
            ax.set_title("performance", fontsize=8)

        else:  # success / failure
            ax.scatter(u1[is_success],  u2[is_success],  c="#2271b2", s=5,
                       alpha=0.5, linewidths=0,
                       label=f"success (≥{threshold:.2f})")
            ax.scatter(u1[~is_success], u2[~is_success], c="#e05c00", s=5,
                       alpha=0.5, linewidths=0, label="failure")
            ax.legend(fontsize=6, markerscale=2.5, framealpha=0.7,
                      loc="best", handlelength=1.0)
            ax.set_title("success / failure", fontsize=8)

        ax.set_xlabel("UMAP 1", fontsize=7)
        ax.set_ylabel("UMAP 2", fontsize=7)
        ax.tick_params(labelsize=6)

    for idx in range(n_panels, n_rows * n_cols):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        ax.set_visible(False)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Each mode: (pdf_stem, coords_stem, embed_label, umap_metric, builder_fn)
# builder_fn(df, meta) returns either a feature matrix (euclidean) or a
# precomputed distance matrix (precomputed).
MODES = [
    (
        "sc_hp_umap_cont",
        "continuous HPs only (Euclidean)",
        "euclidean",
        cont_feature_matrix,
    ),
    (
        "sc_hp_umap_gower",
        "Gower distance (cont + cat, equal weights)",
        "precomputed",
        gower_distance_matrix,
    ),
]


def run_mode(stem, embed_label, umap_metric, builder_fn, meta_all):
    all_coord_rows = []
    pdf_path    = FIGURES_DIR / f"{stem}.pdf"
    coords_path = TABLES_DIR  / f"{stem}_coords.csv"

    print(f"\n=== {embed_label} ===")
    with PdfPages(pdf_path) as pdf:
        for task in TASK_NAMES:
            meta    = meta_all[task]
            df      = load_task_df(task)
            df_prim = primary_df(df).reset_index(drop=True)

            if len(df_prim) < 10:
                print(f"  [skip] {task}: only {len(df_prim)} primary networks")
                continue

            print(f"  {task}: {len(df_prim)} networks ...", flush=True)
            X = builder_fn(df_prim, meta)

            n_neighbors = min(UMAP_N_NEIGHBORS, len(df_prim) - 1)
            reducer = umap.UMAP(
                n_components=2,
                metric=umap_metric,
                n_neighbors=n_neighbors,
                min_dist=UMAP_MIN_DIST,
                random_state=RANDOM_STATE,
            )
            embedding = reducer.fit_transform(X)

            for i in range(len(df_prim)):
                all_coord_rows.append({
                    "task":        task,
                    "iteration":   int(df_prim.loc[i, "iteration"]),
                    "umap1":       float(embedding[i, 0]),
                    "umap2":       float(embedding[i, 1]),
                    "performance": float(df_prim.loc[i, "performance"]),
                    "success":     bool(df_prim.loc[i, "performance"] >= meta["success_threshold"]),
                })

            fig = make_task_page(embedding, df_prim, task, meta, embed_label)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"  Saved: {pdf_path}")
    pd.DataFrame(all_coord_rows).to_csv(coords_path, index=False)
    print(f"  Saved: {coords_path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    meta_all = task_meta()
    for stem, embed_label, umap_metric, builder_fn in MODES:
        run_mode(stem, embed_label, umap_metric, builder_fn, meta_all)


if __name__ == "__main__":
    main()
