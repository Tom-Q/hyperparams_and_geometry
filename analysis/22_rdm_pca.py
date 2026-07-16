#!/usr/bin/env python3
"""
Step 22: PCA on RDMs — Finding #2.3.

Flatten each network's RDM upper triangle into a vector, unit-normalize it,
stack into an N×D matrix, and run PCA (~20 components via randomized SVD).

RDM selection:
  - Supervised / RL : last hidden layer  (layer_{depth-1}_{metric})
  - RNN tasks       : temporal RDM       (temporal_{metric})
  - Adding          : fixed NaN pairs stripped before normalizing

Outputs:
    output/analysis/{metric}/figures/f2_rdm_pca.pdf
    output/analysis/{metric}/tables/rdm_pca_coords.csv
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ANALYSIS = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import RDM_DIR, TASK_NAMES, RL_TASKS, metric_output_dirs

RNN_TASKS = {"adding", "mnist_rnn"}
NAN_TASKS = {"adding"}
N_COMPONENTS = 20

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

# (column_name, log_scale)
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

# String keys — handles int/float CSV round-trips via _val_to_str()
CAT_COLORS = {
    "hp_optimizer":    {"adam": "#2271b2", "sgd": "#e05c00"},
    "hp_activation":   {"relu": "#2271b2", "sigmoid": "#e05c00", "tanh": "#2ba02b"},
    "hp_depth":        {"1": "#2271b2", "2": "#e05c00"},
    "hp_init_scale":   {"0.1": "#2271b2", "1.0": "#e05c00"},
    "hp_cell_type":    {"gru": "#2271b2", "rnn": "#e05c00"},
    "hp_n_rnn_layers": {"1": "#2271b2", "2": "#e05c00"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _val_to_str(v):
    """Normalise a scalar value to a string key for CAT_COLORS lookup."""
    try:
        fv = float(v)
        if fv == round(fv):
            return str(int(fv))
        return str(fv)
    except (TypeError, ValueError):
        return str(v)


def _rdm_key(task, depth, metric):
    if task in RNN_TASKS:
        return f"temporal_{metric}"
    return f"layer_{max(0, depth - 1)}_{metric}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_rdm_vectors(task, run_ids, metric="cosine"):
    """
    Load RDM upper-triangle vectors from HDF5 for the given run_ids.
    Strips fixed NaN pairs for the adding task.
    Returns dict run_id → float32 vector (only successfully loaded runs).
    """
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

    # Strip fixed NaN pairs for adding (NaN mask is identical across networks)
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
# PCA
# ---------------------------------------------------------------------------

def run_pca(vectors, n_components=N_COMPONENTS):
    """
    Unit-normalize RDM vectors, stack into matrix, run PCA.
    Returns (pca_object, coords array N×n_components, list of run_ids).
    """
    run_ids = sorted(vectors.keys())
    mat = np.vstack([vectors[rid] for rid in run_ids]).astype(np.float64)

    # Unit-normalize each row
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    mat /= norms

    n_comp = min(n_components, mat.shape[0] - 1, mat.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="auto", random_state=42)
    coords = pca.fit_transform(mat)
    return pca, coords, run_ids


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _cat_hps_for_task(task):
    if task in RNN_TASKS:
        return CAT_HPS_RNN
    if task in RL_TASKS:
        return CAT_HPS_RL
    return CAT_HPS_SUPERVISED


def _hp_list(hp_df, task):
    """Return ordered list of (col, log_scale_or_None, kind) for HPs present in hp_df."""
    cont = [(h, log, "cont") for h, log in CONT_HPS
            if h in hp_df.columns and not (h == "hp_batch_size" and task in RL_TASKS)]
    cats = [(h, None, "cat") for h in _cat_hps_for_task(task) if h in hp_df.columns]
    return cont + cats


def make_task_page(pca, coords, hp_df, task):
    """
    Return a matplotlib Figure for one task:
      - Top: scree plot (variance explained per PC)
      - Below: PC1 vs PC2 scatter, one panel per HP, colored by HP value
    """
    hp_list = _hp_list(hp_df, task)
    n_hp = len(hp_list)
    n_cols = 3
    n_scatter_rows = max(1, (n_hp + n_cols - 1) // n_cols)

    fig_h = max(7, 2.5 + 3.2 * n_scatter_rows)
    fig = plt.figure(figsize=(13, fig_h))

    height_ratios = [1.6] + [2.0] * n_scatter_rows
    gs = fig.add_gridspec(
        1 + n_scatter_rows, n_cols,
        height_ratios=height_ratios,
        hspace=0.6, wspace=0.38,
        top=0.93, bottom=0.04, left=0.07, right=0.97,
    )

    # --- Scree plot ---
    ax_scree = fig.add_subplot(gs[0, :])
    n_show = min(15, pca.n_components_)
    var_pct = pca.explained_variance_ratio_[:n_show] * 100
    ax_scree.bar(range(1, n_show + 1), var_pct, color="#2271b2", alpha=0.85, width=0.7)
    ax_scree.set_xlabel("PC", fontsize=8)
    ax_scree.set_ylabel("Var. explained (%)", fontsize=8)
    ax_scree.set_xticks(range(1, n_show + 1))
    ax_scree.set_xticklabels([str(i) for i in range(1, n_show + 1)], fontsize=7)
    ax_scree.tick_params(labelsize=7)

    ax_cum = ax_scree.twinx()
    ax_cum.plot(range(1, n_show + 1), np.cumsum(var_pct), "o-",
                color="#e05c00", ms=4, lw=1.2)
    ax_cum.set_ylabel("Cumulative (%)", fontsize=8, color="#e05c00")
    ax_cum.tick_params(labelsize=7, labelcolor="#e05c00")
    ax_cum.set_ylim(0, 105)

    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100 if pca.n_components_ > 1 else 0
    ax_scree.set_title(
        f"{TASK_SHORT.get(task, task)}  |  N = {len(coords):,} networks  "
        f"|  PC1 = {pc1_var:.1f}%,  PC2 = {pc2_var:.1f}%",
        fontsize=9, fontweight="bold",
    )

    # --- HP scatter panels ---
    pc1 = coords[:, 0]
    pc2 = coords[:, 1] if coords.shape[1] > 1 else np.zeros_like(pc1)

    xl = f"PC1 ({pc1_var:.1f}%)"
    yl = f"PC2 ({pc2_var:.1f}%)"

    for idx, (hp_col, log_flag, kind) in enumerate(hp_list):
        row = 1 + idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        if kind == "cont":
            vals = hp_df[hp_col].values.astype(float)
            mask = np.isfinite(vals) & np.isfinite(pc1) & np.isfinite(pc2)
            if mask.sum() < 3:
                ax.set_visible(False)
                continue

            import matplotlib.colors as mcolors
            vmin, vmax = vals[mask].min(), vals[mask].max()
            if log_flag and vmin > 0 and vmin < vmax:
                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            sc = ax.scatter(pc1[mask], pc2[mask], c=vals[mask], cmap="viridis",
                            norm=norm, s=6, alpha=0.5, linewidths=0)
            cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.03)
            cb.ax.tick_params(labelsize=6)

        else:  # categorical
            vals_raw = hp_df[hp_col]
            colors_map = CAT_COLORS.get(hp_col, {})
            default_tab10 = plt.cm.tab10.colors

            # Unique categories sorted for deterministic order
            unique_cats = sorted(vals_raw.dropna().unique(), key=_val_to_str)
            for i, cat in enumerate(unique_cats):
                mask = vals_raw == cat
                key = _val_to_str(cat)
                color = colors_map.get(key, default_tab10[i % len(default_tab10)])
                ax.scatter(pc1[mask], pc2[mask], c=[color], s=6, alpha=0.5,
                           linewidths=0, label=key)

            ax.legend(fontsize=6, markerscale=2.5, framealpha=0.7,
                      loc="best", handlelength=1.0)

        ax.set_xlabel(xl, fontsize=7)
        ax.set_ylabel(yl, fontsize=7)
        ax.set_title(CAT_LABELS.get(hp_col, CONT_LABELS.get(hp_col, hp_col)), fontsize=8)
        ax.tick_params(labelsize=6)

    # Hide unused grid slots
    for idx in range(n_hp, n_scatter_rows * n_cols):
        row = 1 + idx // n_cols
        col = idx % n_cols
        try:
            ax = fig.add_subplot(gs[row, col])
            ax.set_visible(False)
        except Exception:
            pass

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PCA on network RDMs.")
    parser.add_argument("--metric", choices=["cosine", "pearson"], default="cosine",
                        help="RDM metric (default: cosine).")
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

    pdf_path = out_figures / "f2_rdm_pca.pdf"
    with PdfPages(pdf_path) as pdf:
        for task in TASK_NAMES:
            task_df = all_df[all_df["task"] == task].copy()
            if len(task_df) < 10:
                print(f"  [skip] {task}: only {len(task_df)} networks in stats CSV")
                continue

            run_ids = task_df["run_id"].tolist()
            print(f"  {task}: loading {len(run_ids)} RDM vectors ...", flush=True)

            vectors = load_rdm_vectors(task, run_ids, metric=args.metric)
            if len(vectors) < 10:
                print(f"    [skip] only {len(vectors)} vectors loaded (HDF5 key missing?)")
                continue

            print(f"    {len(vectors)} loaded  →  running PCA ...", flush=True)
            pca, coords, ordered_ids = run_pca(vectors)

            # Align hp_df rows to PCA ordering
            id_set = set(ordered_ids)
            hp_df = (task_df[task_df["run_id"].isin(id_set)]
                     .set_index("run_id")
                     .loc[ordered_ids]
                     .reset_index())

            print(f"    PC1={pca.explained_variance_ratio_[0]*100:.1f}%  "
                  f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%  "
                  f"(top-{pca.n_components_} components)")

            # Accumulate coordinate rows for CSV
            for i, run_id in enumerate(ordered_ids):
                row = {"task": task, "run_id": run_id}
                for pc_idx in range(coords.shape[1]):
                    row[f"pc{pc_idx + 1}"] = float(coords[i, pc_idx])
                all_coord_rows.append(row)

            fig = make_task_page(pca, coords, hp_df, task)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"    Page added to PDF.")

    print(f"\nSaved: {pdf_path}")

    coords_df = pd.DataFrame(all_coord_rows)
    coords_path = out_tables / "rdm_pca_coords.csv"
    coords_df.to_csv(coords_path, index=False)
    print(f"Saved: {coords_path}")


if __name__ == "__main__":
    main()
