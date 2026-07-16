#!/usr/bin/env python3
"""
Step 21: Latent variable analysis — Finding #2.2.

Three theory-driven composite HP variables (equal weights, z-scored continuous HPs),
defined per paradigm:

  Stability    : −lr, +batch_size*, +optimizer(adam=+1/sgd=−1),
                  +init_scale(0.1=+1/1.0=−1), +l2_reg
  Capacity     : +hidden_size, +depth(2=+1/1=−1)
                  [RNN: n_rnn_layers instead of depth]
  Regularization: +l1_reg, +l2_reg

  * batch_size absent for RL tasks (online Q-learning)

Continuous HPs are z-scored within task before summing. The composite sum is then
z-scored within task. Spearman r is computed between each composite score and each
RDM property across successful primary networks.

Outputs:
    output/analysis/{metric}/figures/f2_latent_vars.pdf     — heatmap summary
    output/analysis/{metric}/figures/f2_latent_vars_scatter.pdf — per-task scatters
    output/analysis/{metric}/tables/rdm_latent_vars.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ANALYSIS = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import TABLES_DIR, TASK_NAMES, RL_TASKS, metric_output_dirs

RNN_TASKS = {"adding", "mnist_rnn"}

COMPOSITES = ["stability", "capacity", "regularization"]

COMPOSITE_LABELS = {
    "stability":      "Stability",
    "capacity":       "Capacity",
    "regularization": "Regularization",
}

# Component definitions per paradigm.
# Each entry: (hp_col, kind, param)
#   kind="cont": param is sign (+1 or -1); HP is z-scored then multiplied by sign
#   kind="cat":  param is dict mapping str(value) → score
#
# Note: _hp_val() in script 20 converts integer-valued floats to int before saving,
# so init_scale=1.0 → "1" and depth=2 → "2" when cast to str.
COMPOSITE_DEFS = {
    "supervised": {
        "stability": [
            ("hp_learning_rate", "cont", -1),
            ("hp_batch_size",    "cont", +1),
            ("hp_optimizer",     "cat",  {"adam": +1, "sgd": -1}),
            ("hp_init_scale",    "cat",  {"0.1": +1, "1": -1}),
            ("hp_l2_reg",        "cont", +1),
        ],
        "capacity": [
            ("hp_hidden_size",   "cont", +1),
            ("hp_depth",         "cat",  {"1": -1, "2": +1}),
        ],
        "regularization": [
            ("hp_l1_reg",        "cont", +1),
            ("hp_l2_reg",        "cont", +1),
        ],
    },
    "rnn": {
        "stability": [
            ("hp_learning_rate", "cont", -1),
            ("hp_batch_size",    "cont", +1),
            ("hp_optimizer",     "cat",  {"adam": +1, "sgd": -1}),
            ("hp_init_scale",    "cat",  {"0.1": +1, "1": -1}),
            ("hp_l2_reg",        "cont", +1),
        ],
        "capacity": [
            ("hp_hidden_size",   "cont", +1),
            ("hp_n_rnn_layers",  "cat",  {"1": -1, "2": +1}),
        ],
        "regularization": [
            ("hp_l1_reg",        "cont", +1),
            ("hp_l2_reg",        "cont", +1),
        ],
    },
    "rl": {
        "stability": [
            ("hp_learning_rate", "cont", -1),
            ("hp_optimizer",     "cat",  {"adam": +1, "sgd": -1}),
            ("hp_init_scale",    "cat",  {"0.1": +1, "1": -1}),
            ("hp_l2_reg",        "cont", +1),
        ],
        "capacity": [
            ("hp_hidden_size",   "cont", +1),
            ("hp_depth",         "cat",  {"1": -1, "2": +1}),
        ],
        "regularization": [
            ("hp_l1_reg",        "cont", +1),
            ("hp_l2_reg",        "cont", +1),
        ],
    },
}

RDM_PROPS = ["reliability", "category_corr", "dimensionality", "mean_dissimilarity"]
RDM_LABELS = {
    "reliability":        "reliability",
    "category_corr":      "category\ncorr",
    "dimensionality":     "dimension-\nality",
    "mean_dissimilarity": "mean\ndissim.",
}

PARADIGMS = [
    ("Supervised", ["mnist_dual", "mnist_10way", "fashion_10way", "spirals", "parity"]),
    ("RNN",        ["adding", "mnist_rnn"]),
    ("RL",         ["cartpole", "fourrooms"]),
]

TASK_SHORT = {
    "mnist_dual":    "MNIST\ndual",
    "mnist_10way":   "MNIST\n10way",
    "fashion_10way": "Fashion\n10way",
    "spirals":       "Spirals",
    "parity":        "Parity",
    "adding":        "Adding",
    "mnist_rnn":     "MNIST\nRNN",
    "cartpole":      "CartPole",
    "fourrooms":     "FourRooms",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_paradigm(task):
    if task in RNN_TASKS:
        return "rnn"
    if task in RL_TASKS:
        return "rl"
    return "supervised"


def _col_to_str(col):
    """Normalise a column to string keys for dict mapping.
    Integer-valued floats (1.0, 2.0) → "1", "2" so they match int keys.
    Other values → str as-is.
    """
    try:
        fvals = col.astype(float)
        result = fvals.astype(str)
        int_mask = fvals.notna() & (fvals == fvals.round())
        result[int_mask] = fvals[int_mask].astype(int).astype(str)
        return result
    except (TypeError, ValueError):
        return col.astype(str)


def compute_composite(df, comp_name, paradigm):
    """
    Compute z-scored composite score for each row in df.
    Returns a float64 Series aligned to df.index.
    """
    defs = COMPOSITE_DEFS[paradigm][comp_name]
    parts = []
    for hp_col, kind, param in defs:
        if hp_col not in df.columns:
            continue
        col = df[hp_col]
        if kind == "cont":
            x = col.astype(float)
            z = (x - x.mean()) / (x.std() + 1e-12)
            parts.append(param * z)
        else:
            mapped = _col_to_str(col).map(param).astype(float)
            parts.append(mapped)

    if not parts:
        return pd.Series(np.nan, index=df.index)

    score = sum(parts)
    return (score - score.mean()) / (score.std() + 1e-12)


def spearman_r(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan, np.nan
    r, p = spearmanr(x[mask], y[mask])
    return float(r), float(p)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_analysis(task_dfs):
    """
    Compute composite scores and Spearman r with RDM properties for each task.
    Returns (enriched_task_dfs, results_df).
    enriched_task_dfs: same as task_dfs but with composite columns added.
    results_df: long-form table of (task, composite, rdm_prop, spearman_r, p_value, N).
    """
    rows = []
    enriched = {}

    for task, df in task_dfs.items():
        paradigm = get_paradigm(task)
        df = df.copy()

        for comp in COMPOSITES:
            df[comp] = compute_composite(df, comp, paradigm)

        enriched[task] = df

        for comp in COMPOSITES:
            for prop in RDM_PROPS:
                if comp not in df.columns or prop not in df.columns:
                    continue
                r, p = spearman_r(df[comp].values, df[prop].values)
                n = int((np.isfinite(df[comp]) & np.isfinite(df[prop])).sum())
                rows.append({
                    "task":       task,
                    "composite":  comp,
                    "paradigm":   paradigm,
                    "rdm_prop":   prop,
                    "spearman_r": r,
                    "p_value":    p,
                    "N":          n,
                })

    return enriched, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_heatmap_figure(results_df):
    """
    Summary heatmap: one panel per task (paradigm rows, task cols).
    Rows = composites, cols = RDM properties. Cell value = Spearman r.
    """
    tasks_by_paradigm = []
    for paradigm_name, task_list in PARADIGMS:
        present = [t for t in task_list if t in results_df["task"].values]
        if present:
            tasks_by_paradigm.append((paradigm_name, present))

    n_rows = len(tasks_by_paradigm)
    n_cols = max(len(tl) for _, tl in tasks_by_paradigm)
    fig    = plt.figure(figsize=(max(10, 2.8 * n_cols + 1.5), 3.8 * n_rows))

    vmax   = max(0.3, round(results_df["spearman_r"].abs().max(), 1))
    im_ref = None
    row_idx = 0

    for paradigm_name, task_list in tasks_by_paradigm:
        for col_idx, task in enumerate(task_list):
            sub = results_df[results_df["task"] == task]
            mat = np.full((len(COMPOSITES), len(RDM_PROPS)), np.nan)
            for r, comp in enumerate(COMPOSITES):
                for c, prop in enumerate(RDM_PROPS):
                    val = sub[(sub["composite"] == comp) & (sub["rdm_prop"] == prop)]["spearman_r"]
                    if len(val):
                        mat[r, c] = val.iloc[0]

            ax = fig.add_subplot(n_rows, n_cols, row_idx * n_cols + col_idx + 1)
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            im_ref = im

            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    v = mat[r, c]
                    if np.isfinite(v):
                        ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                                fontsize=6.5, color="white" if abs(v) > 0.4 else "black")

            ax.set_xticks(range(len(RDM_PROPS)))
            ax.set_xticklabels([RDM_LABELS[p] for p in RDM_PROPS],
                               fontsize=7, rotation=30, ha="right")
            ax.set_yticks(range(len(COMPOSITES)))
            ax.set_yticklabels([COMPOSITE_LABELS[c] for c in COMPOSITES], fontsize=7)
            ax.set_title(TASK_SHORT.get(task, task), fontsize=8, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(paradigm_name, fontsize=8, fontweight="bold")

        row_idx += 1

    if im_ref is not None:
        fig.colorbar(im_ref, ax=fig.get_axes(), shrink=0.6, label="Spearman r",
                     location="right", pad=0.02)

    fig.suptitle("Latent variable composites × RDM properties (Spearman r, successful networks)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.92, 0.97])
    return fig


def make_scatter_pages(enriched_dfs, out_path):
    """
    Multi-page PDF: one page per task.
    Each page: 3 composites (rows) × 4 RDM properties (cols).
    """
    TASK_COLOR = "#2271b2"

    with PdfPages(out_path) as pdf:
        for task, df in enriched_dfs.items():
            fig, axes = plt.subplots(len(COMPOSITES), len(RDM_PROPS),
                                     figsize=(13, 8), constrained_layout=True)
            fig.suptitle(f"{TASK_SHORT.get(task, task)} — latent composite scores vs. RDM properties",
                         fontsize=10, fontweight="bold")

            for r, comp in enumerate(COMPOSITES):
                for c, prop in enumerate(RDM_PROPS):
                    ax = axes[r, c]
                    if comp not in df.columns or prop not in df.columns:
                        ax.set_visible(False)
                        continue

                    x = df[comp].values.astype(float)
                    y = df[prop].values.astype(float)
                    mask = np.isfinite(x) & np.isfinite(y)
                    n = mask.sum()

                    ax.scatter(x[mask], y[mask], s=6, alpha=0.35, color=TASK_COLOR)

                    if n >= 10:
                        m, b = np.polyfit(x[mask], y[mask], 1)
                        xl = np.array([x[mask].min(), x[mask].max()])
                        ax.plot(xl, m * xl + b, color="#e05c00", lw=1.5)

                        rval, pval = spearman_r(x, y)
                        p_str = f"p={pval:.3f}" if pval >= 0.001 else "p<.001"
                        ax.set_title(f"r={rval:+.2f}, {p_str}, N={n}", fontsize=7)

                    ax.set_xlabel(COMPOSITE_LABELS[comp], fontsize=7)
                    ax.set_ylabel(RDM_LABELS[prop], fontsize=7)
                    ax.tick_params(labelsize=6)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Latent variable analysis.")
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

    task_dfs = {}
    for task in TASK_NAMES:
        sub = all_df[all_df["task"] == task].copy()
        if len(sub):
            task_dfs[task] = sub
            print(f"  {task}: {len(sub)} networks")

    print("\nComputing composites and Spearman correlations ...")
    enriched_dfs, results_df = run_analysis(task_dfs)

    csv_path = out_tables / "rdm_latent_vars.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    print("\nTop correlations (|r| > 0.3):")
    strong = results_df[results_df["spearman_r"].abs() > 0.3].sort_values(
        "spearman_r", key=abs, ascending=False)
    for _, row in strong.iterrows():
        print(f"  {row['task']:15s} {row['composite']:15s} → {row['rdm_prop']:20s}"
              f"  r={row['spearman_r']:+.3f}  p={row['p_value']:.3f}  N={row['N']}")

    fig = make_heatmap_figure(results_df)
    heatmap_path = out_figures / "f2_latent_vars.pdf"
    fig.savefig(heatmap_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {heatmap_path.name}")

    scatter_path = out_figures / "f2_latent_vars_scatter.pdf"
    make_scatter_pages(enriched_dfs, scatter_path)
    print(f"Saved: {scatter_path.name}")


if __name__ == "__main__":
    main()
