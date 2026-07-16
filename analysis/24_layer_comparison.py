#!/usr/bin/env python3
"""
Step 24: Layer comparison for HP effects — Finding #2.5.

Three-way comparison across depth groups:
  (A) depth=1 networks → layer_0  (their only layer)
  (B) depth=2 networks → layer_0  (first hidden, H units)
  (C) depth=2 networks → layer_1  (second hidden, H//2 units)

RNN tasks are excluded (they use n_rnn_layers, not depth).
hp_depth is excluded as a predictor within each group (fixed per group).

RDM properties computed per layer from HDF5:
  reliability       : LOO Spearman r with group-mean RDM (within each group)
  category_corr     : Spearman r with primary category model (from cache)
  mean_dissimilarity: mean of upper triangle

Outputs:
    output/analysis/{metric}/figures/f2_layer_comparison.pdf
    output/analysis/{metric}/tables/rdm_layer_comparison.csv
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
from scipy.stats import spearmanr

ANALYSIS = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import (
    CACHE_DIR, RDM_DIR, TABLES_DIR, TASK_NAMES, RL_TASKS, metric_output_dirs
)

RNN_TASKS = {"adding", "mnist_rnn"}
# Depth=2 makes no sense for RNN tasks; exclude them entirely
LAYER_TASKS = [t for t in TASK_NAMES if t not in RNN_TASKS]

PRIMARY_MODEL = {
    "mnist_dual":    "output",
    "mnist_10way":   "digit",
    "fashion_10way": "class",
    "spirals":       "spatial",
    "parity":        "hamming_diff",
    "cartpole":      "euclidean",
    "fourrooms":     "euclidean",
}

# Continuous HPs (same as script 20)
CONT_HPS = ["hp_learning_rate", "hp_l1_reg", "hp_l2_reg", "hp_hidden_size", "hp_batch_size"]
CONT_LABELS = {
    "hp_learning_rate": "lr",
    "hp_l1_reg":        "l1",
    "hp_l2_reg":        "l2",
    "hp_hidden_size":   "hidden_size",
    "hp_batch_size":    "batch_size",
}

# Categorical HPs — depth excluded (all depth=2); same three activation contrasts as script 20
CAT_HPS_SUPERVISED = [
    ("hp_optimizer",  "sgd",     "adam"),
    ("hp_activation", "relu",    "sigmoid"),
    ("hp_activation", "sigmoid", "tanh"),
    ("hp_activation", "tanh",    "relu"),
    ("hp_init_scale", "0.1",     "1.0"),
]
CAT_HPS_RL = CAT_HPS_SUPERVISED  # RL has no batch_size but same categorical HPs

CAT_LABELS = {
    "hp_optimizer":               "optimizer\n(sgd→adam)",
    "hp_activation:relu/sigmoid": "activation\n(relu→sig)",
    "hp_activation:sigmoid/tanh": "activation\n(sig→tanh)",
    "hp_activation:tanh/relu":    "activation\n(tanh→relu)",
    "hp_init_scale":              "init_scale\n(0.1→1.0)",
}

RDM_PROPS = ["reliability", "category_corr", "mean_dissimilarity"]
RDM_LABELS = {
    "reliability":        "reliability",
    "category_corr":      "category\ncorr",
    "mean_dissimilarity": "mean\ndissim.",
}

PARADIGMS = [
    ("Supervised", ["mnist_dual", "mnist_10way", "fashion_10way", "spirals", "parity"]),
    ("RL",         ["cartpole", "fourrooms"]),
]

TASK_SHORT = {
    "mnist_dual":    "MNIST\ndual",
    "mnist_10way":   "MNIST\n10way",
    "fashion_10way": "Fashion\n10way",
    "spirals":       "Spirals",
    "parity":        "Parity",
    "cartpole":      "CartPole",
    "fourrooms":     "FourRooms",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hp_key(hp_attr, lev_a, lev_b, cat_hps):
    if sum(1 for h, _, _ in cat_hps if h == hp_attr) > 1:
        return f"{hp_attr}:{lev_a}/{lev_b}"
    return hp_attr


def _hp_val(v):
    if isinstance(v, (str, bytes)):
        s = v.decode() if isinstance(v, bytes) else v
        return s.strip("'\"")
    f = float(v)
    return int(f) if f == int(f) else f


def load_thresholds():
    data = __import__("json").load(open(TABLES_DIR / "success_thresholds.json"))
    return {k: (float(v["upper"]) if isinstance(v, dict) else None)
            for k, v in data.items() if k != "_alpha"}


def load_cat_model_vec(task):
    """Upper-triangle vector for the primary category model of this task."""
    npz_path = CACHE_DIR / "category_models" / f"{task}.npz"
    model_name = PRIMARY_MODEL.get(task)
    if not npz_path.exists() or model_name is None:
        return None
    models = dict(np.load(npz_path))
    if model_name not in models:
        return None
    mat = models[model_name].astype(np.float32)
    ri, ci = np.triu_indices(mat.shape[0], k=1)
    return mat[ri, ci]


def loo_spearman(mat):
    """LOO Spearman r for each row vs mean of remaining rows. mat: (N, D)."""
    N = mat.shape[0]
    if N < 3:
        return np.full(N, np.nan)
    mat_f64 = mat.astype(np.float64)
    group_sum = mat_f64.sum(axis=0)
    results = np.zeros(N)
    for i in range(N):
        vec_i = mat_f64[i]
        loo = (group_sum - vec_i) / (N - 1)
        r, _ = spearmanr(vec_i, loo)
        results[i] = r
    return results


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_depth_layers(task, threshold, metric="cosine"):
    """
    Single HDF5 pass: collect layer RDMs for all successful primaries.
    Returns three lists of row-dicts (each with run_id, hp_*, vec):
      d1_rows   — depth=1, layer_0
      d2l0_rows — depth=2, layer_0
      d2l1_rows — depth=2, layer_1
    """
    h5_path = RDM_DIR / f"{task}_rdms.h5"
    if not h5_path.exists():
        return [], [], []

    ckpt = "final" if task in RL_TASKS else "best"
    d1, d2l0, d2l1 = [], [], []

    with h5py.File(h5_path, "r") as h5:
        runs_grp = h5.get("runs", {})
        for run_id in sorted(runs_grp.keys()):
            rg = runs_grp[run_id]
            if bool(rg.attrs.get("is_repeat", False)):
                continue
            perf = float(rg.attrs.get("performance", float("nan")))
            if threshold is not None and perf < threshold:
                continue
            depth = int(rg.attrs.get("hp_depth", 1))
            if depth not in (1, 2):
                continue

            cg = rg.get(ckpt)
            if cg is None:
                continue

            def _load_vec(key):
                if key not in cg:
                    return None
                ds = cg[key]
                if ds.attrs.get("degenerate", False) or len(ds) == 0:
                    return None
                v = ds[:].astype(np.float32)
                if not np.all(np.isfinite(v)):
                    raise ValueError(f"{task}/{run_id}/{key}: NaN in RDM")
                return v

            k0 = f"layer_0_{metric}"
            k1 = f"layer_1_{metric}"
            hp = {"run_id": run_id, "performance": perf}
            for k, v in rg.attrs.items():
                if k.startswith("hp_"):
                    hp[k] = _hp_val(v)

            if depth == 1:
                v0 = _load_vec(k0)
                if v0 is not None:
                    d1.append({**hp, "vec": v0})
            else:
                v0 = _load_vec(k0)
                v1 = _load_vec(k1)
                if v0 is not None:
                    d2l0.append({**hp, "vec": v0})
                if v1 is not None:
                    d2l1.append({**hp, "vec": v1})

    return d1, d2l0, d2l1


def build_df(rows, task, depth_label, layer_idx):
    """
    Compute per-network RDM properties for one group of rows.
    Each row must have 'vec' key with the RDM upper-triangle vector.
    """
    if not rows:
        return None

    cat_vec = load_cat_model_vec(task)
    mat = np.vstack([r["vec"] for r in rows])
    rel = loo_spearman(mat)

    records = []
    for i, row in enumerate(rows):
        rec = {k: v for k, v in row.items() if k != "vec"}
        rec["task"]        = task
        rec["depth_label"] = depth_label
        rec["layer"]       = layer_idx
        rec["reliability"] = float(rel[i])
        rec["mean_dissimilarity"] = float(mat[i].mean())
        if cat_vec is not None:
            r, _ = spearmanr(mat[i], cat_vec)
            rec["category_corr"] = float(r)
        else:
            rec["category_corr"] = np.nan
        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Effect sizes (same logic as script 20)
# ---------------------------------------------------------------------------

def spearman_r(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan
    return spearmanr(x[mask], y[mask])[0]


def signed_eta(df, hp_attr, level_a, level_b, rdm_prop):
    vals_a = df[df[hp_attr].astype(str) == level_a][rdm_prop].dropna().values
    vals_b = df[df[hp_attr].astype(str) == level_b][rdm_prop].dropna().values
    if len(vals_a) < 5 or len(vals_b) < 5:
        return np.nan
    all_v = np.concatenate([vals_a, vals_b])
    grand  = all_v.mean()
    ss_tot = ((all_v - grand) ** 2).sum()
    if ss_tot < 1e-12:
        return np.nan
    ss_bet = (len(vals_a) * (vals_a.mean() - grand) ** 2 +
              len(vals_b) * (vals_b.mean() - grand) ** 2)
    sign = 1.0 if vals_b.mean() > vals_a.mean() else -1.0
    return sign * np.sqrt(ss_bet / ss_tot)


def compute_effects(df, task):
    is_rl  = task in RL_TASKS
    cat_hps = CAT_HPS_RL if is_rl else CAT_HPS_SUPERVISED
    rows = []

    for prop in RDM_PROPS:
        if prop not in df.columns:
            continue
        y = df[prop].values.astype(float)

        for hp in CONT_HPS:
            if hp == "hp_batch_size" and is_rl:
                continue
            if hp not in df.columns:
                continue
            r = spearman_r(df[hp].values.astype(float), y)
            rows.append({"hp": hp, "hp_type": "continuous", "rdm_prop": prop, "effect": r})

        for hp_attr, lev_a, lev_b in cat_hps:
            if hp_attr not in df.columns:
                continue
            hp_key = _hp_key(hp_attr, lev_a, lev_b, cat_hps)
            e = signed_eta(df, hp_attr, lev_a, lev_b, prop)
            rows.append({"hp": hp_key, "hp_type": "categorical",
                         "rdm_prop": prop, "effect": e})

    return rows


def hp_row_order(task):
    is_rl  = task in RL_TASKS
    cat_hps = CAT_HPS_RL if is_rl else CAT_HPS_SUPERVISED
    cont = [h for h in CONT_HPS
            if not (h == "hp_batch_size" and is_rl)]
    cat  = [_hp_key(h, la, lb, cat_hps) for h, la, lb in cat_hps]
    return cont + cat


def hp_label(hp_key, is_cat):
    if is_cat:
        return CAT_LABELS.get(hp_key, hp_key)
    return CONT_LABELS.get(hp_key, hp_key)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_layer_figure(effects_by_task, layer_label, vmax_global):
    """
    One heatmap figure for a single layer, layout identical to script 20.
    """
    tasks_by_paradigm = []
    for paradigm_name, task_list in PARADIGMS:
        present = [t for t in task_list if t in effects_by_task]
        if present:
            tasks_by_paradigm.append((paradigm_name, present))

    n_rows = len(tasks_by_paradigm)
    n_cols = max(len(tl) for _, tl in tasks_by_paradigm)
    fig_w  = max(10, 3.0 * n_cols + 1.5)
    fig_h  = 5.5 * n_rows

    fig = plt.figure(figsize=(fig_w, fig_h))
    all_axes = []
    im_ref   = None
    row_idx  = 0

    for paradigm_name, task_list in tasks_by_paradigm:
        n_task = len(task_list)
        for col, task in enumerate(task_list):
            eff_df = effects_by_task[task]
            hp_order = [h for h in hp_row_order(task) if h in eff_df["hp"].values]
            mat = np.full((len(hp_order), len(RDM_PROPS)), np.nan)
            for r, hp in enumerate(hp_order):
                for c, prop in enumerate(RDM_PROPS):
                    val = eff_df[(eff_df["hp"] == hp) & (eff_df["rdm_prop"] == prop)]["effect"]
                    if len(val):
                        mat[r, c] = val.iloc[0]

            ax = fig.add_subplot(n_rows, n_cols, row_idx * n_cols + col + 1)
            all_axes.append(ax)

            row_labels = []
            for h in hp_order:
                sub  = eff_df[eff_df["hp"] == h]
                is_c = sub["hp_type"].iloc[0] == "categorical" if len(sub) else False
                row_labels.append(hp_label(h, is_c))

            col_labels = [RDM_LABELS[p] for p in RDM_PROPS]

            im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax_global, vmax=vmax_global,
                           aspect="auto")
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, fontsize=7, rotation=30, ha="right")
            ax.set_yticks(range(len(row_labels)))
            if col == 0:
                ax.set_yticklabels(row_labels, fontsize=7)
                ax.set_ylabel(paradigm_name, fontsize=8, fontweight="bold")
            else:
                ax.set_yticklabels([])
            ax.set_title(TASK_SHORT.get(task, task), fontsize=8, fontweight="bold")

            for ri in range(mat.shape[0]):
                for ci in range(mat.shape[1]):
                    v = mat[ri, ci]
                    if np.isfinite(v):
                        ax.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                                fontsize=5.5,
                                color="white" if abs(v) > 0.5 * vmax_global else "black")
            if im_ref is None:
                im_ref = im

        for col in range(n_task, n_cols):
            ax_e = fig.add_subplot(n_rows, n_cols, row_idx * n_cols + col + 1)
            ax_e.set_visible(False)

        row_idx += 1

    if im_ref is not None:
        cbar = fig.colorbar(im_ref, ax=all_axes, orientation="vertical",
                            fraction=0.015, pad=0.03, shrink=0.8)
        cbar.set_label("signed effect size\n(Spearman r / signed √η²)", fontsize=7)

    fig.suptitle(
        f"HP × RDM-property effects — depth=2 networks, {layer_label}\n"
        "(depth excluded as predictor; dimensionality excluded — requires activations)",
        fontsize=9)
    fig.subplots_adjust(hspace=0.55, wspace=0.08, right=0.88, bottom=0.10)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Layer comparison for HP effects (depth=2 networks).")
    parser.add_argument("--metric", choices=["cosine", "pearson"], default="cosine")
    args = parser.parse_args()

    out_figures, out_tables = metric_output_dirs(args.metric)
    out_figures.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds()

    print("Loading networks (depth=1 and depth=2) ...")
    # Three groups: (effects_dict, page_label, depth_label, layer_idx)
    groups = [
        ({}, "depth=1, layer_0  (only layer)",    "d1",  0),
        ({}, "depth=2, layer_0  (first hidden)",  "d2l0", 0),
        ({}, "depth=2, layer_1  (second hidden)", "d2l1", 1),
    ]
    all_csv_rows = []

    for task in LAYER_TASKS:
        d1_rows, d2l0_rows, d2l1_rows = load_all_depth_layers(
            task, thresholds.get(task), metric=args.metric)

        parts = [
            (d1_rows,   "d1",   0, groups[0][0]),
            (d2l0_rows, "d2l0", 0, groups[1][0]),
            (d2l1_rows, "d2l1", 1, groups[2][0]),
        ]
        n_parts = [len(r) for r, *_ in parts]
        if not any(n_parts):
            print(f"  [skip] {task}: no networks loaded")
            continue

        rel_means = []
        for rows, depth_label, layer_idx, effects_dict in parts:
            df = build_df(rows, task, depth_label, layer_idx)
            if df is None or len(df) < 10:
                rel_means.append(f"{depth_label}=—")
                continue
            rel_means.append(f"{depth_label}={df['reliability'].mean():.3f}")
            eff = pd.DataFrame(compute_effects(df, task))
            if len(eff):
                effects_dict[task] = eff
            for _, row in eff.iterrows():
                all_csv_rows.append({**row, "task": task,
                                     "depth_label": depth_label, "layer": layer_idx})

        print(f"  {task}: N={n_parts}  LOO r: {',  '.join(rel_means)}")

    # Shared colour scale across all three groups
    all_eff_dfs = [df for ed, *_ in groups for df in ed.values()]
    if not all_eff_dfs:
        print("No effects computed — nothing to plot.")
        return
    vmax_global = max(0.3, round(
        max(np.nanmax(np.abs(df["effect"].values)) for df in all_eff_dfs), 1
    ))
    print(f"\nGlobal |effect| max: {vmax_global:.2f}")

    pdf_path = out_figures / "f2_layer_comparison.pdf"
    with PdfPages(pdf_path) as pdf:
        for effects_dict, label, *_ in groups:
            if not effects_dict:
                continue
            fig = make_layer_figure(effects_dict, label, vmax_global)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved: {pdf_path}")

    csv_path = out_tables / "rdm_layer_comparison.csv"
    pd.DataFrame(all_csv_rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
