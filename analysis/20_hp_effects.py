#!/usr/bin/env python3
"""
Step 20: Direct HP effects on RDM properties — Finding #2.1.

For each (task, HP, RDM property) triple, computes:
  - Continuous HPs: signed Spearman r
  - Binary categorical HPs: signed sqrt(eta^2), sign = mean(level_B) > mean(level_A)

RDM properties per network (successful primary networks only):
  - reliability      : LOO Spearman r with group-mean RDM (from rdm_noise_ceiling.csv)
  - category_corr    : Spearman r with primary category model (task-specific)
  - dimensionality   : participation ratio of last hidden layer (from rdm_dimensionality.csv)
  - mean_dissimilarity: mean of the RDM upper triangle

Outputs:
    output/analysis/figures/f2_hp_effects.pdf
    output/analysis/tables/rdm_hp_effects.csv
"""

import argparse
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
from analysis_utils import FIGURES_DIR, RDM_DIR, TABLES_DIR, TASK_NAMES, RL_TASKS, metric_output_dirs

TASK_DIR_OVERRIDES = {}
RNN_TASKS          = {"adding", "mnist_rnn"}

# Primary category model per task (used for category_corr)
PRIMARY_MODEL = {
    "mnist_dual":    "output",
    "mnist_10way":   "digit",
    "fashion_10way": "class",
    "spirals":       "spatial",
    "parity":        "hamming_diff",
    "adding":        "sum",
    "mnist_rnn":     "digit",
    "cartpole":      "euclidean",
    "fourrooms":     "euclidean",
}

# Continuous HPs per paradigm
CONT_HPS = ["hp_learning_rate", "hp_l1_reg", "hp_l2_reg", "hp_hidden_size", "hp_batch_size"]
CONT_LABELS = {
    "hp_learning_rate": "lr",
    "hp_l1_reg":        "l1",
    "hp_l2_reg":        "l2",
    "hp_hidden_size":   "hidden_size",
    "hp_batch_size":    "batch_size",
}

# Categorical HPs: (attr_name, level_A_label, level_B_label)
# Effect sign = positive if level_B has higher mean RDM property.
# Levels must match the string representation stored in the DataFrame
# (integer HPs stored as int → "1"/"2"; strings stored as-is).
CAT_HPS_SUPERVISED = [
    ("hp_optimizer",   "sgd",    "adam"),
    ("hp_activation",  "relu",   "sigmoid"),
    ("hp_activation",  "sigmoid","tanh"),
    ("hp_activation",  "tanh",   "relu"),
    ("hp_depth",       "1",      "2"),       # stored as int
    ("hp_init_scale",  "0.1",    "1.0"),
]
CAT_HPS_RNN = [
    ("hp_optimizer",    "sgd",  "adam"),
    ("hp_cell_type",    "gru",  "rnn"),    # Elman RNN (not lstm)
    ("hp_n_rnn_layers", "1",    "2"),      # stored as int
    ("hp_init_scale",   "0.1",  "1.0"),
]
CAT_LABELS = {
    "hp_optimizer":                  "optimizer\n(sgd→adam)",
    "hp_activation:relu/sigmoid":    "activation\n(relu→sig)",
    "hp_activation:sigmoid/tanh":    "activation\n(sig→tanh)",
    "hp_activation:tanh/relu":       "activation\n(tanh→relu)",
    "hp_depth":                      "depth\n(1→2)",
    "hp_init_scale":                 "init_scale\n(0.1→1.0)",
    "hp_cell_type":                  "cell_type\n(gru→rnn)",
    "hp_n_rnn_layers":               "n_rnn_layers\n(1→2)",
}

RDM_PROPS    = ["reliability", "category_corr", "dimensionality", "mean_dissimilarity"]
RDM_LABELS   = {
    "reliability":       "reliability",
    "category_corr":     "category\ncorr",
    "dimensionality":    "dimension-\nality",
    "mean_dissimilarity":"mean\ndissim.",
}

# Paradigm grouping for figure layout
PARADIGMS = [
    ("Supervised",  ["mnist_dual", "mnist_10way", "fashion_10way", "spirals", "parity"]),
    ("RNN",         ["adding", "mnist_rnn"]),
    ("RL",          ["cartpole", "fourrooms"]),
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
# Data assembly
# ---------------------------------------------------------------------------

def load_thresholds():
    data = json.load(open(TABLES_DIR / "success_thresholds.json"))
    return {k: (float(v["upper"]) if isinstance(v, dict) else None)
            for k, v in data.items() if k != "_alpha"}


def _hp_val(v):
    """
    Normalise an HP attribute value for storage in the DataFrame.
    - Strings stored as plain str (strip any surrounding quotes).
    - Integer-valued numerics stored as int (so depth=1 → "1" via astype(str)).
    - Floating-point numerics stored as float.
    """
    if isinstance(v, (str, bytes)):
        s = v.decode() if isinstance(v, bytes) else v
        return s.strip("'\"")
    f = float(v)
    if f == int(f):
        return int(f)
    return f


def _adding_sum_model():
    """Return the 4950-length upper-triangle sum-model RDM for adding."""
    sys.path.insert(0, str(ANALYSIS.parent))
    from tasks import TASKS
    inputs, meta = TASKS["adding"]().get_rdm_stimuli()
    targets = meta["targets"].astype(np.float32)
    ri, ci  = np.triu_indices(len(targets), k=1)
    return np.abs(targets[ri] - targets[ci])


def load_per_network_stats(thresholds, metric="cosine"):
    """
    For each successful primary network, load:
      HPs, reliability, category_corr, dimensionality, mean_dissimilarity.
    Returns: dict task -> DataFrame.
    """
    _, metric_tables = metric_output_dirs(metric)
    nc_df  = pd.read_csv(metric_tables / "rdm_noise_ceiling.csv")
    cat_df = pd.read_csv(metric_tables / "rdm_category_structure.csv")
    dim_df = pd.read_csv(TABLES_DIR / "rdm_dimensionality.csv")

    # Build lookup dicts: (task, run_id) -> value
    nc_lookup = {(r["task"], r["run_id"]): r["loo_spearman_r"]
                 for _, r in nc_df.iterrows()}
    cat_lookup = {}
    for _, r in cat_df.iterrows():
        if r["perf_cat"] != "successful":
            continue
        cat_lookup[(r["task"], r["run_id"], r["model_name"])] = r["spearman_r"]
    dim_lookup = {(r["task"], r["run_id"]): r["pr_last"]
                  for _, r in dim_df.iterrows()}

    # Pre-compute adding sum-model vector (computed inline; not in cat CSV)
    adding_sum_vec = _adding_sum_model()

    task_dfs = {}
    ckpt_map = {t: ("final" if t in RL_TASKS else "best") for t in TASK_NAMES}

    for task in TASK_NAMES:
        h5_path = RDM_DIR / f"{task}_rdms.h5"
        if not h5_path.exists():
            continue
        th            = thresholds.get(task)
        primary_model = PRIMARY_MODEL.get(task)
        is_rnn        = task in RNN_TASKS
        ckpt          = ckpt_map[task]

        rows = []
        with h5py.File(h5_path, "r") as h5:
            runs_grp = h5.get("runs", {})
            for run_id in sorted(runs_grp.keys()):
                rg = runs_grp[run_id]
                if bool(rg.attrs.get("is_repeat", False)):
                    continue
                perf = float(rg.attrs.get("performance", float("nan")))
                if th is not None and perf < th:
                    continue

                depth = int(rg.attrs.get("hp_depth", 1))
                cg    = rg.get(ckpt)
                if cg is None:
                    continue

                # Last hidden layer key
                if is_rnn:
                    parsed = []
                    for k in cg.keys():
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
                    lkey  = f"layer_{max_l}_t_{max_t}_{metric}"
                else:
                    lkey = f"layer_{max(0, depth - 1)}_{metric}"

                ds = cg.get(lkey)
                if ds is None or ds.attrs.get("degenerate", False) or len(ds) == 0:
                    continue
                vec = ds[:].astype(np.float32)

                row = {
                    "task":        task,
                    "run_id":      run_id,
                    "performance": perf,
                }
                # HPs — normalised for consistent string comparison
                for k, v in rg.attrs.items():
                    if k.startswith("hp_"):
                        row[k] = _hp_val(v)

                # RDM properties
                row["reliability"]        = nc_lookup.get((task, run_id), np.nan)
                row["dimensionality"]     = dim_lookup.get((task, run_id), np.nan)
                row["mean_dissimilarity"] = float(vec.mean()) if len(vec) > 0 else np.nan

                if task == "adding":
                    # Category corr not in CSV; compute inline vs sum model
                    row["category_corr"] = float(spearmanr(vec, adding_sum_vec)[0])
                else:
                    row["category_corr"] = cat_lookup.get((task, run_id, primary_model), np.nan)

                rows.append(row)

        if rows:
            task_dfs[task] = pd.DataFrame(rows)
            print(f"  {task}: {len(rows)} successful primary networks")

    return task_dfs


# ---------------------------------------------------------------------------
# Effect size computation
# ---------------------------------------------------------------------------

def _hp_key(hp_attr, lev_a, lev_b, cat_hps):
    """Unique key for a categorical HP comparison. Uses composite form when the
    same attribute appears more than once in the list (e.g. three activation pairs)."""
    if sum(1 for h, _, _ in cat_hps if h == hp_attr) > 1:
        return f"{hp_attr}:{lev_a}/{lev_b}"
    return hp_attr

def spearman_r(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan
    return spearmanr(x[mask], y[mask])[0]


def signed_eta(df, hp_attr, level_a, level_b, rdm_prop):
    """
    Compute signed sqrt(eta^2) for a binary categorical HP.
    Sign: positive if mean(level_B) > mean(level_A).
    """
    vals_a = df[df[hp_attr].astype(str) == level_a][rdm_prop].dropna().values
    vals_b = df[df[hp_attr].astype(str) == level_b][rdm_prop].dropna().values
    if len(vals_a) < 5 or len(vals_b) < 5:
        return np.nan
    all_vals = np.concatenate([vals_a, vals_b])
    grand_mean = all_vals.mean()
    ss_total   = ((all_vals - grand_mean) ** 2).sum()
    if ss_total < 1e-12:
        return np.nan
    mean_a, mean_b = vals_a.mean(), vals_b.mean()
    ss_between = len(vals_a) * (mean_a - grand_mean)**2 + len(vals_b) * (mean_b - grand_mean)**2
    eta2 = ss_between / ss_total
    sign = 1.0 if mean_b > mean_a else -1.0
    return sign * np.sqrt(eta2)


def compute_effects(df, task):
    """
    Compute all HP × RDM property effect sizes for one task.
    Returns list of dicts.
    """
    is_rnn = task in RNN_TASKS
    is_rl  = task in RL_TASKS
    rows   = []

    for prop in RDM_PROPS:
        if prop not in df.columns:
            continue
        y = df[prop].values.astype(float)

        # Continuous HPs
        for hp in CONT_HPS:
            if hp == "hp_batch_size" and is_rl:
                continue   # RL has no batch_size
            if hp not in df.columns:
                continue
            x = df[hp].values.astype(float)
            r = spearman_r(x, y)
            rows.append({"task": task, "hp": hp, "hp_type": "continuous",
                         "rdm_prop": prop, "effect": r})

        # Categorical HPs
        cat_hps = CAT_HPS_RNN if is_rnn else CAT_HPS_SUPERVISED
        for hp_attr, lev_a, lev_b in cat_hps:
            if hp_attr not in df.columns:
                continue
            hp_key = _hp_key(hp_attr, lev_a, lev_b, cat_hps)
            e = signed_eta(df, hp_attr, lev_a, lev_b, prop)
            rows.append({"task": task, "hp": hp_key, "hp_type": "categorical",
                         "level_a": lev_a, "level_b": lev_b,
                         "rdm_prop": prop, "effect": e})

    return rows


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def hp_row_order(task):
    is_rnn = task in RNN_TASKS
    cat_hps = CAT_HPS_RNN if is_rnn else CAT_HPS_SUPERVISED
    cont = [h for h in CONT_HPS
            if not (h == "hp_batch_size" and task in RL_TASKS)]
    cat  = [_hp_key(h, la, lb, cat_hps) for h, la, lb in cat_hps]
    return cont + cat


def hp_label(hp_attr, is_cat, task):
    if is_cat:
        return CAT_LABELS.get(hp_attr, hp_attr)
    return CONT_LABELS.get(hp_attr, hp_attr)


def make_heatmap(ax, mat, row_labels, col_labels, title):
    """Draw a signed diverging heatmap on ax."""
    vmax = max(0.3, np.nanmax(np.abs(mat)))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(title, fontsize=8, fontweight="bold")
    # Annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.5, color="white" if abs(v) > 0.4 else "black")
    return im


def make_figure(effects_df, task_dfs):
    """
    One row per paradigm. Each row has one subplot per task, sharing the HP y-axis.
    Paradigms are separated by vertical whitespace.
    """
    if not task_dfs:
        return None

    tasks_by_paradigm = []
    for paradigm_name, task_list in PARADIGMS:
        present = [t for t in task_list if t in task_dfs]
        if present:
            tasks_by_paradigm.append((paradigm_name, present))

    n_rows  = len(tasks_by_paradigm)
    n_cols  = max(len(tl) for _, tl in tasks_by_paradigm)
    fig_w   = max(12, 3.0 * n_cols + 1.5)
    fig_h   = 5.5 * n_rows

    fig = plt.figure(figsize=(fig_w, fig_h))
    vmax_global = 0.0
    all_axes    = []
    all_mats    = []
    im_ref      = None

    # First pass: compute all matrices and global vmax
    for paradigm_name, task_list in tasks_by_paradigm:
        for task in task_list:
            sub      = effects_df[effects_df["task"] == task]
            hp_order = [h for h in hp_row_order(task) if len(sub[sub["hp"]==h]) > 0]
            mat = np.full((len(hp_order), len(RDM_PROPS)), np.nan)
            for r, hp in enumerate(hp_order):
                for c, prop in enumerate(RDM_PROPS):
                    val = sub[(sub["hp"]==hp) & (sub["rdm_prop"]==prop)]["effect"]
                    if len(val):
                        mat[r, c] = val.iloc[0]
            all_mats.append((paradigm_name, task, hp_order, mat))
            vmax_global = max(vmax_global, np.nanmax(np.abs(mat)))

    vmax_global = max(0.3, round(vmax_global, 1))

    # Second pass: draw
    row_idx = 0
    for paradigm_name, task_list in tasks_by_paradigm:
        n_task = len(task_list)
        for col, task in enumerate(task_list):
            _, _, hp_order, mat = next(
                (x for x in all_mats if x[0]==paradigm_name and x[1]==task), (None,None,None,None))
            if mat is None:
                continue

            ax = fig.add_subplot(n_rows, n_cols, row_idx * n_cols + col + 1)
            all_axes.append(ax)

            row_labels = []
            for h in hp_order:
                sub  = effects_df[(effects_df["task"]==task) & (effects_df["hp"]==h)]
                is_c = (sub["hp_type"].iloc[0] == "categorical") if len(sub) else False
                row_labels.append(hp_label(h, is_c, task))

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
                                color="white" if abs(v) > 0.5*vmax_global else "black")
            if im_ref is None:
                im_ref = im

        # Hide unused subplots in this row
        for col in range(n_task, n_cols):
            ax_empty = fig.add_subplot(n_rows, n_cols, row_idx * n_cols + col + 1)
            ax_empty.set_visible(False)

        row_idx += 1

    # Shared colorbar
    cbar = fig.colorbar(im_ref, ax=all_axes, orientation="vertical",
                        fraction=0.015, pad=0.03, shrink=0.8)
    cbar.set_label("signed effect size\n(Spearman r / signed √η²)", fontsize=7)

    metric_defs = (
        "Metric definitions  —  "
        "reliability: LOO Spearman r of each network's RDM vs. group-mean RDM  |  "
        "category corr: Spearman r vs. primary category model "
        "(digit / class / spatial / euclidean / hamming / sum per task)  |  "
        "dimensionality: participation ratio PR = (Σλ)²/Σλ²  |  "
        "mean dissim.: mean pairwise cosine distance across all stimulus pairs"
    )
    fig.text(0.01, 0.005, metric_defs, fontsize=5.5, color="#444444",
             wrap=True, ha="left")

    effect_note = (
        "Effect sizes: continuous HPs → Spearman r  |  "
        "categorical HPs → signed √η²  (positive = level_B > level_A)"
    )
    fig.text(0.01, 0.025, effect_note, fontsize=6, color="grey")

    fig.suptitle(
        "HP × RDM-property effect sizes — successful primary networks, last hidden layer",
        fontsize=10)
    fig.subplots_adjust(hspace=0.55, wspace=0.08, right=0.88, bottom=0.10)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HP effects on RDM properties.")
    parser.add_argument("--metric", choices=["cosine", "pearson"], default="cosine",
                        help="RDM metric to use (default: cosine).")
    args = parser.parse_args()

    out_figures, out_tables = metric_output_dirs(args.metric)
    out_figures.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds()

    print("Loading per-network stats ...")
    task_dfs = load_per_network_stats(thresholds, metric=args.metric)

    print("\nComputing HP effects ...")
    all_effects = []
    for task, df in task_dfs.items():
        effects = compute_effects(df, task)
        all_effects.extend(effects)
        print(f"  {task}: {len(effects)} (HP, prop) pairs")

    effects_df = pd.DataFrame(all_effects)

    # Save per-network stats (used by script 21)
    per_net = pd.concat(list(task_dfs.values()), ignore_index=True)
    per_net.to_csv(out_tables / "rdm_per_network_stats.csv", index=False)

    # Save HP effects
    csv_path = out_tables / "rdm_hp_effects.csv"
    effects_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Figure
    fig = make_figure(effects_df, task_dfs)
    if fig:
        out = out_figures / "f2_hp_effects.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")

    # Print top effects per task for a quick summary
    print("\n=== Top HP effects (|effect| > 0.15) ===")
    strong = effects_df[effects_df["effect"].abs() > 0.15].sort_values("effect", key=abs, ascending=False)
    for _, row in strong.iterrows():
        print(f"  {row['task']:15s} {row['hp']:22s} → {row['rdm_prop']:20s}  effect={row['effect']:+.3f}")


if __name__ == "__main__":
    main()
