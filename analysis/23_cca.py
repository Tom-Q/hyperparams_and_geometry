#!/usr/bin/env python3
"""
Step 23: Canonical Correlation Analysis — Finding #2.4.

Two CCA analyses per task (successful primary networks only):

  CCA-composites : X = [stability, capacity, regularization]
  CCA-hps        : X = all individual HPs (continuous z-scored,
                       binary categoricals ±1 effect-coded,
                       activation 3-level via two Helmert contrasts)

  Both use: Y = [reliability, category_corr, dimensionality, mean_dissimilarity]

Implementation: direct SVD on the cross-correlation matrix.
  Cxx^(-1/2) @ Cxy @ Cyy^(-1/2) = U S Vt
  Canonical r = S[0]; X-weights = Cxx^(-1/2) @ U[:,0]
Ridge regularisation (α=0.01) added to Cxx and Cyy for stability.

Significance: permutation test (n=1000, shuffling Y rows).

Outputs:
    output/analysis/{metric}/figures/f2_cca.pdf
    output/analysis/{metric}/tables/rdm_cca.csv
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

ANALYSIS = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import TASK_NAMES, RL_TASKS, metric_output_dirs

RNN_TASKS = {"adding", "mnist_rnn"}

N_PERM = 1000
RIDGE  = 0.01
RNG    = np.random.default_rng(42)

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

PARADIGMS = [
    ("Supervised", ["mnist_dual", "mnist_10way", "fashion_10way", "spirals", "parity"]),
    ("RNN",        ["adding", "mnist_rnn"]),
    ("RL",         ["cartpole", "fourrooms"]),
]

RDM_PROPS = ["reliability", "category_corr", "dimensionality", "mean_dissimilarity"]
RDM_LABELS = {
    "reliability":        "reliability",
    "category_corr":      "cat. corr",
    "dimensionality":     "dimensionality",
    "mean_dissimilarity": "mean dissim.",
}

COMPOSITES    = ["stability", "capacity", "regularization"]
COMP_LABELS   = {"stability": "Stability", "capacity": "Capacity",
                 "regularization": "Regularization"}

# ── composite definitions (same as script 21) ────────────────────────────────

COMPOSITE_DEFS = {
    "supervised": {
        "stability":      [("hp_learning_rate","cont",-1),("hp_batch_size","cont",+1),
                           ("hp_optimizer","cat",{"adam":+1,"sgd":-1}),
                           ("hp_init_scale","cat",{"0.1":+1,"1":-1}),
                           ("hp_l2_reg","cont",+1)],
        "capacity":       [("hp_hidden_size","cont",+1),
                           ("hp_depth","cat",{"1":-1,"2":+1})],
        "regularization": [("hp_l1_reg","cont",+1),("hp_l2_reg","cont",+1)],
    },
    "rnn": {
        "stability":      [("hp_learning_rate","cont",-1),("hp_batch_size","cont",+1),
                           ("hp_optimizer","cat",{"adam":+1,"sgd":-1}),
                           ("hp_init_scale","cat",{"0.1":+1,"1":-1}),
                           ("hp_l2_reg","cont",+1)],
        "capacity":       [("hp_hidden_size","cont",+1),
                           ("hp_n_rnn_layers","cat",{"1":-1,"2":+1})],
        "regularization": [("hp_l1_reg","cont",+1),("hp_l2_reg","cont",+1)],
    },
    "rl": {
        "stability":      [("hp_learning_rate","cont",-1),
                           ("hp_optimizer","cat",{"adam":+1,"sgd":-1}),
                           ("hp_init_scale","cat",{"0.1":+1,"1":-1}),
                           ("hp_l2_reg","cont",+1)],
        "capacity":       [("hp_hidden_size","cont",+1),
                           ("hp_depth","cat",{"1":-1,"2":+1})],
        "regularization": [("hp_l1_reg","cont",+1),("hp_l2_reg","cont",+1)],
    },
}


def get_paradigm(task):
    if task in RNN_TASKS: return "rnn"
    if task in RL_TASKS:  return "rl"
    return "supervised"


def _col_to_str(col):
    try:
        fv = col.astype(float)
        result = fv.astype(str)
        int_mask = fv.notna() & (fv == fv.round())
        result[int_mask] = fv[int_mask].astype(int).astype(str)
        return result
    except (TypeError, ValueError):
        return col.astype(str)


def _compute_composite(df, comp_name, paradigm):
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
            parts.append(_col_to_str(col).map(param).astype(float))
    if not parts:
        return pd.Series(np.nan, index=df.index)
    s = sum(parts)
    return (s - s.mean()) / (s.std() + 1e-12)


# ── X-matrix builders ────────────────────────────────────────────────────────

def build_composite_X(df, task):
    """Return (matrix, feature_labels, row_valid_mask)."""
    paradigm = get_paradigm(task)
    cols, labels = [], []
    for c in COMPOSITES:
        cols.append(_compute_composite(df, c, paradigm).values)
        labels.append(COMP_LABELS[c])
    X = np.column_stack(cols).astype(float)
    return X, labels, np.all(np.isfinite(X), axis=1)


def build_hp_X(df, task):
    """
    Individual HP feature matrix with effect coding.
    Returns (matrix, feature_labels, row_valid_mask).
    Feature encoding:
      continuous  → z-scored within task
      binary cat  → ±1 (positive level = +1)
      activation  → two Helmert contrasts (orthogonal)
    """
    is_rnn = task in RNN_TASKS
    is_rl  = task in RL_TASKS
    cols, labels = [], []

    def add_cont(hp, label):
        if hp in df.columns:
            cols.append(df[hp].astype(float).values)
            labels.append(label)

    def add_bin(hp, label, pos_str):
        if hp in df.columns:
            coded = np.where(_col_to_str(df[hp]) == pos_str, 1.0, -1.0).astype(float)
            cols.append(coded)
            labels.append(label)

    # Continuous HPs (all paradigms share these except batch_size for RL)
    add_cont("hp_learning_rate", "lr")
    add_cont("hp_l1_reg",        "l1")
    add_cont("hp_l2_reg",        "l2")
    add_cont("hp_hidden_size",   "hidden\nsize")
    if not is_rl:
        add_cont("hp_batch_size", "batch\nsize")

    # Binary categoricals
    add_bin("hp_optimizer",  "optimizer\n(adam=+1)",  "adam")
    add_bin("hp_init_scale", "init_scale\n(0.1=+1)",  "0.1")

    if is_rnn:
        add_bin("hp_cell_type",    "cell_type\n(gru=+1)", "gru")
        add_bin("hp_n_rnn_layers", "n_rnn_layers\n(2=+1)", "2")
    else:
        add_bin("hp_depth", "depth\n(2=+1)", "2")
        # Activation: two Helmert contrasts
        if "hp_activation" in df.columns:
            act = _col_to_str(df["hp_activation"])
            # H1: relu vs sigmoid (tanh neutral)
            h1 = np.where(act=="relu", -1.0, np.where(act=="sigmoid", 1.0, 0.0))
            # H2: tanh vs (relu+sigmoid)
            h2 = np.where(act=="tanh", 2.0, -1.0)
            cols.append(h1.astype(float))
            cols.append(h2.astype(float))
            labels.extend(["act:\nrelu↔sig", "act:\ntanh↔oth"])

    if not cols:
        return None, [], np.zeros(len(df), dtype=bool)

    X = np.column_stack(cols).astype(float)
    return X, labels, np.all(np.isfinite(X), axis=1)


def build_Y(df):
    cols = [df[p].astype(float).values for p in RDM_PROPS if p in df.columns]
    labels = [RDM_LABELS[p] for p in RDM_PROPS if p in df.columns]
    Y = np.column_stack(cols).astype(float)
    return Y, labels, np.all(np.isfinite(Y), axis=1)


# ── CCA core ─────────────────────────────────────────────────────────────────

def _zscore(M):
    mu = M.mean(axis=0)
    sd = M.std(axis=0, ddof=1)
    sd[sd < 1e-12] = 1.0
    return (M - mu) / sd


def _mat_inv_sqrt(C):
    evals, evecs = np.linalg.eigh(C)
    evals = np.maximum(evals, 1e-10)
    return evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T


def cca_svd(X_z, Y_z, ridge=RIDGE):
    """
    CCA via SVD on the cross-correlation matrix.
    X_z, Y_z: already z-scored, shape (N, p) and (N, q).
    Returns (canonical_rs, X_weights_all_pairs, Y_weights_all_pairs).
    """
    N = X_z.shape[0]
    Cxx = X_z.T @ X_z / (N - 1) + ridge * np.eye(X_z.shape[1])
    Cyy = Y_z.T @ Y_z / (N - 1) + ridge * np.eye(Y_z.shape[1])
    Cxy = X_z.T @ Y_z / (N - 1)

    Cxx_isqrt = _mat_inv_sqrt(Cxx)
    Cyy_isqrt = _mat_inv_sqrt(Cyy)

    K = Cxx_isqrt @ Cxy @ Cyy_isqrt
    U, S, Vt = np.linalg.svd(K, full_matrices=False)

    Wx = Cxx_isqrt @ U          # (p, k)
    Wy = Cyy_isqrt @ Vt.T       # (q, k)
    return S, Wx, Wy


def permutation_p(X_z, Y_z, r_obs, rng=RNG, n_perm=N_PERM):
    idx = np.arange(len(Y_z))
    count = 0
    for _ in range(n_perm):
        rng.shuffle(idx)
        S, _, _ = cca_svd(X_z, Y_z[idx])
        if S[0] >= r_obs:
            count += 1
    return (count + 1) / (n_perm + 1)


# ── Analysis loop ─────────────────────────────────────────────────────────────

def run_all(task_dfs):
    """
    Run composites-CCA and HP-CCA for every task.
    Returns dict: task → {composites: {...}, hps: {...}}
    """
    results = {}
    for task, df in task_dfs.items():
        print(f"  {task} ...", flush=True)
        Y_raw, y_labels, y_valid = build_Y(df)
        task_res = {}

        for kind, builder in [("composites", build_composite_X),
                               ("hps",        build_hp_X)]:
            X_raw, x_labels, x_valid = builder(df, task)
            if X_raw is None:
                task_res[kind] = None
                continue

            valid = x_valid & y_valid
            N = int(valid.sum())
            if N < 20:
                task_res[kind] = None
                continue

            X_z = _zscore(X_raw[valid])
            Y_z = _zscore(Y_raw[valid])

            S, Wx, Wy = cca_svd(X_z, Y_z)
            r = float(S[0])

            # Fix sign so first canonical variate pair is positively correlated
            Ax = X_z @ Wx[:, 0]
            Ay = Y_z @ Wy[:, 0]
            if np.corrcoef(Ax, Ay)[0, 1] < 0:
                Wx[:, 0] *= -1
                Wy[:, 0] *= -1

            print(f"    [{kind}] r={r:.3f}, N={N}, running permutation test ...",
                  flush=True)
            p = permutation_p(X_z, Y_z, r)

            task_res[kind] = {
                "r": r, "p": p, "N": N,
                "wx": Wx[:, 0].copy(),
                "wy": Wy[:, 0].copy(),
                "x_labels": x_labels,
                "y_labels": y_labels,
                "all_r": S.tolist(),
            }

        results[task] = task_res
    return results


# ── Figure helpers ────────────────────────────────────────────────────────────

def _p_str(p):
    if p < 0.001: return "p<.001"
    return f"p={p:.3f}"


def _make_heatmap_page(results, kind, tasks_by_paradigm, title):
    """
    One page: X-weights heatmap (top) + Y-weights heatmap (bottom), tasks as columns.
    For HP-CCA the X feature rows differ by paradigm, so we draw paradigm groups.
    """
    # Collect all present tasks and their weight vectors
    task_data = []
    for paradigm_name, task_list in tasks_by_paradigm:
        for task in task_list:
            if task not in results or results[task][kind] is None:
                continue
            d = results[task][kind]
            task_data.append((paradigm_name, task, d))

    if not task_data:
        return None

    n_y = len(RDM_PROPS)

    # For the X side: figure out whether all tasks share the same x_labels
    # (composites always do; HPs differ by paradigm)
    all_x_labels = [d["x_labels"] for _, _, d in task_data]
    uniform_x = all(xl == all_x_labels[0] for xl in all_x_labels)

    if uniform_x:
        return _make_uniform_page(task_data, kind, n_y, title)
    else:
        return _make_paradigm_page(task_data, tasks_by_paradigm, results, kind, n_y, title)


def _make_uniform_page(task_data, kind, n_y, title):
    """All tasks share the same x_labels — single pair of heatmaps."""
    x_labels = task_data[0][2]["x_labels"]
    n_x = len(x_labels)
    n_tasks = len(task_data)

    fig_w = max(10, 1.8 * n_tasks + 2.0)
    fig_h = max(6, 0.55 * (n_x + n_y) + 2.5)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h),
                             gridspec_kw={"height_ratios": [n_x, n_y], "hspace": 0.5})

    task_labels = [TASK_SHORT.get(t, t) for _, t, _ in task_data]
    r_vals      = [d["r"] for _, _, d in task_data]

    # X-weights heatmap
    X_mat = np.column_stack([d["wx"] for _, _, d in task_data])  # (n_x, n_tasks)
    _draw_heatmap(axes[0], X_mat, row_labels=x_labels,
                  col_labels=task_labels, col_vals=r_vals,
                  ylabel="HP / composite weights")

    # Y-weights heatmap
    Y_mat = np.column_stack([d["wy"] for _, _, d in task_data])  # (n_y, n_tasks)
    y_labels_row = [RDM_LABELS[p] for p in RDM_PROPS]
    _draw_heatmap(axes[1], Y_mat, row_labels=y_labels_row,
                  col_labels=task_labels, col_vals=r_vals,
                  ylabel="RDM property weights", show_p=True,
                  p_vals=[d["p"] for _, _, d in task_data],
                  n_vals=[d["N"] for _, _, d in task_data])

    fig.suptitle(title, fontsize=10, fontweight="bold")
    return fig


def _make_paradigm_page(task_data_all, tasks_by_paradigm, results, kind, n_y, title):
    """Paradigm-specific x_labels — draw one panel group per paradigm."""
    # Group task_data by paradigm
    by_paradigm = {}
    for pname, task, d in task_data_all:
        by_paradigm.setdefault(pname, []).append((task, d))

    paradigm_names = [pn for pn, _ in tasks_by_paradigm if pn in by_paradigm]
    n_paradigms = len(paradigm_names)

    # Compute row heights: each paradigm needs (n_x_p + n_y) rows
    row_heights = []
    for pn in paradigm_names:
        n_x = len(by_paradigm[pn][0][1]["x_labels"])
        row_heights.extend([n_x, n_y])
    # Add a small gap row between paradigms
    interleaved = []
    for i, pn in enumerate(paradigm_names):
        n_x = len(by_paradigm[pn][0][1]["x_labels"])
        interleaved.append(n_x)
        interleaved.append(n_y)
        if i < len(paradigm_names) - 1:
            interleaved.append(0.4)  # gap

    # Widths: one per task (use the widest paradigm)
    max_n_tasks = max(len(by_paradigm[pn]) for pn in paradigm_names)
    fig_w = max(10, 1.8 * max_n_tasks + 2.5)
    total_rows = sum(interleaved)
    fig_h = max(8, 0.55 * total_rows + 2.0)

    n_grid_rows = len(interleaved)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(n_grid_rows, max_n_tasks, height_ratios=interleaved,
                          hspace=0.5, wspace=0.3,
                          top=0.93, bottom=0.06, left=0.14, right=0.97)

    grid_row = 0
    for pn in paradigm_names:
        tasks_d = by_paradigm[pn]
        n_tasks = len(tasks_d)
        task_labels = [TASK_SHORT.get(t, t) for t, _ in tasks_d]
        n_x = len(tasks_d[0][1]["x_labels"])

        # X-weights
        ax_x = fig.add_subplot(gs[grid_row, :n_tasks])
        X_mat = np.column_stack([d["wx"] for _, d in tasks_d])
        _draw_heatmap(ax_x, X_mat,
                      row_labels=tasks_d[0][1]["x_labels"],
                      col_labels=task_labels,
                      col_vals=[d["r"] for _, d in tasks_d],
                      ylabel=f"{pn}\nHP weights")

        # Y-weights
        ax_y = fig.add_subplot(gs[grid_row + 1, :n_tasks])
        Y_mat = np.column_stack([d["wy"] for _, d in tasks_d])
        _draw_heatmap(ax_y, Y_mat,
                      row_labels=[RDM_LABELS[p] for p in RDM_PROPS],
                      col_labels=task_labels,
                      col_vals=[d["r"] for _, d in tasks_d],
                      ylabel="RDM weights",
                      show_p=True,
                      p_vals=[d["p"] for _, d in tasks_d],
                      n_vals=[d["N"] for _, d in tasks_d])

        # Hide unused columns in this paradigm's rows
        for col in range(n_tasks, max_n_tasks):
            for r in [grid_row, grid_row + 1]:
                try:
                    ax = fig.add_subplot(gs[r, col])
                    ax.set_visible(False)
                except Exception:
                    pass

        grid_row += 2
        if grid_row < n_grid_rows:
            grid_row += 1  # skip gap row

    fig.suptitle(title, fontsize=10, fontweight="bold")
    return fig


def _draw_heatmap(ax, mat, row_labels, col_labels, col_vals,
                  ylabel="", show_p=False, p_vals=None, n_vals=None):
    """
    Draw a diverging heatmap. col_vals = canonical r per column (annotated on top).
    """
    vmax = max(0.3, np.nanmax(np.abs(mat)))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7, labelpad=3)

    # Annotate cells
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            if np.isfinite(v):
                ax.text(c, r, f"{v:+.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(v) > 0.5 * vmax else "black")

    # Column headers: canonical r (and optionally p, N)
    for c, (r_val, col_lbl) in enumerate(zip(col_vals, col_labels)):
        txt = f"r={r_val:.2f}"
        if show_p and p_vals is not None:
            txt += f"\n{_p_str(p_vals[c])}"
            if n_vals is not None:
                txt += f"\nN={n_vals[c]}"
        ax.set_title(txt if not show_p else "", fontsize=6.5)  # only on top panel

    if show_p and p_vals is not None:
        # Annotate below x-axis ticks
        for c, (r_val, p_val) in enumerate(zip(col_vals, p_vals)):
            n_str = f" N={n_vals[c]}" if n_vals else ""
            ax.annotate(
                f"r={r_val:.2f}\n{_p_str(p_val)}{n_str}",
                xy=(c, mat.shape[0] - 0.5), xycoords="data",
                xytext=(c, mat.shape[0] + 0.3), textcoords="data",
                fontsize=6, ha="center", va="bottom", annotation_clip=False,
            )


# ── CSV output ────────────────────────────────────────────────────────────────

def make_csv(results):
    rows = []
    for task, task_res in results.items():
        for kind in ("composites", "hps"):
            d = task_res.get(kind)
            if d is None:
                continue
            row = {
                "task": task, "kind": kind,
                "canonical_r": d["r"], "p_value": d["p"], "N": d["N"],
            }
            for i, r in enumerate(d["all_r"]):
                row[f"r_pair{i+1}"] = r
            for i, (lbl, w) in enumerate(zip(d["x_labels"], d["wx"])):
                row[f"wx_{i+1}_{lbl.replace(chr(10),'_')}"] = w
            for i, (lbl, w) in enumerate(zip(d["y_labels"], d["wy"])):
                row[f"wy_{i+1}_{lbl}"] = w
            rows.append(row)
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CCA on HP composites / individual HPs vs RDM properties.")
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
    task_dfs = {}
    for task in TASK_NAMES:
        sub = all_df[all_df["task"] == task].copy()
        if len(sub) >= 20:
            task_dfs[task] = sub
            print(f"  {task}: {len(sub)} networks")

    print("\nRunning CCA ...")
    results = run_all(task_dfs)

    print("\nCanonical correlations:")
    for task, tr in results.items():
        for kind in ("composites", "hps"):
            d = tr.get(kind)
            if d:
                print(f"  {task:16s} [{kind:10s}]  r={d['r']:.3f}  {_p_str(d['p'])}  N={d['N']}")

    tasks_by_paradigm = []
    for paradigm_name, task_list in PARADIGMS:
        present = [t for t in task_list if t in results]
        if present:
            tasks_by_paradigm.append((paradigm_name, present))

    pdf_path = out_figures / "f2_cca.pdf"
    with PdfPages(pdf_path) as pdf:
        for kind, label in [("composites", "CCA — composites"),
                             ("hps",        "CCA — individual HPs")]:
            full_title = (
                f"{label} × RDM properties (first canonical pair)\n"
                f"metric={args.metric}, ridge={RIDGE}, {N_PERM} permutations"
            )
            fig = _make_heatmap_page(results, kind, tasks_by_paradigm, full_title)
            if fig is not None:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    print(f"\nSaved: {pdf_path}")

    csv_path = out_tables / "rdm_cca.csv"
    make_csv(results).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
