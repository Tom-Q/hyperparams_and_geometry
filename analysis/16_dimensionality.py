#!/usr/bin/env python3
"""
Step 16: Effective dimensionality — Finding #1.5.

For each primary network, computes the participation ratio (PR) of the
activation covariance in stimulus space:

    PR = (Σ λ_i)² / Σ λ_i²

where λ_i are eigenvalues of the (N_stim × N_stim) Gram matrix X_c @ X_c.T,
and X_c = activations - mean_across_stimuli.

PR = 1 → all variance on one dimension (maximally collapsed).
PR = N_stim → flat spectrum (maximally spread, like white noise).

We compute this for the last hidden layer (best/final checkpoint). For depth=2
networks, we also compute per-layer PR to see how dimensionality changes
from layer_0 → layer_1.

Outputs:
    output/analysis/figures/f1_dimensionality.pdf
    output/analysis/tables/rdm_dimensionality.csv
"""

import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import (
    DATASET_DIR, FIGURES_DIR, RDM_DIR, TABLES_DIR, TASK_NAMES, RL_TASKS, task_meta,
)

TASK_DIR_OVERRIDES = {"adding": "adding_failed_run"}
RNN_TASKS          = {"adding", "mnist_rnn"}

TASK_LABELS = {
    "mnist_dual":    "MNIST dual\n(N=200)",
    "mnist_10way":   "MNIST 10-way\n(N=100)",
    "fashion_10way": "Fashion\n(N=100)",
    "spirals":       "Spirals\n(N=198)",
    "parity":        "Parity\n(N=118)",
    "adding":        "Adding\n(N=100)",
    "mnist_rnn":     "MNIST RNN\n(N=100)",
    "cartpole":      "CartPole\n(N=196)",
    "fourrooms":     "FourRooms\n(N=68)",
}


# ---------------------------------------------------------------------------
# PR computation
# ---------------------------------------------------------------------------

def participation_ratio(acts):
    """
    Compute participation ratio from (N_stim, H) activation matrix.
    Returns float, or np.nan if degenerate.
    """
    if not np.all(np.isfinite(acts)):
        return np.nan
    X_c = acts - acts.mean(axis=0)                     # centre across stimuli
    G   = X_c @ X_c.T                                  # (N_stim, N_stim)
    lam = np.linalg.eigvalsh(G)
    lam = lam[lam > 1e-10 * lam.max()]                 # keep positive eigenvalues
    if len(lam) == 0 or lam.sum() < 1e-12:
        return np.nan
    return float(lam.sum() ** 2 / (lam ** 2).sum())


# ---------------------------------------------------------------------------
# Per-task data loading
# ---------------------------------------------------------------------------

def ckpt_name(task):
    return "final" if task in RL_TASKS else "best"


def rnn_last_layer_key(npz_keys):
    """Infer last-layer, last-timestep key from npz for an RNN task."""
    parsed = []
    for k in npz_keys:
        if "_t_" not in k:
            continue
        parts = k.split("_")
        try:
            parsed.append((int(parts[1]), int(parts[3]), k))
        except (IndexError, ValueError):
            pass
    if not parsed:
        return None
    max_l = max(p[0] for p in parsed)
    max_t = max(p[1] for p in parsed if p[0] == max_l)
    return f"layer_{max_l}_t_{max_t}"


def load_task_dimensionality(task):
    """
    Load best/final npz for all primary networks, compute PR per layer.
    Returns list of dicts.
    """
    dirname  = TASK_DIR_OVERRIDES.get(task, task)
    task_dir = DATASET_DIR / dirname
    ckpt     = ckpt_name(task)
    h5_path  = RDM_DIR / f"{task}_rdms.h5"

    if not task_dir.exists() or not h5_path.exists():
        return []

    rows = []

    with h5py.File(h5_path, "r") as h5:
        runs_grp = h5.get("runs", {})
        for run_id in sorted(runs_grp.keys()):
            rg = runs_grp[run_id]
            if bool(rg.attrs.get("is_repeat", False)):
                continue

            perf       = float(rg.attrs.get("performance", float("nan")))
            depth      = int(rg.attrs.get("hp_depth", 1))
            hidden_sz  = int(rg.attrs.get("hp_hidden_size", 0))

            npz_path = task_dir / run_id / f"{ckpt}.npz"
            if not npz_path.exists():
                continue

            try:
                npz = np.load(npz_path)
            except Exception:
                continue

            row = {
                "task":        task,
                "run_id":      run_id,
                "performance": perf,
                "depth":       depth,
                "hidden_size": hidden_sz,
                "pr_last":     np.nan,
                "pr_l0":       np.nan,
                "pr_l1":       np.nan,
            }

            if task in RNN_TASKS:
                # last layer = max layer at last timestep
                last_key = rnn_last_layer_key(list(npz.keys()))
                if last_key is not None and last_key in npz:
                    row["pr_last"] = participation_ratio(npz[last_key])
                # also L0 last-t and L1 last-t separately
                t_keys = sorted(set(int(k.split("_t_")[1]) for k in npz.keys() if "_t_" in k))
                if t_keys:
                    max_t = max(t_keys)
                    k0 = f"layer_0_t_{max_t}"
                    k1 = f"layer_1_t_{max_t}"
                    if k0 in npz:
                        row["pr_l0"] = participation_ratio(npz[k0])
                    if k1 in npz:
                        row["pr_l1"] = participation_ratio(npz[k1])
            else:
                # last layer = layer_{depth-1}
                last_key = f"layer_{depth - 1}"
                if last_key in npz:
                    row["pr_last"] = participation_ratio(npz[last_key])
                if "layer_0" in npz:
                    row["pr_l0"] = participation_ratio(npz["layer_0"])
                if "layer_1" in npz:
                    row["pr_l1"] = participation_ratio(npz["layer_1"])

            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def load_thresholds():
    path = TABLES_DIR / "success_thresholds.json"
    if not path.exists():
        return {}
    data = json.load(open(path))
    return {k: (float(v["upper"]) if isinstance(v, dict) else None)
            for k, v in data.items() if k != "_alpha"}


def plot_dimensionality_main(df, thresholds):
    """
    3-panel figure:
      Left:   Violin of PR per task (all primary networks, last hidden layer)
      Middle: Scatter PR vs. normalised performance (all networks)
      Right:  Scatter PR vs. hidden_size (successful networks)
    """
    fig = plt.figure(figsize=(18, 6))
    gs  = fig.add_gridspec(1, 3, wspace=0.35)
    ax_v = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1])
    ax_h = fig.add_subplot(gs[2])

    tasks_ordered = [t for t in TASK_NAMES if t in df["task"].unique()]
    palette = plt.cm.tab10(np.linspace(0, 0.9, len(tasks_ordered)))

    # --- Left: violin per task (successful networks only) ---
    pos, labels = [], []
    for i, task in enumerate(tasks_ordered):
        th  = thresholds.get(task)
        sub = df[df["task"] == task]
        if th is not None:
            sub = sub[sub["performance"] >= th]
        vals = sub["pr_last"].dropna()
        if len(vals) < 3:
            continue
        pos.append(i)
        labels.append(TASK_LABELS.get(task, task))
        parts = ax_v.violinplot([vals], positions=[i], showmedians=True, showextrema=True)
        parts["bodies"][0].set_facecolor(palette[i])
        parts["bodies"][0].set_alpha(0.6)
    ax_v.set_xticks(pos)
    ax_v.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax_v.set_ylabel("Participation ratio (PR)", fontsize=9)
    ax_v.set_title("PR distribution per task\n(last hidden layer, successful networks)", fontsize=9)

    # --- Middle: PR vs. normalised performance ---
    meta = task_meta()
    for i, task in enumerate(tasks_ordered):
        sub = df[df["task"] == task].copy()
        th  = thresholds.get(task)
        chance = meta[task].get("chance_perf", 0)
        if th and th != chance:
            sub["norm_perf"] = (sub["performance"] - chance) / (th - chance)
        else:
            sub["norm_perf"] = sub["performance"]
        ax_p.scatter(sub["norm_perf"], sub["pr_last"],
                     s=1.5, alpha=0.3, color=palette[i], rasterized=True)
    ax_p.axvline(1, color="grey", lw=0.8, ls=":")
    ax_p.axhline(1, color="grey", lw=0.6, ls="--")
    ax_p.set_xlabel("norm. performance", fontsize=9)
    ax_p.set_ylabel("Participation ratio (PR)", fontsize=9)
    ax_p.set_title("PR vs. performance\n(all primary networks)", fontsize=9)
    ax_p.set_xlim(left=-0.3)

    # --- Right: PR vs. hidden_size (successful only) ---
    for i, task in enumerate(tasks_ordered):
        th  = thresholds.get(task)
        sub = df[df["task"] == task]
        if th is not None:
            sub = sub[sub["performance"] >= th]
        sub = sub.dropna(subset=["pr_last", "hidden_size"])
        if len(sub) == 0:
            continue
        # Jitter hidden_size slightly to show overlapping points
        jitter = np.random.default_rng(42).uniform(-0.4, 0.4, len(sub))
        ax_h.scatter(sub["hidden_size"] + jitter, sub["pr_last"],
                     s=1.5, alpha=0.3, color=palette[i], rasterized=True,
                     label=TASK_LABELS.get(task, task).replace("\n", " "))
    ax_h.axhline(1, color="grey", lw=0.6, ls="--")
    ax_h.set_xlabel("hidden_size", fontsize=9)
    ax_h.set_ylabel("Participation ratio (PR)", fontsize=9)
    ax_h.set_title("PR vs. hidden_size\n(successful networks only)", fontsize=9)
    ax_h.legend(fontsize=6, markerscale=3, loc="upper left", framealpha=0.8)

    fig.suptitle("Effective dimensionality — participation ratio of activation covariance\n"
                 "(stimulus space covariance, last hidden layer)",
                 fontsize=11)
    return fig


def plot_layer_dimensionality(df, thresholds):
    """
    Depth=2 networks: scatter PR_L0 vs PR_L1 per task,
    coloured by performance.
    """
    tasks_depth2 = [t for t in TASK_NAMES
                    if t in df["task"].unique() and
                    df[(df["task"] == t) & (df["depth"] == 2)]["pr_l1"].notna().sum() > 10]

    if not tasks_depth2:
        return None

    n = len(tasks_depth2)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                             squeeze=False)

    for ax, task in zip(axes.flatten(), tasks_depth2):
        sub = df[(df["task"] == task) & (df["depth"] == 2)].dropna(
            subset=["pr_l0", "pr_l1"])
        th  = thresholds.get(task)
        meta = task_meta()
        chance = meta[task].get("chance_perf", 0)
        if th and th != chance:
            sub = sub.copy()
            sub["norm_perf"] = (sub["performance"] - chance) / (th - chance)
        else:
            sub = sub.copy()
            sub["norm_perf"] = sub["performance"]

        sc = ax.scatter(sub["pr_l0"], sub["pr_l1"],
                        c=sub["norm_perf"], cmap="RdYlGn",
                        vmin=0, vmax=1.5, s=3, alpha=0.5, rasterized=True)
        lim = max(sub["pr_l0"].max(), sub["pr_l1"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4)
        ax.set_xlabel("PR layer 0", fontsize=8)
        ax.set_ylabel("PR layer 1", fontsize=8)
        ax.set_title(TASK_LABELS.get(task, task).replace("\n", " "), fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

    # Colorbar
    plt.colorbar(sc, ax=axes.flatten()[-1], label="norm. performance")

    for ax in axes.flatten()[len(tasks_depth2):]:
        ax.set_visible(False)

    fig.suptitle("Effective dimensionality: layer 0 vs. layer 1\n"
                 "(depth=2 networks; colour = normalised performance)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds()

    all_rows = []
    for task in TASK_NAMES:
        print(f"  {task} ...", end="", flush=True)
        rows = load_task_dimensionality(task)
        if not rows:
            print(" [no data]")
            continue
        all_rows.extend(rows)
        pr_vals = [r["pr_last"] for r in rows if np.isfinite(r["pr_last"])]
        print(f" {len(rows)} networks, PR med={np.median(pr_vals):.1f} "
              f"min={np.min(pr_vals):.1f} max={np.max(pr_vals):.1f}")

    df = pd.DataFrame(all_rows)

    # Save CSV
    csv_path = TABLES_DIR / "rdm_dimensionality.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Main figure
    fig_main = plot_dimensionality_main(df, thresholds)
    out_main = FIGURES_DIR / "f1_dimensionality.pdf"
    fig_main.savefig(out_main, bbox_inches="tight", dpi=150)
    plt.close(fig_main)
    print(f"Saved: {out_main}")

    # Layer comparison figure
    fig_layer = plot_layer_dimensionality(df, thresholds)
    if fig_layer is not None:
        out_layer = FIGURES_DIR / "f1_dimensionality_layers.pdf"
        fig_layer.savefig(out_layer, bbox_inches="tight", dpi=150)
        plt.close(fig_layer)
        print(f"Saved: {out_layer}")


if __name__ == "__main__":
    main()
