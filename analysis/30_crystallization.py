#!/usr/bin/env python3
"""
Step 30: Crystallization — similarity to final representation over learning.

For each network, compute Spearman r between the RDM at each perf checkpoint
and the reference (best for supervised/rnn, final for rl) RDM.
Plots 'similarity to final' curves over normalized performance, per task.

Outputs:
    output/analysis/{metric}/figures/f3_crystallization.pdf
    output/analysis/{metric}/tables/f3_crystallization.csv
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ANALYSIS = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import (
    FIGURES_DIR, RDM_DIR, TABLES_DIR, TASK_NAMES, RL_TASKS,
    metric_output_dirs,
)

RNN_TASKS = {"adding", "mnist_rnn"}

PERF_LEVELS = [0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95]

PARADIGMS = [
    ("Supervised", ["mnist_dual", "mnist_10way", "fashion_10way", "spirals", "parity"]),
    ("RNN",        ["adding", "mnist_rnn"]),
    ("RL",         ["cartpole", "fourrooms"]),
]

TASK_SHORT = {
    "mnist_dual":    "MNIST dual",
    "mnist_10way":   "MNIST 10way",
    "fashion_10way": "Fashion 10way",
    "spirals":       "Spirals",
    "parity":        "Parity",
    "adding":        "Adding",
    "mnist_rnn":     "MNIST RNN",
    "cartpole":      "CartPole",
    "fourrooms":     "FourRooms",
}

PARADIGM_COLOR = {
    "Supervised": "#2166ac",
    "RNN":        "#1a9641",
    "RL":         "#d73027",
}


def label_to_float(label):
    """'0p025' → 0.025, '0p9' → 0.9, '0p85' → 0.85"""
    return float(label.replace("p", "."))


def ref_ckpt_name(task):
    return "final" if task in RL_TASKS else "best"


def last_layer_key(task, rg, metric):
    """Return the dataset key for the last hidden layer RDM of this network."""
    if task in RNN_TASKS:
        return f"temporal_{metric}"
    depth = int(rg.attrs.get("hp_depth", 1))
    return f"layer_{depth - 1}_{metric}"


def load_rdm_vec(cg, key):
    """Load RDM upper-triangle vector. Returns None if absent/degenerate."""
    ds = cg.get(key)
    if ds is None or ds.attrs.get("degenerate", False) or len(ds) == 0:
        return None
    return ds[:].astype(np.float64)


def load_crystallization_data(task, metric):
    """
    For each primary network in task, collect (perf_level, spearman_r) pairs
    where spearman_r is the correlation between the perf_* RDM and the reference.

    Returns a list of dicts with keys: run_id, perf_level, spearman_r.
    """
    h5_path = RDM_DIR / f"{task}_rdms.h5"
    if not h5_path.exists():
        return []

    ref_name = ref_ckpt_name(task)
    rows = []

    with h5py.File(h5_path, "r") as h5:
        for run_id, rg in h5["runs"].items():
            if rg.attrs.get("is_repeat", False):
                continue

            lkey = last_layer_key(task, rg, metric)
            ref_cg = rg.get(ref_name)
            if ref_cg is None:
                continue
            ref_vec = load_rdm_vec(ref_cg, lkey)
            if ref_vec is None:
                continue

            for ckpt_name, cg in rg.items():
                if not ckpt_name.startswith("perf_"):
                    continue
                perf_level = label_to_float(ckpt_name[len("perf_"):])
                vec = load_rdm_vec(cg, lkey)
                if vec is None:
                    continue
                r, _ = spearmanr(vec, ref_vec)
                rows.append({
                    "task":       task,
                    "run_id":     run_id,
                    "perf_level": perf_level,
                    "spearman_r": float(r),
                })

    return rows


def summarize_by_level(rows_df):
    """
    For each task × perf_level, compute mean, SE, and n.
    Returns a DataFrame with those columns.
    """
    if rows_df.empty:
        return rows_df
    grouped = rows_df.groupby(["task", "perf_level"])["spearman_r"]
    summary = grouped.agg(
        mean="mean",
        std="std",
        n="count",
    ).reset_index()
    summary["se"] = summary["std"] / np.sqrt(summary["n"])
    summary["ci95"] = 1.96 * summary["se"]
    return summary


def crystallization_threshold(summary, task, threshold=0.9):
    """First perf_level where mean >= threshold, or None."""
    sub = summary[summary["task"] == task].sort_values("perf_level")
    above = sub[sub["mean"] >= threshold]
    return float(above["perf_level"].iloc[0]) if len(above) else None


def make_figure(all_rows_df, summary_df):
    tasks_by_paradigm = []
    for paradigm_name, task_list in PARADIGMS:
        present = [t for t in task_list if t in all_rows_df["task"].values]
        if present:
            tasks_by_paradigm.append((paradigm_name, present))

    n_rows = len(tasks_by_paradigm)
    n_cols = max(len(tl) for _, tl in tasks_by_paradigm)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )

    for row_idx, (paradigm_name, task_list) in enumerate(tasks_by_paradigm):
        color = PARADIGM_COLOR[paradigm_name]
        for col_idx, task in enumerate(task_list):
            ax = axes[row_idx][col_idx]
            task_sum = summary_df[summary_df["task"] == task].sort_values("perf_level")
            task_raw = all_rows_df[all_rows_df["task"] == task]

            if task_sum.empty:
                ax.set_visible(False)
                continue

            # Per-network traces (thin, low alpha)
            for run_id, grp in task_raw.groupby("run_id"):
                grp = grp.sort_values("perf_level")
                ax.plot(grp["perf_level"], grp["spearman_r"],
                        color=color, alpha=0.05, linewidth=0.5)

            # Mean ± 95% CI
            ax.fill_between(
                task_sum["perf_level"],
                task_sum["mean"] - task_sum["ci95"],
                task_sum["mean"] + task_sum["ci95"],
                color=color, alpha=0.25,
            )
            ax.plot(task_sum["perf_level"], task_sum["mean"],
                    color=color, linewidth=2.0, zorder=3)

            # Threshold line at r=0.9
            ax.axhline(0.9, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)

            # Mark first level exceeding 0.9
            thr = crystallization_threshold(summary_df, task)
            if thr is not None:
                ax.axvline(thr, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
                ax.text(thr + 0.01, 0.05, f"≥0.9\n@{thr:.2f}",
                        fontsize=6, color="grey", va="bottom")

            ax.set_xlim(0, 1.0)
            ax.set_ylim(-0.1, 1.05)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_xticklabels(["0", ".2", ".4", ".6", ".8", "1"], fontsize=7)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0", ".25", ".5", ".75", "1"], fontsize=7)
            ax.set_title(TASK_SHORT.get(task, task), fontsize=8, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(paradigm_name + "\nSpearman r", fontsize=7, fontweight="bold")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Normalised performance", fontsize=7)
            n_networks = task_raw["run_id"].nunique()
            ax.text(0.02, 0.97, f"n={n_networks}", transform=ax.transAxes,
                    fontsize=6, va="top", color="grey")

        for col_idx in range(len(task_list), n_cols):
            axes[row_idx][col_idx].set_visible(False)

    fig.suptitle(
        "Crystallization: similarity to final representation over learning\n"
        "(Spearman r between RDM at each perf checkpoint and reference [best/final] RDM)",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["cosine", "pearson"], default="cosine")
    args = parser.parse_args()

    out_figures, out_tables = metric_output_dirs(args.metric)
    out_figures.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for task in TASK_NAMES:
        print(f"  {task} ...", flush=True)
        rows = load_crystallization_data(task, args.metric)
        all_rows.extend(rows)
        n_nets = len({r["run_id"] for r in rows})
        print(f"    {n_nets} networks, {len(rows)} (network × checkpoint) pairs")

    if not all_rows:
        print("No data found.")
        return

    df = pd.DataFrame(all_rows)
    summary = summarize_by_level(df)

    # Summary stats
    print("\n=== Crystallization thresholds (mean r first >= 0.9) ===")
    for task in TASK_NAMES:
        thr = crystallization_threshold(summary, task)
        if thr is not None:
            print(f"  {task:20s}  perf_level={thr:.3f}")
        else:
            n_max = summary[summary["task"] == task]["mean"].max() if task in summary["task"].values else float("nan")
            print(f"  {task:20s}  never reached (max mean r={n_max:.3f})")

    # Save
    df.to_csv(out_tables / "f3_crystallization.csv", index=False)
    summary.to_csv(out_tables / "f3_crystallization_summary.csv", index=False)
    print(f"\nSaved tables.")

    fig = make_figure(df, summary)
    out_path = out_figures / "f3_crystallization.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
