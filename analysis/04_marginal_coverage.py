#!/usr/bin/env python3
"""
Step 4: Marginal coverage.

For each task and each HP:
  - Continuous (10 equal bins on unit [0,1]): fraction of bins with >=1 primary
    network and fraction with >=1 successful network.
  - Categorical (one bin per level): fraction of levels covered.

Output:
  analysis/figures/marginal_coverage.pdf
  analysis/tables/marginal_coverage.csv
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import (
    FIGURES_DIR,
    PRODUCTION_DIR,
    TABLES_DIR,
    TASK_NAMES,
    load_task_df,
    primary_df,
    task_meta,
)

N_CONT_BINS = 10

# Display-friendly HP name abbreviations
HP_LABELS = {
    "learning_rate": "lr",
    "l1_reg":        "l1",
    "l2_reg":        "l2",
    "hidden_size":   "hidden",
    "batch_size":    "batch",
    "depth":         "depth",
    "activation":    "activ.",
    "optimizer":     "optim.",
    "init_scale":    "init",
    "cell_type":     "cell",
    "n_rnn_layers":  "layers",
    "n_steps":       "steps",
    "n_envs":        "envs",
}


def load_adding_failed() -> pd.DataFrame:
    state_path = PRODUCTION_DIR / "adding_failed_run" / "bo_state.json"
    meta = task_meta()["adding"]
    cont_names = meta["cont_param_names"]
    rows = []
    for obs in json.load(open(state_path)):
        row = {
            "task":        "adding",
            "paradigm":    meta["paradigm"],
            "iteration":   obs["iteration"],
            "is_repeat":   obs.get("is_repeat", False),
            "performance": obs["performance"],
        }
        for k, v in obs["config"].items():
            row[k] = v
        unit_vals = obs.get("cont_unit_vals", [])
        for i, name in enumerate(cont_names):
            row[f"unit_{name}"] = unit_vals[i] if i < len(unit_vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_for_task(df_prim, df_succ, cont_names, cat_names, cat_choices):
    """Return list of coverage dicts for one task."""
    rows = []

    # --- Continuous HPs --------------------------------------------------
    edges = np.linspace(0, 1, N_CONT_BINS + 1)
    for name in cont_names:
        col = f"unit_{name}"
        if col not in df_prim.columns:
            continue
        prim_vals = df_prim[col].dropna().values
        succ_vals = df_succ[col].dropna().values

        # np.digitize returns 1-indexed bin numbers; cap top edge into last bin
        def n_occupied(vals):
            if len(vals) == 0:
                return 0
            b = np.clip(np.digitize(vals, edges) - 1, 0, N_CONT_BINS - 1)
            return len(np.unique(b))

        n_prim = n_occupied(prim_vals)
        n_succ = n_occupied(succ_vals)
        rows.append({
            "hp":                 name,
            "hp_label":           HP_LABELS.get(name, name),
            "hp_type":            "continuous",
            "n_bins":             N_CONT_BINS,
            "n_bins_primary":     n_prim,
            "n_bins_successful":  n_succ,
            "coverage_primary":   n_prim / N_CONT_BINS,
            "coverage_successful": n_succ / N_CONT_BINS,
        })

    # --- Categorical HPs -------------------------------------------------
    for name in cat_names:
        if name not in df_prim.columns:
            continue
        levels  = cat_choices[name]
        n_total = len(levels)
        n_prim  = sum(1 for l in levels if (df_prim[name] == l).any())
        n_succ  = sum(1 for l in levels if (df_succ[name] == l).any())
        rows.append({
            "hp":                 name,
            "hp_label":           HP_LABELS.get(name, name),
            "hp_type":            "categorical",
            "n_bins":             n_total,
            "n_bins_primary":     n_prim,
            "n_bins_successful":  n_succ,
            "coverage_primary":   n_prim / n_total,
            "coverage_successful": n_succ / n_total,
        })

    return rows


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    meta       = task_meta()
    thresholds = json.load(open(TABLES_DIR / "success_thresholds.json"))

    plot_tasks = [t for t in TASK_NAMES if t != "adding"] + ["adding"]
    display    = {"adding": "adding (ref)"}

    # --- Compute coverage ------------------------------------------------
    all_rows = []
    task_cov = {}   # task → list of coverage dicts

    for task in plot_tasks:
        if task == "adding":
            p = PRODUCTION_DIR / "adding_failed_run" / "bo_state.json"
            if not p.exists():
                continue
            df = load_adding_failed()
        else:
            p = PRODUCTION_DIR / task / "bo_state.json"
            if not p.exists():
                continue
            df = load_task_df(task)

        prim = primary_df(df)
        upper = thresholds.get(task, {}).get("upper")
        succ = prim[prim["performance"] >= upper] if upper is not None else prim.iloc[0:0]

        m = meta[task]
        cov = coverage_for_task(
            prim, succ,
            m["cont_param_names"],
            m["cat_param_names"],
            m["cat_param_choices"],
        )
        for row in cov:
            row["task"]     = task
            row["paradigm"] = m["paradigm"]
        all_rows.extend(cov)
        task_cov[task] = cov
        print(f"  {task}: {len(prim)} primary, {len(succ)} successful")

    df_cov = pd.DataFrame(all_rows)
    df_cov.to_csv(TABLES_DIR / "marginal_coverage.csv", index=False)

    # --- Plot ------------------------------------------------------------
    # Layout: one column per paradigm, tasks stacked in rows within each column
    paradigm_order = ["supervised", "rnn", "rl"]
    paradigm_tasks = {p: [t for t in plot_tasks if t in task_cov and meta[t]["paradigm"] == p]
                      for p in paradigm_order}
    max_rows = max(len(v) for v in paradigm_tasks.values())
    ncols    = len(paradigm_order)

    fig, axes = plt.subplots(
        max_rows, ncols,
        figsize=(ncols * 5.5, max_rows * 2.8),
        squeeze=False,
    )

    bar_w = 0.35

    for col_i, paradigm in enumerate(paradigm_order):
        tasks_in_col = paradigm_tasks[paradigm]
        for row_i, task in enumerate(tasks_in_col):
            ax  = axes[row_i][col_i]
            cov = task_cov[task]

            labels   = [r["hp_label"] for r in cov]
            cov_prim = [r["coverage_primary"] for r in cov]
            cov_succ = [r["coverage_successful"] for r in cov]
            x        = np.arange(len(labels))

            # Shade to distinguish continuous vs categorical
            cat_mask = [r["hp_type"] == "categorical" for r in cov]

            ax.bar(x - bar_w/2, cov_prim, bar_w, label="primary",    color="steelblue", alpha=0.85)
            ax.bar(x + bar_w/2, cov_succ, bar_w, label="successful",  color="seagreen",  alpha=0.85)

            # Divider between continuous and categorical
            n_cont = sum(1 for r in cov if r["hp_type"] == "continuous")
            if n_cont < len(cov):
                ax.axvline(n_cont - 0.5, color="grey", lw=0.8, ls="--")
                ax.text(n_cont - 0.45, 1.02, "categorical", fontsize=6,
                        color="grey", transform=ax.get_xaxis_transform())
                ax.text(-0.45, 1.02, "continuous", fontsize=6,
                        color="grey", transform=ax.get_xaxis_transform())

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_ylim(0, 1.08)
            ax.set_ylabel("fraction covered", fontsize=7)
            ax.set_title(display.get(task, task), fontsize=9, fontweight="bold")
            ax.tick_params(axis="y", labelsize=7)
            ax.axhline(1.0, color="black", lw=0.5, ls=":")
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=7)

        # Hide unused rows in this column
        for row_i in range(len(tasks_in_col), max_rows):
            axes[row_i][col_i].set_visible(False)

    # Column headers (paradigm labels)
    for col_i, paradigm in enumerate(paradigm_order):
        axes[0][col_i].set_title(
            f"{paradigm.upper()}\n{display.get(paradigm_tasks[paradigm][0], paradigm_tasks[paradigm][0])}",
            fontsize=9, fontweight="bold"
        )

    fig.suptitle(
        f"Marginal HP coverage — primary (blue) vs successful (green)  |  "
        f"continuous: {N_CONT_BINS} bins, categorical: all levels",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = FIGURES_DIR / "marginal_coverage.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nSaved figure: {out}")
    print(f"Saved table:  {TABLES_DIR / 'marginal_coverage.csv'}")


if __name__ == "__main__":
    main()
