#!/usr/bin/env python3
"""
Step 2: Sorted performance curves with alpha threshold lines.

For each task, plots performance (y) vs. percentile rank (x), sorted ascending.
Overlays threshold lines for alpha in [0.95, 0.90, 0.85, 0.80, 0.75].

Thresholds are in normalised performance space:
  empirical_max  = 95th percentile of primary performances
  upper(alpha)   = chance + alpha * (empirical_max - chance)   → successful
  lower(alpha)   = chance + (1-alpha) * (empirical_max - chance) → near-chance

Three categories (for a chosen alpha):
  performance >= upper(alpha)  →  successful
  performance <= lower(alpha)  →  near chance
  in between                   →  better than chance

Inspect the figure, pick an alpha, then fill in
analysis/tables/success_thresholds.json.

Output:
  analysis/figures/performance_lorenz.pdf
  analysis/tables/success_thresholds.json  (template, if not already present)
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

ALPHAS  = [0.90]
COLORS  = ["#2166ac"]
EMP_MAX_PERCENTILE = 95   # use p95 as empirical ceiling to avoid outliers


def load_adding_failed(production_dir: Path) -> pd.DataFrame:
    """Load adding_failed_run using 'adding' task metadata."""
    state_path = production_dir / "adding_failed_run" / "bo_state.json"
    meta = task_meta()["adding"]
    cont_names = meta["cont_param_names"]
    rows = []
    for obs in json.load(open(state_path)):
        row = {
            "task":        "adding_failed_run",
            "paradigm":    meta["paradigm"],
            "iteration":   obs["iteration"],
            "is_repeat":   obs.get("is_repeat", False),
            "repeat_of":   obs.get("repeat_of"),
            "performance": obs["performance"],
        }
        for k, v in obs["config"].items():
            row[k] = v
        unit_vals = obs.get("cont_unit_vals", [])
        for i, name in enumerate(cont_names):
            row[f"unit_{name}"] = unit_vals[i] if i < len(unit_vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    meta = task_meta()

    # --- Load data -------------------------------------------------------
    plot_tasks = [t for t in TASK_NAMES if t != "adding"] + ["adding_failed_run"]
    display    = {"adding_failed_run": "adding (ref run)"}

    task_data = {}
    for task in plot_tasks:
        if task == "adding_failed_run":
            p = PRODUCTION_DIR / "adding_failed_run" / "bo_state.json"
            if not p.exists():
                print(f"  [skip] adding_failed_run: not found")
                continue
            df = load_adding_failed(PRODUCTION_DIR)
        else:
            p = PRODUCTION_DIR / task / "bo_state.json"
            if not p.exists():
                print(f"  [skip] {task}: no bo_state.json")
                continue
            df = load_task_df(task)
        task_data[task] = primary_df(df)
        print(f"  loaded {task}: {len(task_data[task])} primary obs")

    # --- Plot ------------------------------------------------------------
    ncols   = 3
    nrows   = (len(task_data) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
    axes = axes.flatten()

    ordered = [t for t in plot_tasks if t in task_data]

    for ax, task in zip(axes, ordered):
        df   = task_data[task]
        perf = np.sort(df["performance"].values)
        x    = np.linspace(0, 1, len(perf))

        meta_key  = "adding" if task == "adding_failed_run" else task
        m         = meta[meta_key]
        chance    = m["chance_perf"]
        emp_max   = np.percentile(perf, EMP_MAX_PERCENTILE)

        ax.plot(x, perf, lw=1.5, color="steelblue", zorder=3)
        ax.axhline(chance, color="black", lw=1.0, ls=":", zorder=2,
                   label="chance")

        for alpha, color in zip(ALPHAS, COLORS):
            upper = chance + alpha       * (emp_max - chance)
            lower = chance + (1 - alpha) * (emp_max - chance)
            ax.axhline(upper, color=color, lw=1.1, ls="-",  zorder=2)
            ax.axhline(lower, color=color, lw=1.1, ls="--", zorder=2,
                       label=f"α={alpha}")

        ax.set_title(display.get(task, task), fontsize=10, fontweight="bold")
        ax.set_xlabel("percentile rank", fontsize=8)
        ax.set_ylabel(m["metric_name"], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.text(0.98, 0.03, f"n = {len(df)}", transform=ax.transAxes,
                fontsize=7, ha="right", va="bottom", color="grey")

    # Shared legend from last axis
    handles, labels = axes[len(ordered) - 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=8,
               title="— upper  -- lower", title_fontsize=7,
               framealpha=0.9, ncol=2)

    for ax in axes[len(ordered):]:
        ax.set_visible(False)

    fig.suptitle(
        "Sorted performance — primary obs  |  solid = upper threshold, dashed = lower threshold",
        fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])

    out = FIGURES_DIR / "performance_lorenz.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Print per-task threshold values for each alpha (raw performance)
    print(f"\n{'task':<20} {'chance':>8} {'p95':>8}  " +
          "  ".join(f"α={a} (up/lo)" for a in ALPHAS))
    for task in ordered:
        df      = task_data[task]
        perf    = df["performance"].values
        mk      = "adding" if task == "adding_failed_run" else task
        chance  = meta[mk]["chance_perf"]
        emp_max = np.percentile(perf, EMP_MAX_PERCENTILE)
        vals    = "  ".join(
            f"{chance + a*(emp_max-chance):.3f}/{chance + (1-a)*(emp_max-chance):.3f}"
            for a in ALPHAS
        )
        print(f"{display.get(task,task):<20} {chance:>8.3f} {emp_max:>8.3f}  {vals}")

    # --- Threshold template ----------------------------------------------
    thresh_path = TABLES_DIR / "success_thresholds.json"
    if not thresh_path.exists():
        template = {"_alpha": None}
        for t in ordered:
            key = "adding" if t == "adding_failed_run" else t
            template[key] = {"upper": None, "lower": None}
        json.dump(template, open(thresh_path, "w"), indent=2)
        print(f"\nCreated threshold template: {thresh_path}")
        print("Set _alpha and the script will compute upper/lower automatically,")
        print("or override per-task values manually.")
    else:
        print(f"\nThreshold file already exists ({thresh_path}); not overwritten.")


if __name__ == "__main__":
    main()
