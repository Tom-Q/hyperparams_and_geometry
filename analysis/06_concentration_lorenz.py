#!/usr/bin/env python3
"""
Step 6: Concentration Lorenz curves and Gini coefficients.

Using the same 2-bin hypercubes as Step 5, for each task:
  - Count successful networks per occupied cell.
  - Lorenz curve: x = cumulative fraction of occupied cells (sorted ascending
    by successful count), y = cumulative fraction of successful networks.
  - Gini coefficient = area between curve and the diagonal of perfect equality.

A Gini of 0 means successful networks are perfectly spread across occupied cells.
A Gini of 1 means all successful networks are in a single cell.

Output:
  analysis/figures/concentration_lorenz.pdf
  analysis/tables/gini.csv
"""

import json
import sys
from collections import Counter, defaultdict
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


def cell_key(row, cont_names, cat_names):
    cont_bins = tuple(0 if row[f"unit_{n}"] < 0.5 else 1 for n in cont_names)
    cat_vals  = tuple(row[n] for n in cat_names)
    return cont_bins + cat_vals


def gini(values: np.ndarray) -> float:
    """Gini coefficient of an array of non-negative values."""
    v = np.sort(values.astype(float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum() / (n * v.sum())) - (n + 1) / n)


def lorenz_curve(values: np.ndarray):
    """Return (x, y) for the Lorenz curve, with (0,0) prepended."""
    v = np.sort(values.astype(float))
    n = len(v)
    x = np.concatenate([[0], np.arange(1, n + 1) / n])
    cs = np.concatenate([[0], np.cumsum(v)])
    y = cs / cs[-1] if cs[-1] > 0 else cs
    return x, y


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    meta       = task_meta()
    thresholds = json.load(open(TABLES_DIR / "success_thresholds.json"))

    plot_tasks = [t for t in TASK_NAMES if t != "adding"] + ["adding"]

    # --- Compute per-task Lorenz data ------------------------------------
    task_lorenz = {}   # task → (x, y, gini_val, n_occupied, n_succ)
    gini_rows   = []

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

        prim  = primary_df(df)
        upper = thresholds.get(task, {}).get("upper")
        succ  = prim[prim["performance"] >= upper] if upper is not None else prim.iloc[0:0]

        m          = meta[task]
        cont_names = m["cont_param_names"]
        cat_names  = m["cat_param_names"]

        # Count successful networks per occupied cell
        succ_per_cell = defaultdict(int)
        for _, obs in prim.iterrows():
            succ_per_cell[cell_key(obs, cont_names, cat_names)] += 0  # ensure cell exists
        for _, obs in succ.iterrows():
            succ_per_cell[cell_key(obs, cont_names, cat_names)] += 1

        counts    = np.array(list(succ_per_cell.values()))
        g         = gini(counts)
        x, y      = lorenz_curve(counts)

        task_lorenz[task] = (x, y, g)
        gini_rows.append({
            "task":       task,
            "paradigm":   m["paradigm"],
            "n_occupied": len(counts),
            "n_successful": int(counts.sum()),
            "gini":       round(g, 4),
        })
        print(f"  {task:<20}  occupied={len(counts):>4}  "
              f"successful={int(counts.sum()):>4}  Gini={g:.3f}")

    df_gini = pd.DataFrame(gini_rows)
    df_gini.to_csv(TABLES_DIR / "gini.csv", index=False)

    # --- Plot ------------------------------------------------------------
    paradigm_order = ["supervised", "rnn", "rl"]
    paradigm_tasks = {
        p: [t for t in plot_tasks if t in task_lorenz and meta[t]["paradigm"] == p]
        for p in paradigm_order
    }

    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, paradigm in zip(axes, paradigm_order):
        tasks = paradigm_tasks[paradigm]
        for i, task in enumerate(tasks):
            x, y, g = task_lorenz[task]
            label = f"{'adding (ref)' if task == 'adding' else task}  G={g:.3f}"
            ax.plot(x, y, lw=1.8, color=colors[i], label=label)

        # Diagonal = perfect equality
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect equality")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("cumulative fraction of occupied cells", fontsize=8)
        ax.set_ylabel("cumulative fraction of successful networks", fontsize=8)
        ax.set_title(paradigm.upper(), fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_aspect("equal")

    fig.suptitle(
        "Concentration Lorenz curves — successful networks across 2-bin hypercubes\n"
        "Curve below diagonal → concentrated; on diagonal → uniform",
        fontsize=9,
    )
    fig.tight_layout()

    out = FIGURES_DIR / "concentration_lorenz.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nSaved figure: {out}")
    print(f"Saved table:  {TABLES_DIR / 'gini.csv'}")


if __name__ == "__main__":
    main()
