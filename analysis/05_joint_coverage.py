#!/usr/bin/env python3
"""
Step 5: Joint coverage using 2-bin hypercubes.

Each primary network is assigned to a hypercube cell:
  - continuous HPs: bin 0 if unit value < 0.5, bin 1 if >= 0.5
  - categorical HPs: the value itself (each level is its own bin)

Total cells = 2^n_cont * n_cat_combos
  supervised: 2^5 * 24 = 768
  rnn:        2^5 * 16 = 512
  rl:         2^4 * 24 = 384

Output:
  analysis/tables/joint_coverage.csv  (printed and saved)
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import (
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
    """Compute a hashable hypercube key for one observation."""
    cont_bins = tuple(0 if row[f"unit_{n}"] < 0.5 else 1 for n in cont_names)
    cat_vals  = tuple(row[n] for n in cat_names)
    return cont_bins + cat_vals


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    meta       = task_meta()
    thresholds = json.load(open(TABLES_DIR / "success_thresholds.json"))

    plot_tasks = [t for t in TASK_NAMES if t != "adding"] + ["adding"]

    rows = []
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
        n_cont     = len(cont_names)
        n_cat_combos = m["n_cat_combos"]
        n_total    = (2 ** n_cont) * n_cat_combos

        # Count networks per cell
        cell_counts = Counter()
        for _, obs in prim.iterrows():
            key = cell_key(obs, cont_names, cat_names)
            cell_counts[key] += 1

        # Track which cells have ≥1 successful network
        succ_cells = set()
        for _, obs in succ.iterrows():
            succ_cells.add(cell_key(obs, cont_names, cat_names))

        n_occupied    = len(cell_counts)
        n_with_success = len(succ_cells)
        counts         = list(cell_counts.values())
        mean_per_occ   = np.mean(counts) if counts else 0
        max_per_cell   = max(counts) if counts else 0

        rows.append({
            "task":              task,
            "paradigm":          m["paradigm"],
            "n_cont_dims":       n_cont,
            "n_cat_combos":      n_cat_combos,
            "n_hypercubes_total": n_total,
            "n_occupied":        n_occupied,
            "pct_occupied":      round(100 * n_occupied / n_total, 1),
            "n_with_success":    n_with_success,
            "pct_with_success":  round(100 * n_with_success / n_total, 1),
            "mean_per_occupied": round(mean_per_occ, 2),
            "max_per_cell":      max_per_cell,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(TABLES_DIR / "joint_coverage.csv", index=False)
    print(df_out.to_string(index=False))
    print(f"\nSaved: {TABLES_DIR / 'joint_coverage.csv'}")


if __name__ == "__main__":
    main()
