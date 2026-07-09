#!/usr/bin/env python3
"""
Step 3: Master summary table.

One row per task with counts, repeat rate, activation coverage, and
network breakdown into the three performance categories (alpha=0.9):
  successful       — performance >= upper threshold
  better_than_chance — between lower and upper
  near_chance      — performance <= lower threshold

Output:
  analysis/tables/task_summary.csv  (printed and saved)
"""

import json
import sys
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
    rows = []
    for obs in json.load(open(state_path)):
        row = {
            "task":        "adding",
            "paradigm":    meta["paradigm"],
            "iteration":   obs["iteration"],
            "is_repeat":   obs.get("is_repeat", False),
            "performance": obs["performance"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    meta       = task_meta()
    thresholds = json.load(open(TABLES_DIR / "success_thresholds.json"))
    inv        = pd.read_csv(TABLES_DIR / "disk_inventory.csv")

    plot_tasks = [t for t in TASK_NAMES if t != "adding"] + ["adding"]

    rows = []
    for task in plot_tasks:
        if task == "adding":
            p = PRODUCTION_DIR / "adding_failed_run" / "bo_state.json"
            if not p.exists():
                print(f"  [skip] adding: no adding_failed_run data")
                continue
            df = load_adding_failed()
        else:
            p = PRODUCTION_DIR / task / "bo_state.json"
            if not p.exists():
                print(f"  [skip] {task}: no bo_state.json")
                continue
            df = load_task_df(task)

        prim = primary_df(df)
        perf = prim["performance"].values

        thresh = thresholds.get(task, {})
        upper  = thresh.get("upper")
        lower  = thresh.get("lower")

        n_successful         = int((perf >= upper).sum()) if upper is not None else None
        n_near_chance        = int((perf <= lower).sum()) if lower is not None else None
        n_better_than_chance = (len(perf) - n_successful - n_near_chance
                                if upper is not None else None)

        # adding data comes from adding_failed_run
        inv_key = "adding_failed_run" if task == "adding" else task
        inv_row = inv[inv["task"] == inv_key]
        n_with_act = int(inv_row["n_with_activations"].values[0]) if len(inv_row) else None

        m = meta[task]
        rows.append({
            "task":                  task,
            "paradigm":              m["paradigm"],
            "n_networks":            len(df),
            "n_primary":             len(prim),
            "n_repeats":             int(df["is_repeat"].sum()),
            "repeat_rate":           round(df["is_repeat"].mean(), 3),
            "n_with_activations":    n_with_act,
            "chance_perf":           m["chance_perf"],
            "upper_threshold":       round(upper, 4) if upper is not None else None,
            "lower_threshold":       round(lower, 4) if lower is not None else None,
            "n_successful":    n_successful,
            "n_partial":       n_better_than_chance,
            "n_near_chance":   n_near_chance,
            "pct_successful":  round(100 * n_successful / len(perf), 1) if upper is not None else None,
            "pct_partial":     round(100 * n_better_than_chance / len(perf), 1) if upper is not None else None,
            "pct_near_chance": round(100 * n_near_chance / len(perf), 1) if lower is not None else None,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(TABLES_DIR / "task_summary.csv", index=False)

    # Print with readable formatting
    print_cols = [
        "task", "paradigm", "n_primary", "n_with_activations",
        "chance_perf", "upper_threshold", "lower_threshold",
        "n_successful", "n_partial", "n_near_chance",
        "pct_successful", "pct_partial", "pct_near_chance",
    ]
    print(df_out[print_cols].to_string(index=False))
    print(f"\nSaved: {TABLES_DIR / 'task_summary.csv'}")


if __name__ == "__main__":
    main()
