#!/usr/bin/env python3
"""
Step 1: Disk inventory.

Compares bo_state.json entries against what is physically on disk for each task.
Saves analysis/tables/disk_inventory.csv.

For 'adding': the fresh retraining run is at output/production/adding/ (in progress).
The old corrupted run is shown separately as adding_failed_run for reference.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ANALYSIS = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import (
    PRODUCTION_DIR,
    TABLES_DIR,
    TASK_EXPECTED_OBS,
    TASK_NAMES,
    disk_inventory,
    disk_inventory_all,
)


def orphaned_dirs(task_name: str, production_dir: Path = None) -> int:
    """Count run_NNNN_r0 dirs on disk not referenced in bo_state.json."""
    if production_dir is None:
        production_dir = PRODUCTION_DIR
    task_dir = Path(production_dir) / task_name
    state_path = task_dir / "bo_state.json"
    if not state_path.exists():
        return 0
    known = {o["iteration"] for o in json.load(open(state_path))}
    count = 0
    for d in task_dir.iterdir():
        if d.is_dir() and d.name.startswith("run_"):
            parts = d.name.split("_")
            try:
                if int(parts[1]) not in known:
                    count += 1
            except (IndexError, ValueError):
                pass
    return count


def summarise(inv: pd.DataFrame, task_name: str, production_dir: Path = None) -> dict:
    t = inv[inv["task"] == task_name]
    if t.empty:
        return None
    n_in_state   = len(t)
    n_primary    = int((~t["is_repeat"]).sum())
    n_repeat     = int(t["is_repeat"].sum())
    n_run_dirs   = int(t["has_run_dir"].sum())
    n_with_act   = int(t["has_activations"].sum())
    n_missing    = n_run_dirs - n_with_act
    n_orphaned   = orphaned_dirs(task_name, production_dir)
    return {
        "task":                  task_name,
        "n_in_state":            n_in_state,
        "n_primary":             n_primary,
        "n_repeat":              n_repeat,
        "n_run_dirs":            n_run_dirs,
        "n_with_activations":    n_with_act,
        "n_missing_activations": n_missing,
        "n_orphaned":            n_orphaned,
    }


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Primary inventory (production/) ----------------------------------
    print("Loading disk inventory from output/production/ ...")
    inv = disk_inventory_all()

    rows = []
    for task in TASK_NAMES:
        row = summarise(inv, task)
        if row is None:
            print(f"  [skip] {task}: no bo_state.json")
            continue
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "disk_inventory.csv", index=False)

    print("\n=== Disk inventory — output/production/ ===\n")
    print(df.to_string(index=False))

    # --- Flags ------------------------------------------------------------
    flags = []
    for _, row in df.iterrows():
        expected = TASK_EXPECTED_OBS.get(row["task"], 1000)
        if row["n_missing_activations"] > 0:
            flags.append(f"  {row['task']}: {int(row['n_missing_activations'])} run dirs missing activations")
        if row["n_orphaned"] > 0:
            flags.append(f"  {row['task']}: {int(row['n_orphaned'])} orphaned run dirs (on disk, not in state)")
        if row["n_in_state"] < expected:
            flags.append(f"  {row['task']}: {int(row['n_in_state'])}/{expected} observations (in progress)")
    if flags:
        print("\n[!] Issues / in-progress:")
        for f in flags:
            print(f)

    # --- adding_failed_run reference -------------------------------------
    failed_dir = PRODUCTION_DIR.parent / "production" / ".."  # relative anchor
    failed_run = PRODUCTION_DIR / ".." / "production" / "adding_failed_run"
    failed_run = (PRODUCTION_DIR.parent / "production" / "adding_failed_run").resolve()
    # Reuse disk_inventory but point at adding_failed_run as if it were the task dir
    af_state = failed_run / "bo_state.json"
    if af_state.exists():
        # Temporarily inject as a custom production_dir trick:
        # disk_inventory looks for production_dir / task_name / bo_state.json
        # so we pass production_dir = adding_failed_run.parent and task_name = "adding_failed_run"
        af_inv = disk_inventory("adding_failed_run", production_dir=failed_run.parent)
        af_row = summarise(af_inv, "adding_failed_run", production_dir=failed_run.parent)
        if af_row:
            print("\n=== adding_failed_run (reference — do not use for analysis) ===\n")
            af_df = pd.DataFrame([af_row])
            print(af_df.to_string(index=False))
            # Highlight the known-missing iterations
            missing_iters = af_inv[~af_inv["has_run_dir"]]["iteration"].tolist()
            if missing_iters:
                print(f"\n  Missing run dirs (iterations): {missing_iters}")

    print(f"\nSaved: {TABLES_DIR / 'disk_inventory.csv'}")


if __name__ == "__main__":
    main()
