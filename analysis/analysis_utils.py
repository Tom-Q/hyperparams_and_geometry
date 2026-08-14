"""
Shared utilities for all analysis scripts.

Provides:
  task_meta()          — per-task metadata (paradigm, HP names, chance perf, etc.)
  load_task_df()       — bo_state.json → flat DataFrame for one task
  load_all_tasks()     — all tasks concatenated, with parquet cache
  disk_inventory()     — per-iteration disk check (run dir, activations present)
  primary_df()         — filter DataFrame to non-repeat observations
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT   = Path(__file__).parent.parent
ANALYSIS    = Path(__file__).parent
OUTPUT_DIR  = REPO_ROOT / "output"

# Final training dataset (copy of production, frozen for analysis)
DATASET_DIR = OUTPUT_DIR / "final_dataset"

# Analysis outputs
ANALYSIS_OUT = OUTPUT_DIR / "analysis"
FIGURES_DIR  = ANALYSIS_OUT / "figures"
TABLES_DIR   = ANALYSIS_OUT / "tables"
CACHE_DIR    = ANALYSIS_OUT / "cache"
RDM_DIR      = ANALYSIS_OUT / "rdms"

# Backwards-compat alias — scripts still being migrated may import this
PRODUCTION_DIR = DATASET_DIR

sys.path.insert(0, str(REPO_ROOT))

TASK_NAMES = [
    "mnist_dual", "mnist_10way", "fashion_10way", "spirals", "parity",
    "adding", "mnist_rnn",
    "cartpole", "fourrooms",
    "cifar10",
    "qbert",
]

# Expected total observations per task (used for inventory completeness checks)
TASK_EXPECTED_OBS = {
    "mnist_dual":    1000,
    "mnist_10way":   1000,
    "fashion_10way": 1000,
    "spirals":       1000,
    "parity":        1000,
    "adding":        1000,
    "mnist_rnn":      200,
    "cartpole":      1000,
    "fourrooms":     1000,
    "cifar10":       1000,
    "qbert":           32,
}

# RL tasks save final.npz (training ends at best performance); others save best.npz
RL_TASKS = {"cartpole", "fourrooms", "qbert"}

# Tasks where "successful" is determined by an HDF5 attribute rather than a
# performance score threshold (Q*bert uses frac_level5 >= 0.5).
ATTR_SUCCESS_TASKS = {"qbert"}


# ---------------------------------------------------------------------------
# Depth helpers
# ---------------------------------------------------------------------------

def get_depth(rg) -> int:
    """Return depth (number of hidden FC layers) from an HDF5 run group.

    All tasks store this as hp_depth. The HDF5 writer normalises CIFAR's
    n_fc_layers → depth at write time, but this fallback handles any HDF5
    written before that normalisation was in place.
    """
    return int(rg.attrs.get("hp_depth", rg.attrs.get("hp_n_fc_layers", 1)))


def get_depth_from_config(config: dict) -> int:
    """Return depth from a metadata.json config dict.

    Other tasks use key 'depth'; CIFAR uses 'n_fc_layers'.
    metadata.json is never rewritten, so we must check both.
    """
    d = config.get("depth") if config.get("depth") is not None else config.get("n_fc_layers")
    return int(d) if d is not None else 1


# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def task_meta() -> dict:
    """
    Return a dict keyed by task name with:
      paradigm, chance_perf, max_metric, success_threshold, metric_name,
      cont_param_names, cat_param_names, cat_param_choices, n_cat_combos
    """
    from tasks import TASKS
    from src.bo import _cont_params_for_task, cat_params_for_task

    meta = {}
    for name in TASK_NAMES:
        task = TASKS[name]()
        cont = _cont_params_for_task(task)
        cat  = cat_params_for_task(task)
        n_cat_combos = 1
        for _, choices in cat:
            n_cat_combos *= len(choices)
        meta[name] = {
            "paradigm":         task.paradigm,
            "chance_perf":      task.chance_perf,
            "max_metric":       task.max_metric,
            "success_threshold": task.success_threshold,
            "metric_name":      task.metric_name,
            "cont_param_names": [p[0] for p in cont],
            "cat_param_names":  [p[0] for p in cat],
            "cat_param_choices": {p[0]: p[1] for p in cat},
            "n_cat_combos":     n_cat_combos,
        }
    return meta


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_task_df(task_name: str, production_dir: Path = None) -> pd.DataFrame:
    """
    Load bo_state.json for one task into a flat DataFrame.

    Columns:
      task, paradigm, iteration, is_repeat, repeat_of, performance
      <all raw HP values from config>
      unit_<name> for each continuous HP (log-normalized [0,1] from cont_unit_vals)
    """
    if production_dir is None:
        production_dir = PRODUCTION_DIR

    state_path = Path(production_dir) / task_name / "bo_state.json"
    # Q*bert lives in production until copied to final_dataset
    if not state_path.exists() and task_name == "qbert":
        state_path = OUTPUT_DIR / "production" / "qbert" / "bo_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"No bo_state.json for {task_name} at {state_path}")

    observations = json.load(open(state_path))
    meta         = task_meta()[task_name]
    cont_names   = meta["cont_param_names"]

    rows = []
    for obs in observations:
        # Q*bert bo_state entries don't have is_repeat / repeat_of / cont_unit_vals
        row = {
            "task":        task_name,
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


def load_all_tasks(production_dir: Path = None, use_cache: bool = True) -> pd.DataFrame:
    """
    Load all tasks into one concatenated DataFrame.

    Caches to analysis/cache/all_observations.parquet on first call.
    Pass use_cache=False to force a full reload (e.g. after adding new runs).
    """
    if production_dir is None:
        production_dir = PRODUCTION_DIR

    cache_path = CACHE_DIR / "all_observations.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    dfs = []
    for name in TASK_NAMES:
        state_path = Path(production_dir) / name / "bo_state.json"
        if not state_path.exists():
            print(f"  [skip] {name}: no bo_state.json")
            continue
        df = load_task_df(name, production_dir)
        dfs.append(df)
        print(f"  loaded {name}: {len(df)} obs")

    combined = pd.concat(dfs, ignore_index=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path, index=False)
    print(f"Cached {len(combined)} total observations → {cache_path}")

    return combined


# ---------------------------------------------------------------------------
# Disk inventory
# ---------------------------------------------------------------------------

def disk_inventory(task_name: str, production_dir: Path = None) -> pd.DataFrame:
    """
    For each iteration in bo_state.json, check what is present on disk.

    Returns a DataFrame with columns:
      task, iteration, is_repeat, performance,
      has_run_dir, has_activations (best.npz present)
    """
    if production_dir is None:
        production_dir = PRODUCTION_DIR

    task_dir   = task_run_dir(task_name)
    state_path = task_dir / "bo_state.json"
    observations = json.load(open(state_path))

    rows = []
    for obs in observations:
        it      = obs["iteration"]
        run_dir = task_dir / f"run_{it:04d}_r0"
        has_dir = run_dir.exists()
        act_file = "final.npz" if task_name in RL_TASKS else "best.npz"
        has_act = (run_dir / act_file).exists() if has_dir else False
        rows.append({
            "task":             task_name,
            "iteration":        it,
            "is_repeat":        obs.get("is_repeat", False),
            "performance":      obs["performance"],
            "has_run_dir":      has_dir,
            "has_activations":  has_act,
        })

    return pd.DataFrame(rows)


def disk_inventory_all(production_dir: Path = None) -> pd.DataFrame:
    """Run disk_inventory for all tasks and concatenate."""
    if production_dir is None:
        production_dir = PRODUCTION_DIR
    dfs = []
    for name in TASK_NAMES:
        state_path = Path(production_dir) / name / "bo_state.json"
        if not state_path.exists():
            continue
        dfs.append(disk_inventory(name, production_dir))
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Convenience filters
# ---------------------------------------------------------------------------

def metric_output_dirs(metric: str) -> tuple[Path, Path]:
    """Return (figures_dir, tables_dir) for a specific RDM metric."""
    base = ANALYSIS_OUT / metric
    return base / "figures", base / "tables"


def task_run_dir(task: str) -> Path:
    """Return the directory containing run_XXXX_r0/ subdirectories for a task.

    Checks final_dataset first (the canonical location). For Q*bert, which has
    not yet been copied there, falls back to output/production/qbert.
    """
    d = DATASET_DIR / task
    if d.exists():
        return d
    if task == "qbert":
        return OUTPUT_DIR / "production" / "qbert"
    return d


def is_run_successful(task: str, rg, thresholds: dict) -> bool:
    """Return True if this run group should be counted as successful/functional.

    Q*bert uses the is_functional HDF5 attribute (frac_level5 >= 0.5).
    All other tasks compare performance against the per-task score threshold.
    """
    is_func = rg.attrs.get("is_functional", None)
    if is_func is not None:
        return bool(is_func)
    perf = float(rg.attrs.get("performance", float("nan")))
    threshold = thresholds.get(task, -float("inf"))
    return np.isfinite(perf) and perf >= threshold


def primary_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return only primary (non-repeat) observations."""
    return df[~df["is_repeat"]].copy()


def successful_df(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    Return primary observations above the per-task success threshold.
    thresholds: dict mapping task_name → float threshold value.
    """
    prim = primary_df(df)
    mask = prim.apply(lambda row: row["performance"] >= thresholds.get(row["task"], np.inf), axis=1)
    return prim[mask].copy()
