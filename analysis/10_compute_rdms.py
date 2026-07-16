#!/usr/bin/env python3
"""
Step 10: Compute cosine- and Pearson-distance RDMs for all tasks.

For each network, loads every checkpoint .npz file and stores two RDMs per
activation key in an HDF5 file:
  {key}_cosine  — cosine distance
  {key}_pearson — cosine distance on across-units mean-centered activations
                  (equivalent to 1 − Pearson correlation)

Key format in .npz:
  MLP tasks   — "layer_0", "layer_1"  (shape: N_stimuli × hidden_size)
  RNN tasks   — "layer_{L}_t_{T}"     (shape: N_stimuli × hidden_size)

Output: analysis/rdms/{task}_rdms.h5

HDF5 layout:
  meta/          — task name, n_stimuli, n_pairs
  runs/{run_id}/
    [attrs]      — iteration, is_repeat, performance, all HP values, cont_unit_vals
    {checkpoint}/
      {key}_cosine  — float32 array of length n_pairs (upper triangle, row-major)
      {key}_pearson — float32 array of length n_pairs (upper triangle, row-major)

Usage:
  python 10_compute_rdms.py                        # all tasks
  python 10_compute_rdms.py --task mnist_dual      # one task
  python 10_compute_rdms.py --overwrite            # recompute existing entries
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

ANALYSIS = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import DATASET_DIR, RDM_DIR, TASK_NAMES, task_meta

TASK_DIR_OVERRIDES = {}

# activation vectors with norm below this are considered degenerate
ZERO_NORM_THRESHOLD = 1e-8


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def upper_triangle_indices(n):
    return np.triu_indices(n, k=1)


def compute_cosine_rdm(activations: np.ndarray):
    """
    Compute cosine-distance RDM upper triangle from (N_stimuli, D) activations.
    Returns float32 array of length N_stimuli*(N_stimuli-1)//2, or None if any
    activation vector has near-zero norm (degenerate network).
    """
    if np.any(~np.isfinite(activations)):
        return None
    norms = np.linalg.norm(activations, axis=1)
    if np.any(norms < ZERO_NORM_THRESHOLD) or not np.all(np.isfinite(norms)):
        return None
    dist = cosine_distances(activations.astype(np.float32))
    n = dist.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    return dist[rows, cols].astype(np.float32)


def compute_pearson_rdm(activations: np.ndarray):
    """
    Compute Pearson-distance RDM upper triangle from (N_stimuli, D) activations.
    Pearson distance = cosine distance on across-units mean-centered activations
    (subtract per-stimulus mean across all units). Returns float32 array of
    length N_stimuli*(N_stimuli-1)//2, or None if degenerate.
    """
    if np.any(~np.isfinite(activations)):
        return None
    centered = activations - activations.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    if np.any(norms < ZERO_NORM_THRESHOLD) or not np.all(np.isfinite(norms)):
        return None
    dist = cosine_distances(centered.astype(np.float32))
    n = dist.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    return dist[rows, cols].astype(np.float32)


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def write_run_attrs(run_grp, iteration, bo_entry, run_meta):
    """Write per-run metadata as HDF5 attributes (idempotent)."""
    run_grp.attrs["iteration"] = iteration if iteration is not None else -1
    run_grp.attrs["is_repeat"] = bool(bo_entry.get("is_repeat", False))
    perf = bo_entry.get("performance")
    run_grp.attrs["performance"] = float(perf) if perf is not None else float("nan")
    run_grp.attrs["best_metric"] = float(run_meta.get("best_metric", float("nan")))

    for hp_name, hp_val in run_meta.get("config", {}).items():
        if isinstance(hp_val, (int, float, bool, str)):
            run_grp.attrs[f"hp_{hp_name}"] = hp_val
        else:
            run_grp.attrs[f"hp_{hp_name}"] = str(hp_val)

    unit_vals = bo_entry.get("cont_unit_vals")
    if unit_vals is not None:
        run_grp.attrs["cont_unit_vals"] = np.array(unit_vals, dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-task processing
# ---------------------------------------------------------------------------

def process_task(task: str, overwrite: bool):
    dirname = TASK_DIR_OVERRIDES.get(task, task)
    task_dir = DATASET_DIR / dirname

    bo_path = task_dir / "bo_state.json"
    if not bo_path.exists():
        print(f"[skip] {task}: no bo_state.json at {bo_path}")
        return

    RDM_DIR.mkdir(parents=True, exist_ok=True)
    h5_path = RDM_DIR / f"{task}_rdms.h5"
    print(f"\n{'='*60}")
    print(f"Task: {task}  ({dirname})")
    print(f"Output: {h5_path}")

    bo_state = {o["iteration"]: o for o in json.load(open(bo_path))}
    n_obs = len(bo_state)

    # Counts for progress summary
    n_runs_found = 0
    n_rdms_computed = 0
    n_rdms_skipped = 0
    n_degenerate = 0
    flagged = []

    with h5py.File(h5_path, "a") as h5:
        # Task-level metadata (write once)
        if "meta" not in h5:
            h5.create_group("meta")
        h5["meta"].attrs["task"] = task

        for iteration, bo_entry in sorted(bo_state.items()):
            run_id = f"run_{iteration:04d}_r0"
            run_dir = task_dir / run_id

            if not run_dir.exists():
                continue

            meta_path = run_dir / "metadata.json"
            if not meta_path.exists():
                continue

            n_runs_found += 1
            run_meta = json.load(open(meta_path))
            run_grp = h5.require_group(f"runs/{run_id}")

            # Write run-level attrs on first visit, or whenever overwriting
            if "iteration" not in run_grp.attrs or overwrite:
                write_run_attrs(run_grp, iteration, bo_entry, run_meta)

            ckpt_files = sorted(run_dir.glob("*.npz"))
            if not ckpt_files:
                continue

            # Record n_stimuli / n_pairs from first RDM we successfully compute
            n_pairs_written = False

            for ckpt_path in ckpt_files:
                ckpt_name = ckpt_path.stem
                ckpt_grp = run_grp.require_group(ckpt_name)

                try:
                    npz = np.load(ckpt_path)
                except Exception as e:
                    print(f"  [warn] cannot load {ckpt_path.name} for {run_id}: {e}")
                    continue

                for key in sorted(npz.keys()):
                    cosine_key = f"{key}_cosine"
                    pearson_key = f"{key}_pearson"

                    need_cosine = cosine_key not in ckpt_grp or overwrite
                    need_pearson = pearson_key not in ckpt_grp or overwrite

                    if not need_cosine and not need_pearson:
                        n_rdms_skipped += 2
                        continue

                    activations = npz[key]

                    if need_cosine:
                        rdm_c = compute_cosine_rdm(activations)
                        if cosine_key in ckpt_grp:
                            del ckpt_grp[cosine_key]
                        if rdm_c is None:
                            n_degenerate += 1
                            flagged.append(f"{run_id}/{ckpt_name}/{cosine_key}")
                            ds = ckpt_grp.create_dataset(
                                cosine_key, data=np.array([], dtype=np.float32))
                            ds.attrs["degenerate"] = True
                        else:
                            ckpt_grp.create_dataset(
                                cosine_key, data=rdm_c,
                                compression="gzip", compression_opts=4,
                                shuffle=True,
                            )
                            n_rdms_computed += 1
                            if not n_pairs_written and "n_stimuli" not in h5["meta"].attrs:
                                h5["meta"].attrs["n_stimuli"] = activations.shape[0]
                                h5["meta"].attrs["n_pairs"] = len(rdm_c)
                                n_pairs_written = True

                    if need_pearson:
                        rdm_p = compute_pearson_rdm(activations)
                        if pearson_key in ckpt_grp:
                            del ckpt_grp[pearson_key]
                        if rdm_p is None:
                            n_degenerate += 1
                            flagged.append(f"{run_id}/{ckpt_name}/{pearson_key}")
                            ds = ckpt_grp.create_dataset(
                                pearson_key, data=np.array([], dtype=np.float32))
                            ds.attrs["degenerate"] = True
                        else:
                            ckpt_grp.create_dataset(
                                pearson_key, data=rdm_p,
                                compression="gzip", compression_opts=4,
                                shuffle=True,
                            )
                            n_rdms_computed += 1

            if n_runs_found % 100 == 0:
                print(f"  {n_runs_found}/{n_obs} runs processed ...", flush=True)

        # Persist full flagged list in meta/flagged (overwrite on each run)
        if "meta/flagged" in h5:
            del h5["meta/flagged"]
        if flagged:
            h5.create_dataset(
                "meta/flagged",
                data=np.array(flagged, dtype=h5py.special_dtype(vlen=str)),
            )
        h5["meta"].attrs["n_flagged"] = len(flagged)

    print(f"  runs with activations: {n_runs_found}")
    print(f"  RDMs computed:  {n_rdms_computed}")
    print(f"  RDMs skipped (already in file): {n_rdms_skipped}")
    if n_degenerate:
        print(f"  FLAGGED (NaN or near-zero activations): {n_degenerate}")
        for tag in flagged[:20]:
            print(f"    {tag}")
        if len(flagged) > 20:
            print(f"    ... and {len(flagged)-20} more")
        print(f"  (all flagged paths stored in meta/flagged in the HDF5)")

    h5_size = h5_path.stat().st_size / 1e9
    print(f"  HDF5 file size: {h5_size:.2f} GB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute cosine-distance RDMs for all (or selected) tasks."
    )
    parser.add_argument(
        "--task", nargs="+", default=None,
        help="Task name(s) to process. Default: all tasks.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute and overwrite already-stored RDM entries.",
    )
    args = parser.parse_args()

    tasks = args.task if args.task else TASK_NAMES
    for task in tasks:
        if task not in TASK_NAMES:
            print(f"[warn] unknown task '{task}', skipping")
            continue
        process_task(task, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
