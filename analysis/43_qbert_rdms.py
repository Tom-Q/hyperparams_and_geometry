#!/usr/bin/env python3
"""
Compute cosine- and Pearson-distance RDMs for Q*bert analysis networks.

Reads activation npz files from output/production/qbert/run_XXXX_r0/ and
produces an HDF5 with the same layout as the other task RDM files, so that
downstream analysis scripts can treat Q*bert like any other task.

Source networks (run_model_r1, run_0000_r0, run_0003_r0, run_0004_r0) are
excluded. Only layer_0 and layer_1 activation arrays are processed.

Per-run attributes include is_functional (frac_level5 >= 0.5 at any eval).

Output: output/analysis/rdms/qbert_rdms.h5

Usage:
    python analysis/43_qbert_rdms.py
    python analysis/43_qbert_rdms.py --overwrite
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

ANALYSIS  = Path(__file__).parent
REPO_ROOT = ANALYSIS.parent
sys.path.insert(0, str(ANALYSIS))

from analysis_utils import RDM_DIR

QBERT_DIR   = REPO_ROOT / "output" / "production" / "qbert"
H5_PATH     = RDM_DIR / "qbert_rdms.h5"

# Source networks used for stimulus extraction — excluded from analysis
SOURCE_RUNS = {"run_model_r1", "run_0000_r0", "run_0003_r0", "run_0004_r0"}

FUNCTIONAL_THRESHOLD = 0.5   # min frac_level5 at any eval to be considered functional
ZERO_NORM_THRESHOLD  = 1e-8
ACTIVATION_KEYS      = {"layer_0", "layer_1"}


# ── RDM computation (identical to 10_compute_rdms.py) ─────────────────────────

def compute_cosine_rdm(activations: np.ndarray):
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


# ── Q*bert-specific helpers ────────────────────────────────────────────────────

def max_frac_level5(run_dir: Path) -> float:
    log = run_dir / "training_log.csv"
    if not log.exists():
        return 0.0
    rows = list(csv.DictReader(open(log)))
    if not rows:
        return 0.0
    return max(float(r["frac_level5"]) for r in rows)


def write_run_attrs(run_grp, iteration, bo_entry, run_meta, frac_l5):
    run_grp.attrs["iteration"]       = iteration
    run_grp.attrs["is_repeat"]       = False
    run_grp.attrs["performance"]     = float(bo_entry.get("performance", float("nan")))
    run_grp.attrs["best_metric"]     = float(run_meta.get("best_metric", float("nan")))
    run_grp.attrs["max_frac_level5"] = float(frac_l5)
    run_grp.attrs["is_functional"]   = bool(frac_l5 >= FUNCTIONAL_THRESHOLD)
    run_grp.attrs["stop_reason"]     = run_meta.get("stop_reason", "")
    run_grp.attrs["max_level_ever"]  = int(run_meta.get("max_level_ever", 0))

    for hp_name, hp_val in run_meta.get("config", {}).items():
        if isinstance(hp_val, (int, float, bool, str)):
            run_grp.attrs[f"hp_{hp_name}"] = hp_val
        else:
            run_grp.attrs[f"hp_{hp_name}"] = str(hp_val)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute already-stored RDM entries.")
    args = parser.parse_args()

    RDM_DIR.mkdir(parents=True, exist_ok=True)

    bo_state = {e["iteration"]: e
                for e in json.load(open(QBERT_DIR / "bo_state.json"))}
    print(f"Q*bert: {len(bo_state)} entries in bo_state.json")

    n_runs_found = n_rdms_computed = n_rdms_skipped = n_degenerate = 0
    flagged = []

    with h5py.File(H5_PATH, "a") as h5:
        if "meta" not in h5:
            h5.create_group("meta")
        h5["meta"].attrs["task"] = "qbert"

        for iteration, bo_entry in sorted(bo_state.items()):
            run_id  = f"run_{iteration:04d}_r0"
            run_dir = QBERT_DIR / run_id

            if run_id in SOURCE_RUNS:
                continue
            if not run_dir.exists():
                continue

            meta_path = run_dir / "metadata.json"
            if not meta_path.exists():
                continue

            n_runs_found += 1
            run_meta = json.load(open(meta_path))
            frac_l5  = max_frac_level5(run_dir)
            run_grp  = h5.require_group(f"runs/{run_id}")

            if "iteration" not in run_grp.attrs or args.overwrite:
                write_run_attrs(run_grp, iteration, bo_entry, run_meta, frac_l5)

            functional_tag = "FUNCTIONAL" if frac_l5 >= FUNCTIONAL_THRESHOLD else "non-functional"
            print(f"  {run_id}  perf={bo_entry.get('performance', '?'):.0f}  "
                  f"frac_l5={frac_l5:.2f}  {functional_tag}")

            ckpt_files = sorted(run_dir.glob("*.npz"))
            n_stimuli_written = "n_stimuli" in h5["meta"].attrs

            for ckpt_path in ckpt_files:
                ckpt_name = ckpt_path.stem
                ckpt_grp  = run_grp.require_group(ckpt_name)

                try:
                    npz = np.load(ckpt_path)
                except Exception as e:
                    print(f"    [warn] cannot load {ckpt_path.name}: {e}")
                    continue

                for key in sorted(ACTIVATION_KEYS & set(npz.keys())):
                    cosine_key  = f"{key}_cosine"
                    pearson_key = f"{key}_pearson"

                    need_cosine  = cosine_key  not in ckpt_grp or args.overwrite
                    need_pearson = pearson_key not in ckpt_grp or args.overwrite

                    if not need_cosine and not need_pearson:
                        n_rdms_skipped += 2
                        continue

                    acts = npz[key]
                    assert acts.ndim == 2, \
                        f"Expected 2D activation array, got shape {acts.shape} for {key}"

                    for metric, compute_fn, k in [
                        ("cosine",  compute_cosine_rdm,  cosine_key),
                        ("pearson", compute_pearson_rdm, pearson_key),
                    ]:
                        if (metric == "cosine" and not need_cosine) or \
                           (metric == "pearson" and not need_pearson):
                            continue
                        rdm = compute_fn(acts)
                        if k in ckpt_grp:
                            del ckpt_grp[k]
                        if rdm is None:
                            n_degenerate += 1
                            flagged.append(f"{run_id}/{ckpt_name}/{k}")
                            ds = ckpt_grp.create_dataset(k, data=np.array([], dtype=np.float32))
                            ds.attrs["degenerate"] = True
                        else:
                            ckpt_grp.create_dataset(k, data=rdm,
                                                    compression="gzip", compression_opts=4,
                                                    shuffle=True)
                            n_rdms_computed += 1
                            if not n_stimuli_written:
                                h5["meta"].attrs["n_stimuli"] = acts.shape[0]
                                h5["meta"].attrs["n_pairs"]   = len(rdm)
                                n_stimuli_written = True

        if "meta/flagged" in h5:
            del h5["meta/flagged"]
        if flagged:
            h5.create_dataset("meta/flagged",
                              data=np.array(flagged, dtype=h5py.special_dtype(vlen=str)))
        h5["meta"].attrs["n_flagged"] = len(flagged)

    print(f"\nRuns processed:   {n_runs_found}")
    print(f"RDMs computed:    {n_rdms_computed}")
    print(f"RDMs skipped:     {n_rdms_skipped}")
    if n_degenerate:
        print(f"Degenerate:       {n_degenerate}")
        for tag in flagged[:10]:
            print(f"  {tag}")
    print(f"Output: {H5_PATH}  ({H5_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
