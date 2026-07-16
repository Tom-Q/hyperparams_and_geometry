#!/usr/bin/env python3
"""
Step 10c: Full temporal RDMs (cosine + Pearson) for mnist_rnn.

Produces two 1400×1400 RDMs per network stored as "temporal_cosine" and
"temporal_pearson". Rows/cols are (stimulus, timestep) pairs: 100 stimuli ×
14 timesteps, ordered stimulus-major.

temporal_cosine: 1 - M @ M.T where M = (1400, H) matrix of unit-normalised
  hidden states (mean unit vector trick — valid for cosine only).

temporal_pearson: same but hidden states are first mean-centered across units,
  then unit-normalized, giving Pearson correlation distance.

No NaN entries: every stimulus has a valid activation at every timestep.
Uses the deepest hidden layer (same convention as per-timestep analyses).

Usage:
  python 10c_compute_temporal_rdm_rnn.py
  python 10c_compute_temporal_rdm_rnn.py --overwrite
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ANALYSIS    = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import DATASET_DIR, RDM_DIR

TASK        = "mnist_rnn"
TASK_DIR    = DATASET_DIR / TASK
TASK_H5     = RDM_DIR / f"{TASK}_rdms.h5"
N_STIM      = 100
T           = 14
N_ROWS      = N_STIM * T                   # 1400
N_PAIRS     = N_ROWS * (N_ROWS - 1) // 2  # 979,300


def _unit_norm(mat):
    """Row-normalise to unit vectors; near-zero rows stay near-zero."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.where(norms > 1e-8, norms, 1.0)


def build_temporal_matrices(acts_by_t):
    """
    Build two (N_ROWS, H) = (1400, H) matrices:
      M_cos:  unit-normalised hidden states (for cosine RDM)
      M_raw:  raw hidden states (for Pearson RDM)
    Row ordering: stimulus-major — all timesteps for stim_0, then stim_1, ...
    """
    acts = np.stack([acts_by_t[t] for t in range(T)], axis=0)  # (T, N_STIM, H)
    acts = acts.transpose(1, 0, 2).reshape(N_ROWS, -1).astype(np.float64)  # (N_ROWS, H)
    return _unit_norm(acts).astype(np.float32), acts.astype(np.float32)


def compute_temporal_cosine_rdm(M_cos):
    """Upper-triangle of 1 - M_cos @ M_cos.T. Returns None if non-finite."""
    if not np.all(np.isfinite(M_cos)):
        return None
    Md = M_cos.astype(np.float64)
    D  = np.clip(1.0 - Md @ Md.T, 0.0, 2.0)
    rows, cols = np.triu_indices(N_ROWS, k=1)
    return D[rows, cols].astype(np.float32)


def compute_temporal_pearson_rdm(M_raw):
    """
    Pearson-distance temporal RDM. Mean-centers each row across units,
    unit-normalizes, then 1 - M @ M.T. Returns None if non-finite or
    any row is constant (degenerate).
    """
    if not np.all(np.isfinite(M_raw)):
        return None
    Md = M_raw.astype(np.float64)
    Md -= Md.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Md, axis=1)
    if np.any(norms < 1e-8) or not np.all(np.isfinite(norms)):
        return None
    Md /= norms[:, None]
    D   = np.clip(1.0 - Md @ Md.T, 0.0, 2.0)
    rows, cols = np.triu_indices(N_ROWS, k=1)
    return D[rows, cols].astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not TASK_H5.exists():
        raise FileNotFoundError(
            f"{TASK}_rdms.h5 not found at {TASK_H5}. Run 10_compute_rdms.py first.")

    print(f"Temporal RDM for {TASK}: {N_ROWS} rows ({N_STIM} stimuli × {T} timesteps)")
    print(f"  Upper triangle: {N_PAIRS:,} pairs  (~{N_PAIRS * 4 / 1e6:.1f} MB per network uncompressed)")

    with h5py.File(TASK_H5, "a") as h5:
        meta = h5.require_group("meta")

        # Row metadata — fixed across all networks, written once
        for key in ("temporal_row_stim", "temporal_row_t"):
            if key in meta:
                del meta[key]
        meta.create_dataset("temporal_row_stim",
                            data=np.repeat(np.arange(N_STIM, dtype=np.int32), T))
        meta.create_dataset("temporal_row_t",
                            data=np.tile(np.arange(T, dtype=np.int32), N_STIM))
        meta.attrs["temporal_n_rows"]  = N_ROWS
        meta.attrs["temporal_n_pairs"] = N_PAIRS

        runs_grp = h5.get("runs")
        if runs_grp is None:
            raise RuntimeError(f"No 'runs' group in {TASK_H5}.")

        run_ids    = sorted(runs_grp.keys())
        n_total    = len(run_ids)
        n_computed = 0
        n_skipped  = 0
        n_degen    = 0

        for idx, run_id in enumerate(run_ids):
            run_grp    = runs_grp[run_id]
            run_dir    = TASK_DIR / run_id
            ckpt_files = sorted(run_dir.glob("*.npz")) if run_dir.exists() else []

            for ckpt_path in ckpt_files:
                ckpt_name = ckpt_path.stem
                ckpt_grp  = run_grp.require_group(ckpt_name)

                need_cosine  = "temporal_cosine"  not in ckpt_grp or args.overwrite
                need_pearson = "temporal_pearson" not in ckpt_grp or args.overwrite

                if not need_cosine and not need_pearson:
                    n_skipped += 2
                    continue

                try:
                    npz = np.load(ckpt_path)
                except Exception as e:
                    print(f"  [warn] cannot load {run_id}/{ckpt_name}: {e}")
                    continue

                layers = sorted(set(
                    int(k.split("_t_")[0].split("layer_")[1])
                    for k in npz.keys() if "_t_" in k
                ))
                if not layers:
                    continue
                L = layers[-1]

                acts_by_t = {}
                missing = False
                for t in range(T):
                    key = f"layer_{L}_t_{t}"
                    if key not in npz:
                        missing = True
                        break
                    acts_by_t[t] = npz[key].astype(np.float32)
                if missing:
                    continue

                M_cos, M_raw = build_temporal_matrices(acts_by_t)

                if need_cosine:
                    rdm_c = compute_temporal_cosine_rdm(M_cos)
                    if "temporal_cosine" in ckpt_grp:
                        del ckpt_grp["temporal_cosine"]
                    if rdm_c is None:
                        ds = ckpt_grp.create_dataset(
                            "temporal_cosine", data=np.array([], dtype=np.float32))
                        ds.attrs["degenerate"] = True
                        n_degen += 1
                    else:
                        ckpt_grp.create_dataset(
                            "temporal_cosine", data=rdm_c,
                            compression="gzip", compression_opts=4, shuffle=True)
                        n_computed += 1

                if need_pearson:
                    rdm_p = compute_temporal_pearson_rdm(M_raw)
                    if "temporal_pearson" in ckpt_grp:
                        del ckpt_grp["temporal_pearson"]
                    if rdm_p is None:
                        ds = ckpt_grp.create_dataset(
                            "temporal_pearson", data=np.array([], dtype=np.float32))
                        ds.attrs["degenerate"] = True
                        n_degen += 1
                    else:
                        ckpt_grp.create_dataset(
                            "temporal_pearson", data=rdm_p,
                            compression="gzip", compression_opts=4, shuffle=True)
                        n_computed += 1

            if (idx + 1) % 100 == 0:
                print(f"  {idx + 1}/{n_total} runs processed ...", flush=True)

    print(f"\n  {n_computed} temporal RDMs computed")
    print(f"  {n_skipped} skipped (already present)")
    if n_degen:
        print(f"  {n_degen} degenerate")
    print(f"  Stored in: {TASK_H5}")


if __name__ == "__main__":
    main()
