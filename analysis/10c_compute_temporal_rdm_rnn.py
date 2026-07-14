#!/usr/bin/env python3
"""
Step 10c: Full temporal RDM for mnist_rnn.

Produces a single 1400×1400 RDM per network: rows/cols are all
(stimulus, timestep) pairs (100 stimuli × 14 timesteps), ordered
stimulus-major: (stim_0, t=0), ..., (stim_0, t=13), (stim_1, t=0), ...

Entry [(stim_i, t_a), (stim_j, t_b)] = cosine distance between h_i[t_a]
and h_j[t_b], computed as 1 - M @ M.T where M is the (1400, H) matrix of
unit-normalised hidden states.

WARNING — cosine distance only: this does NOT generalise to other metrics
(cf. reference_code_rdm.py:average_values for the general approach).

No NaN entries: every stimulus has a valid activation at every timestep.
Uses the deepest hidden layer (same convention as other per-timestep analyses).

Stored as key "temporal" in each run/checkpoint group of mnist_rnn_rdms.h5.

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


def build_temporal_matrix(acts_by_t):
    """
    Build (N_ROWS, H) = (1400, H) matrix of unit-normalised hidden states.

    Row ordering: stimulus-major — all timesteps for stim_0, then stim_1, ...
    Row i*T + t = unit(h_i[t]).

    WARNING — cosine distance only: RDM = 1 - M @ M.T.
    """
    # acts_by_t[t] is (N_STIM, H); stack to (T, N_STIM, H), then stim-major reshape
    acts = np.stack([acts_by_t[t] for t in range(T)], axis=0)  # (T, N_STIM, H)
    acts = acts.transpose(1, 0, 2).reshape(N_ROWS, -1).astype(np.float64)  # (N_ROWS, H)
    return _unit_norm(acts).astype(np.float32)


def compute_temporal_rdm(M):
    """
    Upper-triangle of 1 - M @ M.T.
    Returns None if any activation is non-finite (degenerate network).
    """
    if not np.all(np.isfinite(M)):
        return None
    Md   = M.astype(np.float64)
    D    = np.clip(1.0 - Md @ Md.T, 0.0, 2.0)
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

                if "temporal" in ckpt_grp and not args.overwrite:
                    n_skipped += 1
                    continue

                try:
                    npz = np.load(ckpt_path)
                except Exception as e:
                    print(f"  [warn] cannot load {run_id}/{ckpt_name}: {e}")
                    continue

                # Deepest layer
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

                M   = build_temporal_matrix(acts_by_t)
                rdm = compute_temporal_rdm(M)

                if "temporal" in ckpt_grp:
                    del ckpt_grp["temporal"]

                if rdm is None:
                    ds = ckpt_grp.create_dataset(
                        "temporal", data=np.array([], dtype=np.float32))
                    ds.attrs["degenerate"] = True
                    n_degen += 1
                    continue

                ckpt_grp.create_dataset(
                    "temporal", data=rdm,
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
