#!/usr/bin/env python3
"""
Step 10b: Phase-averaged temporal RDM for the Adding task.

Produces a single RDM per network where each row/col is a (stimulus, phase)
pair: 100 stimuli × 6 phases = 600 rows, ordered stimulus-major:
(stim_0, phase_0), ..., (stim_0, phase_5), (stim_1, phase_0), ...

Entry for pair [(stim_i, phase_p), (stim_j, phase_q)] = average cosine
distance over all cross-timestep pairs (t_a in V_i[p], t_b in V_j[q]):

    avg_dist = 1 - m_i[p] · m_j[q]

where m_i[p] = mean of unit(h_i[t]) over valid timesteps t in phase p.
The full RDM is 1 - M @ M.T for the (600, H) matrix M of mean unit vectors.

WARNING — cosine distance only: this factoring does NOT hold for Euclidean,
Mahalanobis, or any other metric. For a different metric, build the full
(N_STIM×T)×(N_STIM×T) pairwise distance matrix and average within groups
explicitly (cf. reference_code_rdm.py:average_values).

Phases (0-indexed):
  0  before flag1       t < flag1         NaN where flag1 == 0
  1  at flag1           t = flag1
  2  between flags      flag1 < t < flag2 NaN where flags adjacent
  3  at flag2           t = flag2
  4  after flag2        flag2 < t < T-1   NaN where flag2 >= T-2
  5  final step         t = T-1

Rows where a stimulus has no valid timesteps in a phase are NaN. The NaN
mask is fixed across all networks (depends only on the fixed stimulus set)
and stored in meta/temporal_valid_mask. Use nan_policy='omit' downstream
for adding only; NaN in any other task indicates a bug.

Stored as key "temporal" in each run/checkpoint group of adding_rdms.h5.
Row metadata stored once in meta/.

Usage:
  python 10b_compute_adding_phases.py
  python 10b_compute_adding_phases.py --overwrite
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ANALYSIS    = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import DATASET_DIR, RDM_DIR

ADDING_DIR  = DATASET_DIR / "adding"
ADDING_H5   = RDM_DIR / "adding_rdms.h5"
T           = 25
N_STIM      = 100
N_PHASES    = 6
N_ROWS      = N_STIM * N_PHASES   # 600

PHASE_DESCS = [
    "before flag1 (t < flag1; NaN where flag1 == 0)",
    "at flag1",
    "between flags (flag1 < t < flag2; NaN where adjacent)",
    "at flag2",
    "after flag2 (flag2 < t < T-1; NaN where flag2 >= T-2)",
    "final step (t = T-1)",
]


# ---------------------------------------------------------------------------
# Stimulus metadata
# ---------------------------------------------------------------------------

def get_flag_positions():
    """Return (N_STIM, 2) int32 array of flag timestep positions."""
    sys.path.insert(0, str(ANALYSIS.parent))
    from tasks import TASKS
    task = TASKS["adding"]()
    inputs, _ = task.get_rdm_stimuli()   # (100, 25, 2)
    pos = []
    for i in range(N_STIM):
        flagged = np.where(inputs[i, :, 1] > 0.5)[0]
        assert len(flagged) == 2, f"stimulus {i}: expected 2 flags, got {len(flagged)}"
        pos.append(sorted(flagged.tolist()))
    return np.array(pos, dtype=np.int32)


def temporal_meta(flag_pos):
    """
    Row labels and valid mask for the 600-row temporal RDM.
    Returns row_stim (600,), row_phase (600,), valid_mask (600,).
    """
    row_stim  = np.repeat(np.arange(N_STIM,  dtype=np.int32), N_PHASES)
    row_phase = np.tile(  np.arange(N_PHASES, dtype=np.int32), N_STIM)
    f1, f2 = flag_pos[:, 0], flag_pos[:, 1]
    phase_valid = np.zeros((N_PHASES, N_STIM), dtype=bool)
    phase_valid[0] = f1 > 0
    phase_valid[1] = True
    phase_valid[2] = f2 > f1 + 1
    phase_valid[3] = True
    phase_valid[4] = f2 < T - 2
    phase_valid[5] = True
    valid_mask = phase_valid[row_phase, row_stim]
    return row_stim, row_phase, valid_mask


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _unit_norm(mat):
    """Row-normalise to unit vectors; near-zero rows stay near-zero."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.where(norms > 1e-8, norms, 1.0)


def build_temporal_matrix(acts_by_t, flag_pos):
    """
    Build (N_ROWS, H) = (600, H) matrix M of phase mean unit vectors.

    M[i * N_PHASES + p] = mean of unit(h_i[t]) over t valid for phase p.
    Rows where stimulus i has no valid timesteps in phase p are NaN.

    WARNING — cosine distance only: RDM = 1 - M @ M.T.
    Does not generalise to other metrics.
    """
    f1, f2 = flag_pos[:, 0], flag_pos[:, 1]
    H  = acts_by_t[0].shape[1]
    M  = np.full((N_ROWS, H), np.nan, dtype=np.float64)

    def _mean_unit(valid_fn):
        acc = np.zeros((N_STIM, H), dtype=np.float64)
        cnt = np.zeros(N_STIM,      dtype=np.int32)
        for t in range(T):
            ok = valid_fn(t)
            if not ok.any():
                continue
            acc[ok] += _unit_norm(acts_by_t[t].astype(np.float64))[ok]
            cnt[ok] += 1
        out = np.full((N_STIM, H), np.nan, dtype=np.float64)
        valid = cnt > 0
        out[valid] = acc[valid] / cnt[valid, None]
        return out

    def _single_step(t_per_stim):
        mat = np.stack([acts_by_t[t_per_stim[i]][i]
                        for i in range(N_STIM)]).astype(np.float64)
        return _unit_norm(mat)

    M[0::N_PHASES] = _mean_unit(lambda t: t < f1)
    M[1::N_PHASES] = _single_step(f1)
    M[2::N_PHASES] = _mean_unit(lambda t: (t > f1) & (t < f2))
    M[3::N_PHASES] = _single_step(f2)
    M[4::N_PHASES] = _mean_unit(lambda t: (t > f2) & (t < T - 1))
    M[5::N_PHASES] = _unit_norm(acts_by_t[T - 1].astype(np.float64))

    return M.astype(np.float32)


def compute_temporal_rdm(M):
    """
    Upper-triangle of 1 - M @ M.T.
    NaN rows in M produce NaN entries (propagates through matrix multiply).
    Returns None if no valid (non-NaN) entries exist at all.
    """
    Md   = M.astype(np.float64)
    gram = Md @ Md.T
    if not np.any(np.isfinite(gram)):
        return None
    D = 1.0 - gram
    finite = np.isfinite(D)
    D[finite] = np.clip(D[finite], 0.0, 2.0)
    rows, cols = np.triu_indices(N_ROWS, k=1)
    return D[rows, cols].astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not ADDING_H5.exists():
        raise FileNotFoundError(
            f"adding_rdms.h5 not found at {ADDING_H5}. Run 10_compute_rdms.py first.")

    print("Loading flag positions ...")
    flag_pos = get_flag_positions()
    row_stim, row_phase, valid_mask = temporal_meta(flag_pos)

    n_pairs = N_ROWS * (N_ROWS - 1) // 2
    print(f"Temporal RDM: {N_ROWS} rows ({N_STIM} stimuli × {N_PHASES} phases)")
    print(f"  {valid_mask.sum()}/{N_ROWS} rows valid  "
          f"({(~valid_mask).sum()} NaN rows fixed across all networks)")
    print(f"  Upper triangle: {n_pairs:,} pairs  (~{n_pairs * 4 / 1e6:.1f} MB per network uncompressed)")
    for p in range(N_PHASES):
        n = valid_mask[row_phase == p].sum()
        print(f"  Phase {p}: {n}/{N_STIM} valid — {PHASE_DESCS[p]}")

    with h5py.File(ADDING_H5, "a") as h5:
        meta = h5.require_group("meta")

        # Row metadata — fixed across all networks, written once
        for key in ("temporal_row_stim", "temporal_row_phase",
                    "temporal_valid_mask", "temporal_phase_descs"):
            if key in meta:
                del meta[key]
        meta.create_dataset("temporal_row_stim",   data=row_stim)
        meta.create_dataset("temporal_row_phase",  data=row_phase)
        meta.create_dataset("temporal_valid_mask", data=valid_mask.astype(np.uint8))
        meta.create_dataset("temporal_phase_descs",
                            data=np.array(PHASE_DESCS, dtype=h5py.special_dtype(vlen=str)))
        meta.attrs["temporal_n_rows"]  = N_ROWS
        meta.attrs["temporal_n_pairs"] = n_pairs

        runs_grp = h5.get("runs")
        if runs_grp is None:
            raise RuntimeError("No 'runs' group in adding_rdms.h5.")

        run_ids    = sorted(runs_grp.keys())
        n_total    = len(run_ids)
        n_computed = 0
        n_skipped  = 0
        n_degen    = 0

        for idx, run_id in enumerate(run_ids):
            run_grp    = runs_grp[run_id]
            run_dir    = ADDING_DIR / run_id
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

                M   = build_temporal_matrix(acts_by_t, flag_pos)
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
        print(f"  {n_degen} degenerate (no valid entries)")
    print(f"  Stored in: {ADDING_H5}")


if __name__ == "__main__":
    main()
