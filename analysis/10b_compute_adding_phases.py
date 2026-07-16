#!/usr/bin/env python3
"""
Step 10b: Phase-averaged temporal RDMs (cosine + Pearson) for the Adding task.

Produces two RDMs per network stored as "temporal_cosine" and "temporal_pearson".
Each has 600 rows/cols: (stimulus, phase) pairs for 100 stimuli × 6 phases,
ordered stimulus-major: (stim_0, phase_0), ..., (stim_0, phase_5), ...

temporal_cosine: 1 - M_cos @ M_cos.T
  where M_cos[i,p] = mean of unit(h_i[t]) over valid timesteps t in phase p.
  (mean unit vector trick — valid only for cosine distance)

temporal_pearson: 1 - M_pear @ M_pear.T
  where M_pear[i,p] = mean of h_i[t] over valid phase timesteps, then
  mean-centered across units and unit-normalized. Equivalent to Pearson
  correlation distance between phase-mean activation patterns.

Phases (0-indexed):
  0  before flag1       t < flag1         NaN where flag1 == 0
  1  at flag1           t = flag1
  2  between flags      flag1 < t < flag2 NaN where flags adjacent
  3  at flag2           t = flag2
  4  after flag2        flag2 < t < T-1   NaN where flag2 >= T-2
  5  final step         t = T-1

Rows where a stimulus has no valid timesteps in a phase are NaN (fixed across
networks, depends only on the stimulus set). Use nan_policy='omit' downstream
for adding only; NaN in any other task indicates a bug.

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


def build_temporal_matrices(acts_by_t, flag_pos):
    """
    Build two (N_ROWS, H) = (600, H) matrices:
      M_cos:  phase mean unit vectors (for cosine RDM, mean unit vector trick)
      M_raw:  phase mean raw activations (for Pearson RDM)
    Rows where a stimulus has no valid timesteps in that phase are NaN in both.
    """
    f1, f2 = flag_pos[:, 0], flag_pos[:, 1]
    H      = acts_by_t[0].shape[1]
    M_cos  = np.full((N_ROWS, H), np.nan, dtype=np.float64)
    M_raw  = np.full((N_ROWS, H), np.nan, dtype=np.float64)

    def _mean_both(valid_fn, row_slice):
        acc_u = np.zeros((N_STIM, H), dtype=np.float64)
        acc_r = np.zeros((N_STIM, H), dtype=np.float64)
        cnt   = np.zeros(N_STIM,      dtype=np.int32)
        for t in range(T):
            ok = valid_fn(t)
            if not np.any(ok):
                continue
            a = acts_by_t[t].astype(np.float64)
            acc_u[ok] += _unit_norm(a)[ok]
            acc_r[ok] += a[ok]
            cnt[ok]   += 1
        valid = cnt > 0
        out_u = np.full((N_STIM, H), np.nan, dtype=np.float64)
        out_r = np.full((N_STIM, H), np.nan, dtype=np.float64)
        out_u[valid] = acc_u[valid] / cnt[valid, None]
        out_r[valid] = acc_r[valid] / cnt[valid, None]
        M_cos[row_slice] = out_u
        M_raw[row_slice] = out_r

    def _single_both(t_per_stim, row_slice):
        mat = np.stack([acts_by_t[t_per_stim[i]][i]
                        for i in range(N_STIM)]).astype(np.float64)
        M_cos[row_slice] = _unit_norm(mat)
        M_raw[row_slice] = mat

    _mean_both(lambda t: t < f1,               slice(0, None, N_PHASES))
    _single_both(f1,                            slice(1, None, N_PHASES))
    _mean_both(lambda t: (t > f1) & (t < f2),  slice(2, None, N_PHASES))
    _single_both(f2,                            slice(3, None, N_PHASES))
    _mean_both(lambda t: (t > f2) & (t < T-1), slice(4, None, N_PHASES))
    mat = acts_by_t[T - 1].astype(np.float64)
    M_cos[5::N_PHASES] = _unit_norm(mat)
    M_raw[5::N_PHASES] = mat

    return M_cos.astype(np.float32), M_raw.astype(np.float32)


def compute_temporal_cosine_rdm(M_cos):
    """
    Upper-triangle of 1 - M_cos @ M_cos.T.
    NaN rows propagate as NaN. Returns None if no finite entries exist.
    """
    Md   = M_cos.astype(np.float64)
    gram = Md @ Md.T
    if not np.any(np.isfinite(gram)):
        return None
    D = 1.0 - gram
    finite = np.isfinite(D)
    D[finite] = np.clip(D[finite], 0.0, 2.0)
    rows, cols = np.triu_indices(N_ROWS, k=1)
    return D[rows, cols].astype(np.float32)


def compute_temporal_pearson_rdm(M_raw):
    """
    Pearson-distance temporal RDM from raw phase-mean activations.
    Mean-centers each valid row across units, unit-normalizes, then 1 - M @ M.T.
    NaN rows (invalid phases) propagate as NaN in the output.
    Returns None if no valid rows or any valid row is constant (degenerate).
    """
    Md = M_raw.astype(np.float64)
    valid = np.all(np.isfinite(Md), axis=1)
    if not valid.any():
        return None
    Md[valid] -= Md[valid].mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Md[valid], axis=1)
    if np.any(norms < 1e-8) or not np.all(np.isfinite(norms)):
        return None
    Md[valid] /= norms[:, None]
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

                M_cos, M_raw = build_temporal_matrices(acts_by_t, flag_pos)

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
        print(f"  {n_degen} degenerate (no valid entries)")
    print(f"  Stored in: {ADDING_H5}")


if __name__ == "__main__":
    main()
