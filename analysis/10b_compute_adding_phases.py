#!/usr/bin/env python3
"""
Step 10b: Semantic phase-aligned RDMs for the Adding task.

The 100 fixed stimuli each have exactly 2 flagged time steps (positions vary
per stimulus). This script computes one activation matrix per phase by
averaging (or selecting) the relevant time steps, then storing a cosine-
distance RDM for each phase.

Six phases:
  phase_1  before flag 1      mean of t < flag1
  phase_2  at flag 1          t = flag1  (single step, per-stimulus)
  phase_3  between flags      mean of flag1 < t < flag2
  phase_4  at flag 2          t = flag2  (single step, per-stimulus)
  phase_5  after flag 2       mean of flag2 < t < 24
  phase_6  final step         t = 24  (same for all stimuli)

Phases 1, 3, 5 may be empty for some stimuli. Those stimuli are excluded
and a boolean validity mask (N_PHASES × 100) is stored in meta/phase_masks.
Phase RDMs therefore have variable length n_valid*(n_valid-1)//2.

Appends layer_N_phase_K keys to existing adding_rdms.h5.

Usage:
  python 10b_compute_adding_phases.py
  python 10b_compute_adding_phases.py --overwrite
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

ANALYSIS = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import DATASET_DIR, RDM_DIR

ADDING_DIR  = DATASET_DIR / "adding_failed_run"
ADDING_H5   = RDM_DIR / "adding_rdms.h5"
T           = 25
N_STIM      = 100
N_PHASES    = 6

PHASE_NAMES = [
    "phase_1", "phase_2", "phase_3",
    "phase_4", "phase_5", "phase_6",
]
PHASE_DESCS = [
    "before flag1 (mean t < flag1)",
    "at flag1 (t = flag1)",
    "between flags (mean flag1 < t < flag2)",
    "at flag2 (t = flag2)",
    "after flag2 (mean flag2 < t < 24)",
    "final step (t = 24)",
]


# ---------------------------------------------------------------------------
# Stimulus flag positions
# ---------------------------------------------------------------------------

def get_flag_positions():
    """Return (100, 2) int array of sorted flag positions per stimulus."""
    sys.path.insert(0, str(ANALYSIS.parent))
    from tasks import TASKS
    task = TASKS["adding"]()
    inputs, _ = task.get_rdm_stimuli()   # (100, 25, 2)
    pos = []
    for i in range(N_STIM):
        flagged = np.where(inputs[i, :, 1] > 0.5)[0]
        assert len(flagged) == 2, f"stimulus {i}: expected 2 flags, got {len(flagged)}"
        pos.append(sorted(flagged.tolist()))
    return np.array(pos, dtype=np.int32)   # (100, 2)


def compute_phase_masks(flag_pos):
    """Boolean (6, 100): True when stimulus has ≥1 time step in phase."""
    f1, f2 = flag_pos[:, 0], flag_pos[:, 1]
    masks = np.zeros((N_PHASES, N_STIM), dtype=bool)
    masks[0] = f1 > 0               # phase 1: at least one t before flag1
    masks[1] = True                  # phase 2: always (single step)
    masks[2] = f2 > f1 + 1          # phase 3: at least one t strictly between flags
    masks[3] = True                  # phase 4: always (single step)
    masks[4] = f2 < T - 2            # phase 5: at least one t in (flag2, 24)
    masks[5] = True                  # phase 6: always (t = 24)
    return masks


# ---------------------------------------------------------------------------
# Phase activation extraction
# ---------------------------------------------------------------------------

def phase_activations(acts_by_t, flag_pos, phase_masks):
    """
    For each of the 6 phases, compute one activation vector per valid stimulus.

    acts_by_t  : dict  t -> (100, H)  float32
    flag_pos   : (100, 2) int
    phase_masks: (6, 100) bool

    Returns list of 6 arrays, each (n_valid, H), or None if n_valid < 2.
    """
    f1, f2 = flag_pos[:, 0], flag_pos[:, 1]
    H = acts_by_t[0].shape[1]
    result = []

    def _mean_phase(t_mask_fn):
        """Average acts over time steps where t_mask_fn(t, stimulus) is True."""
        acc = np.zeros((N_STIM, H), dtype=np.float64)
        cnt = np.zeros(N_STIM, dtype=np.int32)
        for t in range(T):
            ok = t_mask_fn(t)    # (100,) bool
            acc[ok] += acts_by_t[t][ok]
            cnt[ok] += 1
        valid = cnt > 0
        out = np.zeros((N_STIM, H), dtype=np.float32)
        out[valid] = (acc[valid] / cnt[valid, None]).astype(np.float32)
        return out

    # Phase 1 — before flag1
    mask1 = phase_masks[0]
    a1 = _mean_phase(lambda t: t < f1)
    result.append(a1[mask1] if mask1.sum() >= 2 else None)

    # Phase 2 — at flag1
    a2 = np.stack([acts_by_t[f1[i]][i] for i in range(N_STIM)]).astype(np.float32)
    result.append(a2)

    # Phase 3 — between flags
    mask3 = phase_masks[2]
    a3 = _mean_phase(lambda t: (t > f1) & (t < f2))
    result.append(a3[mask3] if mask3.sum() >= 2 else None)

    # Phase 4 — at flag2
    a4 = np.stack([acts_by_t[f2[i]][i] for i in range(N_STIM)]).astype(np.float32)
    result.append(a4)

    # Phase 5 — after flag2
    mask5 = phase_masks[4]
    a5 = _mean_phase(lambda t: (t > f2) & (t < T - 1))
    result.append(a5[mask5] if mask5.sum() >= 2 else None)

    # Phase 6 — final step
    result.append(acts_by_t[T - 1].astype(np.float32))

    return result


# ---------------------------------------------------------------------------
# RDM computation
# ---------------------------------------------------------------------------

def compute_rdm(activations):
    """Cosine-distance upper triangle. Returns None if degenerate."""
    if np.any(~np.isfinite(activations)):
        return None
    norms = np.linalg.norm(activations, axis=1)
    if np.any(norms < 1e-8):
        return None
    dist = cosine_distances(activations.astype(np.float32))
    n = dist.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    return dist[rows, cols].astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not ADDING_H5.exists():
        raise FileNotFoundError(f"adding_rdms.h5 not found at {ADDING_H5}. Run 10_compute_rdms.py first.")

    print("Loading flag positions from fixed stimuli ...")
    flag_pos = get_flag_positions()
    f1, f2 = flag_pos[:, 0], flag_pos[:, 1]
    phase_masks = compute_phase_masks(flag_pos)

    print("Phase validity across 100 stimuli:")
    for k, (name, desc) in enumerate(zip(PHASE_NAMES, PHASE_DESCS)):
        n = phase_masks[k].sum()
        print(f"  {name}: {n}/100 valid  ({desc})")

    with h5py.File(ADDING_H5, "a") as h5:
        # Store masks in meta
        meta = h5["meta"]
        for key in ("phase_masks", "phase_n_valid"):
            if key in meta:
                del meta[key]
        meta.create_dataset("phase_masks",   data=phase_masks.astype(np.uint8))
        meta.create_dataset("phase_n_valid", data=phase_masks.sum(axis=1).astype(np.int32))

        runs_grp = h5.get("runs")
        if runs_grp is None:
            raise RuntimeError("No 'runs' group in adding_rdms.h5.")

        run_ids  = sorted(runs_grp.keys())
        n_total  = len(run_ids)
        n_computed = 0

        for idx, run_id in enumerate(run_ids):
            run_grp = runs_grp[run_id]
            run_dir = ADDING_DIR / run_id
            ckpt_files = sorted(run_dir.glob("*.npz")) if run_dir.exists() else []

            for ckpt_path in ckpt_files:
                ckpt_name = ckpt_path.stem
                ckpt_grp  = run_grp.require_group(ckpt_name)

                # Discover layers in this npz
                try:
                    npz = np.load(ckpt_path)
                except Exception as e:
                    print(f"  [warn] cannot load {run_id}/{ckpt_name}: {e}")
                    continue

                layers = sorted(set(
                    int(k.split("_t_")[0].split("layer_")[1])
                    for k in npz.keys() if "_t_" in k
                ))

                for layer in layers:
                    # Skip if all phases already computed and not overwriting
                    if not args.overwrite:
                        all_done = all(
                            f"layer_{layer}_{p}" in ckpt_grp
                            for p in PHASE_NAMES
                        )
                        if all_done:
                            continue

                    # Load all T time steps for this layer
                    acts_by_t = {}
                    missing = False
                    for t in range(T):
                        key = f"layer_{layer}_t_{t}"
                        if key not in npz:
                            missing = True
                            break
                        acts_by_t[t] = npz[key].astype(np.float32)
                    if missing:
                        continue

                    phase_acts = phase_activations(acts_by_t, flag_pos, phase_masks)

                    for k, (pname, acts) in enumerate(zip(PHASE_NAMES, phase_acts)):
                        hdf5_key = f"layer_{layer}_{pname}"
                        if hdf5_key in ckpt_grp:
                            if args.overwrite:
                                del ckpt_grp[hdf5_key]
                            else:
                                continue

                        if acts is None:
                            ds = ckpt_grp.create_dataset(
                                hdf5_key, data=np.array([], dtype=np.float32))
                            ds.attrs["degenerate"] = True
                            ds.attrs["reason"] = "insufficient_valid_stimuli"
                            continue

                        rdm = compute_rdm(acts)
                        if rdm is None:
                            ds = ckpt_grp.create_dataset(
                                hdf5_key, data=np.array([], dtype=np.float32))
                            ds.attrs["degenerate"] = True
                            continue

                        ckpt_grp.create_dataset(
                            hdf5_key, data=rdm,
                            compression="gzip", compression_opts=4, shuffle=True)
                        n_computed += 1

            if (idx + 1) % 100 == 0:
                print(f"  {idx + 1}/{n_total} runs processed ...")

        print(f"\n  {n_computed} phase RDMs computed.")
        print(f"  Stored in: {ADDING_H5}")


if __name__ == "__main__":
    main()
