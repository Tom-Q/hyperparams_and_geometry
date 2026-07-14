#!/usr/bin/env python3
"""
Step 12b: Temporal category model RDMs for the Adding task.

The adding temporal RDM is 600×600 (100 stimuli × 6 phases). Category models
must be in the same space. Four models:

  phase  — block by phase: 0 if phase_p == phase_q, else 1
  value1 — graded |val1_i - val1_j|, val1 = value at flag1 (earlier flag)
  value2 — graded |val2_i - val2_j|, val2 = value at flag2 (later flag)
  sum    — graded |sum_i - sum_j|, sum = val1 + val2 = task target

For value/sum models, the model value depends only on stimulus identity —
D[k, l] = D_100[stim_k, stim_l] — so the same distance repeats across phases.
For the phase model, it depends only on phase — D[k, l] = float(phase_k != phase_l).

Row metadata (row_stim, row_phase) is read from adding_rdms.h5 (written by 10b).
Run 10b before running this script.

Output:
    output/analysis/cache/category_models/adding_temporal.npz — 4 keys, each 600×600 float32
    output/analysis/figures/category_models_adding_temporal.pdf

Usage:
    python 12b_adding_category_models.py
"""

import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANALYSIS = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import CACHE_DIR, FIGURES_DIR, RDM_DIR

MODELS_DIR = CACHE_DIR / "category_models"
ADDING_H5  = RDM_DIR / "adding_rdms.h5"
N_STIM     = 100
N_PHASES   = 6
N_ROWS     = N_STIM * N_PHASES   # 600


def graded_model(values):
    """Normalised pairwise |v_i - v_j|, scaled to [0, 1]. (N,) → (N, N) float32."""
    v = np.asarray(values, dtype=np.float32)
    D = np.abs(v[:, None] - v[None, :])
    if D.max() > 0:
        D /= D.max()
    np.fill_diagonal(D, 0.0)
    return D


def build_models(inputs, targets, row_stim, row_phase):
    """
    Build 600×600 temporal category model matrices.

    inputs    : (100, 25, 2)  — adding stimuli; channel 0 = value, channel 1 = flag
    targets   : (100,)        — val1 + val2 (task target)
    row_stim  : (600,)        — stimulus index for each temporal row
    row_phase : (600,)        — phase index for each temporal row
    """
    val1 = np.zeros(N_STIM, dtype=np.float32)
    val2 = np.zeros(N_STIM, dtype=np.float32)
    for i in range(N_STIM):
        flagged = np.where(inputs[i, :, 1] > 0.5)[0]
        assert len(flagged) == 2, f"stimulus {i}: expected 2 flags, got {len(flagged)}"
        val1[i] = inputs[i, flagged[0], 0]   # earlier flag
        val2[i] = inputs[i, flagged[1], 0]   # later flag

    # 100-stim graded models expanded to temporal space via row_stim indexing
    D_value1 = graded_model(val1)[np.ix_(row_stim, row_stim)]
    D_value2 = graded_model(val2)[np.ix_(row_stim, row_stim)]
    D_sum    = graded_model(targets.astype(np.float32))[np.ix_(row_stim, row_stim)]

    # Phase model: depends only on row_phase
    D_phase  = (row_phase[:, None] != row_phase[None, :]).astype(np.float32)
    np.fill_diagonal(D_phase, 0.0)

    return {"phase": D_phase, "value1": D_value1, "value2": D_value2, "sum": D_sum}


def make_figure(models, row_stim, row_phase):
    """Four panels, each showing a 600×600 model sorted by (phase, stimulus)."""
    sort_idx = np.lexsort((row_stim, row_phase))   # primary key: phase
    phase_boundaries = [N_STIM * p for p in range(1, N_PHASES)]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    for ax, (name, D) in zip(axes, models.items()):
        D_s = D[np.ix_(sort_idx, sort_idx)]
        im  = ax.imshow(D_s, cmap="Greys", vmin=0, vmax=1, aspect="equal",
                        interpolation="nearest")
        for b in phase_boundaries:
            ax.axhline(b - 0.5, color="#e05c00", lw=0.6, alpha=0.8)
            ax.axvline(b - 0.5, color="#e05c00", lw=0.6, alpha=0.8)
        ax.set_title(name.replace("_", " "), fontsize=9, fontweight="bold")
        ax.set_xlabel("(phase, stim)", fontsize=7)
        ax.set_ylabel("(phase, stim)", fontsize=7)
        ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax, shrink=0.6, label="dissimilarity")

    fig.suptitle(
        "Adding — temporal category models (600×600, sorted by phase then stimulus)",
        fontsize=10, fontweight="bold")
    return fig


def main():
    if not ADDING_H5.exists():
        raise FileNotFoundError(
            f"adding_rdms.h5 not found at {ADDING_H5}. Run 10b first.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ANALYSIS.parent))
    from tasks import TASKS
    inputs, meta = TASKS["adding"]().get_rdm_stimuli()   # (100, 25, 2), {"targets": (100,)}

    with h5py.File(ADDING_H5, "r") as h5:
        row_stim  = h5["meta/temporal_row_stim"][:]
        row_phase = h5["meta/temporal_row_phase"][:]

    assert len(row_stim) == N_ROWS
    models = build_models(inputs, meta["targets"], row_stim, row_phase)

    for name, D in models.items():
        assert D.shape == (N_ROWS, N_ROWS), f"{name}: unexpected shape {D.shape}"
        assert np.all(np.isfinite(D)), f"{name}: contains NaN/Inf"

    out_npz = MODELS_DIR / "adding_temporal.npz"
    np.savez(out_npz, **models)
    print(f"Saved: {out_npz}  models={list(models.keys())}")

    fig = make_figure(models, row_stim, row_phase)
    out_fig = FIGURES_DIR / "category_models_adding_temporal.pdf"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_fig.name}")


if __name__ == "__main__":
    main()
