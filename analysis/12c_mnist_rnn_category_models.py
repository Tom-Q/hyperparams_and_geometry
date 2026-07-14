#!/usr/bin/env python3
"""
Step 12c: Temporal category model RDMs for mnist_rnn.

The mnist_rnn temporal RDM is 1400×1400 (100 stimuli × 14 timesteps).
Two models:

  digit         — static block model: 0 if digit_i == digit_j, else 1
                  (same digit identity at every timestep, no temporal weighting)

  digit_linear  — linearly growing categorical differentiation:
                  D[(i, t_a), (j, t_b)] = ((t_a + t_b) / (2*(T-1))) * float(digit_i != digit_j)
                  At t=0: all pairs look identical (undifferentiated).
                  At t=T-1: same as the static block model.
                  Cross-timestep: weighted by the mean developmental stage.

Row ordering: stimulus-major — (stim_0, t=0), ..., (stim_0, t=13), (stim_1, t=0), ...

Output:
    output/analysis/cache/category_models/mnist_rnn_temporal.npz — 2 keys, each 1400×1400 float32
    output/analysis/figures/category_models_mnist_rnn_temporal.pdf

Usage:
    python 12c_mnist_rnn_category_models.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANALYSIS = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import CACHE_DIR, FIGURES_DIR

MODELS_DIR = CACHE_DIR / "category_models"
N_STIM     = 100
T          = 14
N_ROWS     = N_STIM * T   # 1400


def build_models(digits, row_stim, row_t):
    """
    Build 1400×1400 temporal category model matrices.

    digits   : (100,) int  — digit label for each stimulus
    row_stim : (1400,) int — stimulus index for each temporal row
    row_t    : (1400,) int — timestep for each temporal row
    """
    row_digit = digits[row_stim]   # (1400,) digit for each row

    diff = (row_digit[:, None] != row_digit[None, :]).astype(np.float32)
    np.fill_diagonal(diff, 0.0)

    # Static: block model, timestep-invisible
    D_digit = diff.copy()

    # Linear: weight by mean timestep stage
    weight = (row_t[:, None] + row_t[None, :]) / (2.0 * (T - 1))
    D_linear = weight * diff
    np.fill_diagonal(D_linear, 0.0)

    return {"digit": D_digit, "digit_linear": D_linear}


def make_figure(models, row_t, row_digit):
    """Two panels sorted by (timestep, digit)."""
    sort_idx = np.lexsort((row_digit, row_t))   # primary key: timestep
    t_boundaries = [N_STIM * t for t in range(1, T)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    titles = {
        "digit":        "digit (static)",
        "digit_linear": "digit_linear (grows with time)",
    }
    for ax, (name, D) in zip(axes, models.items()):
        D_s = D[np.ix_(sort_idx, sort_idx)]
        im  = ax.imshow(D_s, cmap="Greys", vmin=0, vmax=1, aspect="equal",
                        interpolation="nearest")
        for b in t_boundaries:
            ax.axhline(b - 0.5, color="#e05c00", lw=0.4, alpha=0.6)
            ax.axvline(b - 0.5, color="#e05c00", lw=0.4, alpha=0.6)
        ax.set_title(titles[name], fontsize=9, fontweight="bold")
        ax.set_xlabel("(timestep, digit)", fontsize=7)
        ax.set_ylabel("(timestep, digit)", fontsize=7)
        ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax, shrink=0.6, label="dissimilarity")

    fig.suptitle(
        "mnist_rnn — temporal category models (1400×1400, sorted by timestep then digit)",
        fontsize=10, fontweight="bold")
    return fig


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ANALYSIS.parent))
    from tasks import TASKS
    _, meta = TASKS["mnist_rnn"]().get_rdm_stimuli()
    digits = meta["digits"].astype(np.int32)   # (100,)

    row_stim = np.repeat(np.arange(N_STIM, dtype=np.int32), T)
    row_t    = np.tile(  np.arange(T,      dtype=np.int32), N_STIM)

    models = build_models(digits, row_stim, row_t)

    for name, D in models.items():
        assert D.shape == (N_ROWS, N_ROWS), f"{name}: unexpected shape {D.shape}"
        assert np.all(np.isfinite(D)), f"{name}: contains NaN/Inf"
        assert D.min() >= 0.0 and D.max() <= 1.0, f"{name}: out of [0,1]"

    out_npz = MODELS_DIR / "mnist_rnn_temporal.npz"
    np.savez(out_npz, **models)
    print(f"Saved: {out_npz}  models={list(models.keys())}")

    row_digit = digits[row_stim]
    fig = make_figure(models, row_t, row_digit)
    out_fig = FIGURES_DIR / "category_models_mnist_rnn_temporal.pdf"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_fig.name}")


if __name__ == "__main__":
    main()
