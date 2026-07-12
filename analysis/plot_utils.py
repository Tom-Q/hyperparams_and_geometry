"""
Shared plotting utilities for RDM analysis figures.
"""

import numpy as np


def vec_to_rdm(vec):
    """
    Reconstruct an N×N symmetric RDM from an upper-triangle vector.
    N is inferred from len(vec) = N*(N-1)//2.
    """
    n_pairs = len(vec)
    n = int(round((1 + np.sqrt(1 + 8 * n_pairs)) / 2))
    assert n * (n - 1) // 2 == n_pairs, f"vec length {n_pairs} is not a valid upper-triangle size"
    D = np.zeros((n, n), dtype=np.float32)
    rows, cols = np.triu_indices(n, k=1)
    D[rows, cols] = vec
    D += D.T
    return D


def plot_rdm(ax, rdm, title="", sort_idx=None, cmap="Greys", vmin=0, vmax=1,
             line_vals=None):
    """
    Plot an RDM as a colour-mapped matrix.

    Parameters
    ----------
    ax        : matplotlib Axes to draw on
    rdm       : (N, N) float array, or upper-triangle vector (1-D, auto-expanded)
    title     : axes title string
    sort_idx  : integer permutation array of length N; if given, rows/cols are
                reordered before plotting
    cmap      : matplotlib colormap name
    vmin/vmax : colour scale limits
    line_vals : integer label array of length N (in the post-sort order); when
                given, white lines are drawn at category boundaries

    Returns
    -------
    AxesImage (the return value of ax.imshow)
    """
    if np.ndim(rdm) == 1:
        rdm = vec_to_rdm(rdm)

    if sort_idx is not None:
        rdm = rdm[np.ix_(sort_idx, sort_idx)]

    im = ax.imshow(rdm, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="equal", interpolation="nearest")

    if line_vals is not None:
        sv = np.asarray(line_vals)
        N = len(sv)
        prev = sv[0]
        for i in range(1, N):
            if sv[i] != prev:
                ax.axhline(i - 0.5, color="#e05c00", lw=0.8, alpha=0.9)
                ax.axvline(i - 0.5, color="#e05c00", lw=0.8, alpha=0.9)
                prev = sv[i]

    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return im
