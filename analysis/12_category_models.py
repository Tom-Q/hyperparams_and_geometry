#!/usr/bin/env python3
"""
Step 12: Generate category model RDMs for each task.

Category models are ideal RDMs based on the known structure of each task's
stimulus set. Used in Finding #1.3 to test how well network representations
reflect task-relevant organization.

Models generated:
  mnist_dual      — output (4 output-category blocks), digit (10 blocks), mixed (graded)
  mnist_10way     — digit (10 blocks)
  fashion_10way   — class (10 blocks)
  mnist_rnn       — digit (10 blocks)
  spirals         — arm (3 blocks)
  parity          — parity_label (2 blocks), hamming_diff (graded |n_ones_a - n_ones_b| / 8)
  cartpole        — angle_diff, euclidean (2D state-space distance)
  fourrooms       — room (4 rooms + hallway), euclidean (grid distance), goal_dist
  adding          — value1 (|val1_i - val1_j|, first addend), sum (|target_i - target_j|)

Outputs:
    output/analysis/cache/category_models/{task}.npz — one key per model name, N×N float32
    output/analysis/figures/category_models_{task}.pdf
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ANALYSIS = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import FIGURES_DIR, CACHE_DIR

MODELS_DIR = CACHE_DIR / "category_models"

# Fashion-MNIST class names for figure labels
FASHION_LABELS = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# ---------------------------------------------------------------------------
# Block / graded model builders
# ---------------------------------------------------------------------------

def block_model(labels):
    """Distance 0 for same label, 1 for different. labels: (N,) int."""
    labels = np.asarray(labels)
    N = len(labels)
    D = np.ones((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if labels[i] == labels[j]:
                D[i, j] = 0.0
    np.fill_diagonal(D, 0.0)
    return D


def graded_model(values, normalize=True):
    """Distance = |v_i - v_j|, optionally normalized to [0, 1]. values: (N,)."""
    v = np.asarray(values, dtype=np.float32)
    D = np.abs(v[:, None] - v[None, :])
    if normalize and D.max() > 0:
        D /= D.max()
    np.fill_diagonal(D, 0.0)
    return D


def euclidean_model(coords, normalize=True):
    """Distance = Euclidean distance in coordinate space. coords: (N, d)."""
    coords = np.asarray(coords, dtype=np.float32)
    # ||a - b||_2
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=-1))
    if normalize and D.max() > 0:
        D /= D.max()
    np.fill_diagonal(D, 0.0)
    return D


# ---------------------------------------------------------------------------
# Per-task model factories
# ---------------------------------------------------------------------------

def models_mnist_dual(inputs, meta):
    digits = meta["digits"]   # (200,)
    tasks  = meta["tasks"]    # (200,) — 0 or 1

    # Output labels per task bit
    # task 0: even=1, odd=0; task 1: small (<5)=1, large=0
    out_labels = np.where(tasks == 0, (digits % 2 == 0).astype(int),
                                      (digits < 5).astype(int))
    # 4 output categories encoded as (task_bit, out_label)
    out_category = tasks * 2 + out_labels   # 0=(0,0)=odd, 1=(0,1)=even, 2=(1,0)=large, 3=(1,1)=small

    D_output = block_model(out_category)

    D_digit = block_model(digits)

    # Mixed: distance = 1 - 0.5*(same_digit + same_output)
    same_digit  = (digits[:, None] == digits[None, :]).astype(np.float32)
    same_output = (out_category[:, None] == out_category[None, :]).astype(np.float32)
    D_mixed = (1.0 - 0.5 * (same_digit + same_output)).astype(np.float32)
    np.fill_diagonal(D_mixed, 0.0)

    return {
        "output": D_output,
        "digit":  D_digit,
        "mixed":  D_mixed,
    }, {
        "output": out_category,
        "digit":  digits,
        "mixed":  out_category,   # sort by output for display
    }


def models_mnist_10way(inputs, meta):
    digits = meta["digits"]
    D = block_model(digits)
    return {"digit": D}, {"digit": digits}


def models_fashion(inputs, meta):
    classes = meta["classes"]
    D = block_model(classes)
    return {"class": D}, {"class": classes}


def models_mnist_rnn(inputs, meta):
    digits = meta["digits"]
    D = block_model(digits)
    return {"digit": D}, {"digit": digits}


def models_spirals(inputs, meta):
    """inputs: (198, 2) noiseless spiral coordinates."""
    classes = meta["classes"]
    D_arm     = block_model(classes)
    D_spatial = euclidean_model(inputs)   # distance in 2D spiral space
    return {
        "arm":     D_arm,
        "spatial": D_spatial,
    }, {
        "arm":     classes,
        "spatial": classes,   # sort by arm for display
    }


def models_parity(inputs, meta):
    n_ones = meta["n_ones"]   # Hamming weight 0-8
    labels = meta["labels"]   # parity 0/1

    D_label = block_model(labels)
    D_hamming = graded_model(n_ones)   # |n_ones_a - n_ones_b| / 8

    return {
        "parity_label": D_label,
        "hamming_diff": D_hamming,
    }, {
        "parity_label": labels,
        "hamming_diff": n_ones,
    }


def models_cartpole(inputs, meta):
    angles = meta["pole_angles"]   # (196,) rad
    vels   = meta["pole_vels"]     # (196,)

    D_angle = graded_model(angles)
    D_eucl  = euclidean_model(np.stack([angles, vels], axis=1))

    return {
        "angle_diff": D_angle,
        "euclidean":  D_eucl,
    }, {
        "angle_diff": angles,
        "euclidean":  angles,   # sort by angle for display
    }


def models_adding(inputs, meta):
    """inputs: (100, 25, 2), meta: {'targets': (100,) sums}."""
    targets = meta["targets"]
    N = inputs.shape[0]

    val1 = np.zeros(N, dtype=np.float32)
    for i in range(N):
        flagged = np.where(inputs[i, :, 1] > 0.5)[0]
        assert len(flagged) == 2, f"stimulus {i}: expected 2 flags"
        val1[i] = inputs[i, flagged[0], 0]

    D_val1 = graded_model(val1)
    D_sum  = graded_model(targets)

    return {
        "value1": D_val1,
        "sum":    D_sum,
    }, {
        "value1": val1,
        "sum":    targets,
    }


def models_fourrooms(inputs, meta):
    rows = meta["rows"].astype(float)
    cols = meta["cols"].astype(float)
    N = len(rows)

    # Room assignment
    room = np.full(N, 4, dtype=int)   # 4 = hallway
    for i in range(N):
        r, c = int(rows[i]), int(cols[i])
        if r <= 4 and c <= 4:
            room[i] = 0   # top-left
        elif r <= 4 and c >= 6:
            room[i] = 1   # top-right
        elif r >= 6 and c <= 4:
            room[i] = 2   # bottom-left
        elif r >= 6 and c >= 6:
            room[i] = 3   # bottom-right
        # else: hallway (row==5 or col==5) stays at 4

    # Manhattan distance to goal (9, 9)
    goal_dist = np.abs(rows - 9) + np.abs(cols - 9)

    D_room   = block_model(room)
    D_eucl   = euclidean_model(np.stack([rows, cols], axis=1))
    D_gdist  = graded_model(goal_dist)

    return {
        "room":      D_room,
        "euclidean": D_eucl,
        "goal_dist": D_gdist,
    }, {
        "room":      room,
        "euclidean": goal_dist,   # sort by goal distance for display
        "goal_dist": goal_dist,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

CMAP = "Greys"


def sort_order(sort_vals):
    """Return indices that sort by sort_vals."""
    return np.argsort(sort_vals, kind="stable")


def draw_category_lines(ax, sort_vals):
    """Draw lines at category boundaries (for integer labels)."""
    sorted_vals = np.array(sort_vals)[sort_order(sort_vals)]
    N = len(sorted_vals)
    prev = sorted_vals[0]
    for i in range(1, N):
        if sorted_vals[i] != prev:
            ax.axhline(i - 0.5, color="#e05c00", lw=0.8, alpha=0.9)
            ax.axvline(i - 0.5, color="#e05c00", lw=0.8, alpha=0.9)
            prev = sorted_vals[i]


def plot_rdm(ax, D, sort_vals, title, cmap=CMAP):
    idx = sort_order(sort_vals)
    D_sorted = D[np.ix_(idx, idx)]
    im = ax.imshow(D_sorted, cmap=cmap, vmin=0, vmax=1, aspect="equal",
                   interpolation="nearest")
    draw_category_lines(ax, sort_vals)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("stimulus", fontsize=7)
    ax.set_ylabel("stimulus", fontsize=7)
    ax.tick_params(labelsize=6)
    return im


def make_task_figure(task_name, model_dict, sort_dict, display_name):
    model_names = list(model_dict.keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2),
                             constrained_layout=True)
    if n == 1:
        axes = [axes]
    last_im = None
    for ax, name in zip(axes, model_names):
        D = model_dict[name]
        sv = sort_dict[name]
        last_im = plot_rdm(ax, D, sv, name.replace("_", " "))
    fig.suptitle(f"{display_name} — category model RDMs", fontsize=11, fontweight="bold")
    fig.colorbar(last_im, ax=axes[-1], shrink=0.8, label="dissimilarity")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TASK_FACTORIES = {
    "mnist_dual":    (models_mnist_dual,  "MNIST dual"),
    "mnist_10way":   (models_mnist_10way, "MNIST 10-way"),
    "fashion_10way": (models_fashion,     "Fashion 10-way"),
    "mnist_rnn":     (models_mnist_rnn,   "MNIST RNN"),
    "spirals":       (models_spirals,     "Spirals"),
    "parity":        (models_parity,      "Parity"),
    "adding":        (models_adding,      "Adding"),
    "cartpole":      (models_cartpole,    "CartPole"),
    "fourrooms":     (models_fourrooms,   "FourRooms"),
}


def main():
    from tasks import TASKS

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for task_name, (factory, display_name) in TASK_FACTORIES.items():
        print(f"  {task_name} ...", end="", flush=True)
        task = TASKS[task_name]()
        inputs, metadata = task.get_rdm_stimuli()
        model_dict, sort_dict = factory(inputs, metadata)

        # Save arrays
        out_npz = MODELS_DIR / f"{task_name}.npz"
        np.savez(out_npz, **model_dict)

        # Verify sizes
        n_stim = list(model_dict.values())[0].shape[0]
        print(f" N={n_stim}, models={list(model_dict.keys())}")

        # Figure
        fig = make_task_figure(task_name, model_dict, sort_dict, display_name)
        out_fig = FIGURES_DIR / f"category_models_{task_name}.pdf"
        fig.savefig(out_fig, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved: {out_fig.name}")


if __name__ == "__main__":
    main()
