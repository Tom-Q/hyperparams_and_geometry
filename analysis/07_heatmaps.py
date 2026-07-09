#!/usr/bin/env python3
"""
Step 7: 2D marginal heatmaps for all HP pairs.

For each task, produces a multi-page PDF:
  Page 1 — Continuous × Continuous  (C(n_cont, 2) pairs, 8×8 bins each)
  Page 2 — Categorical × Continuous (n_cat * n_cont pairs, cat levels × 8 bins)
  Page 3 — Categorical × Categorical (C(n_cat, 2) pairs, levels × levels)

Each pair shows two panels side-by-side:
  Left  — primary network density (count)
  Right — success rate (fraction of primary networks that are successful;
           grey = no primary networks in that cell)

Output:
  analysis/figures/heatmaps_{task}.pdf  (one multi-page PDF per task)
"""

import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ANALYSIS = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import (
    FIGURES_DIR,
    PRODUCTION_DIR,
    TABLES_DIR,
    TASK_NAMES,
    load_task_df,
    primary_df,
    task_meta,
)

N_BINS       = 8
PAIRS_PER_ROW = 2   # pairs per row within a page

HP_LABELS = {
    "learning_rate": "lr",
    "l1_reg":        "l1",
    "l2_reg":        "l2",
    "hidden_size":   "hidden",
    "batch_size":    "batch",
    "depth":         "depth",
    "activation":    "activation",
    "optimizer":     "optimizer",
    "init_scale":    "init_scale",
    "cell_type":     "cell_type",
    "n_rnn_layers":  "rnn_layers",
    "n_steps":       "steps",
    "n_envs":        "envs",
}

SUCCESS_CMAP = plt.cm.RdYlGn.copy()
SUCCESS_CMAP.set_bad(color="#dddddd")   # grey for empty cells


# ---------------------------------------------------------------------------
# Grid builders
# ---------------------------------------------------------------------------

def _bin(vals, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    return np.clip(np.digitize(vals, edges) - 1, 0, n_bins - 1)


def grid_cont_cont(prim, succ, hp_x, hp_y):
    """(N_BINS × N_BINS) density and success-rate arrays."""
    px = _bin(prim[f"unit_{hp_x}"].values)
    py = _bin(prim[f"unit_{hp_y}"].values)
    sx = _bin(succ[f"unit_{hp_x}"].values) if len(succ) else np.array([], int)
    sy = _bin(succ[f"unit_{hp_y}"].values) if len(succ) else np.array([], int)

    density = np.zeros((N_BINS, N_BINS))
    s_count = np.zeros((N_BINS, N_BINS))
    np.add.at(density, (py, px), 1)
    if len(sx):
        np.add.at(s_count, (sy, sx), 1)
    with np.errstate(invalid="ignore"):
        rate = np.where(density > 0, s_count / density, np.nan)
    return density, rate


def grid_cat_cont(prim, succ, cat_hp, cont_hp, levels):
    """(N_BINS × n_levels) density and success-rate arrays.
    Columns = categorical levels (x-axis), rows = continuous bins (y-axis)."""
    n_lev = len(levels)
    lev_idx = {l: i for i, l in enumerate(levels)}

    density = np.zeros((N_BINS, n_lev))
    s_count = np.zeros((N_BINS, n_lev))

    for df, arr in [(prim, density), (succ, s_count)]:
        cont_bins = _bin(df[f"unit_{cont_hp}"].values)
        cat_vals  = df[cat_hp].values
        for cb, cv in zip(cont_bins, cat_vals):
            if cv in lev_idx:
                arr[cb, lev_idx[cv]] += 1

    with np.errstate(invalid="ignore"):
        rate = np.where(density > 0, s_count / density, np.nan)
    return density, rate


def grid_cat_cat(prim, succ, hp_x, hp_y, levels_x, levels_y):
    """(n_levels_y × n_levels_x) density and success-rate arrays."""
    nx, ny = len(levels_x), len(levels_y)
    xi = {l: i for i, l in enumerate(levels_x)}
    yi = {l: i for i, l in enumerate(levels_y)}

    density = np.zeros((ny, nx))
    s_count = np.zeros((ny, nx))

    for df, arr in [(prim, density), (succ, s_count)]:
        for xv, yv in zip(df[hp_x].values, df[hp_y].values):
            if xv in xi and yv in yi:
                arr[yi[yv], xi[xv]] += 1

    with np.errstate(invalid="ignore"):
        rate = np.where(density > 0, s_count / density, np.nan)
    return density, rate


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------

def _draw_pair(ax_d, ax_s, density, rate, xlabel, ylabel, xticks, yticks,
               xticklabels, yticklabels, title):
    """Draw density and success-rate panels for one HP pair."""
    kw = dict(origin="lower", aspect="auto", interpolation="nearest")

    im_d = ax_d.imshow(density, cmap="Blues", **kw)
    plt.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)

    masked = np.ma.masked_invalid(rate)
    im_s = ax_s.imshow(masked, cmap=SUCCESS_CMAP, vmin=0, vmax=1, **kw)
    plt.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04)

    for ax, subtitle in [(ax_d, "density"), (ax_s, "success rate")]:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=6, rotation=30, ha="right")
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=6)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(f"{title} — {subtitle}", fontsize=7)


def _page(pairs_data, page_title, task_display):
    """Render one PDF page for a list of (density, rate, meta) tuples."""
    n_pairs  = len(pairs_data)
    n_rows   = int(np.ceil(n_pairs / PAIRS_PER_ROW))
    n_cols   = PAIRS_PER_ROW * 2
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.8, n_rows * 2.6),
                             squeeze=False)

    for pair_idx, (density, rate, meta) in enumerate(pairs_data):
        row = pair_idx // PAIRS_PER_ROW
        col = (pair_idx % PAIRS_PER_ROW) * 2
        _draw_pair(axes[row][col], axes[row][col + 1],
                   density, rate, **meta)

    # Hide unused panels
    for pair_idx in range(n_pairs, n_rows * PAIRS_PER_ROW):
        row = pair_idx // PAIRS_PER_ROW
        col = (pair_idx % PAIRS_PER_ROW) * 2
        axes[row][col].set_visible(False)
        axes[row][col + 1].set_visible(False)

    fig.suptitle(f"{task_display} — {page_title}", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Per-task PDF
# ---------------------------------------------------------------------------

def save_task_pdf(task: str, df: pd.DataFrame, thresholds: dict, meta: dict):
    m          = meta[task]
    cont_names = m["cont_param_names"]
    cat_names  = m["cat_param_names"]
    cat_choices = m["cat_param_choices"]
    upper      = thresholds.get(task, {}).get("upper")

    prim = primary_df(df)
    succ = prim[prim["performance"] >= upper] if upper is not None else prim.iloc[0:0]

    display = "adding (ref)" if task == "adding" else task

    def cont_ticks():
        ticks = list(range(0, N_BINS, 2))
        labels = [f"{v / N_BINS:.2f}" for v in ticks]
        return ticks, labels

    out = FIGURES_DIR / f"heatmaps_{task}.pdf"
    with PdfPages(out) as pdf:

        # --- Page 1: Continuous × Continuous ----------------------------
        pairs_data = []
        for hp_x, hp_y in itertools.combinations(cont_names, 2):
            density, rate = grid_cont_cont(prim, succ, hp_x, hp_y)
            xt, xl = cont_ticks()
            yt, yl = cont_ticks()
            pairs_data.append((density, rate, dict(
                xlabel=HP_LABELS.get(hp_x, hp_x),
                ylabel=HP_LABELS.get(hp_y, hp_y),
                xticks=xt, yticks=yt,
                xticklabels=xl, yticklabels=yl,
                title=f"{HP_LABELS.get(hp_x,hp_x)} × {HP_LABELS.get(hp_y,hp_y)}",
            )))
        fig = _page(pairs_data, "Continuous × Continuous", display)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # --- Page 2: Categorical × Continuous ---------------------------
        pairs_data = []
        # Group by categorical HP so related plots are adjacent
        for cat_hp in cat_names:
            levels = cat_choices[cat_hp]
            for cont_hp in cont_names:
                density, rate = grid_cat_cont(prim, succ, cat_hp, cont_hp, levels)
                xt = list(range(len(levels)))
                yt, yl = cont_ticks()
                pairs_data.append((density, rate, dict(
                    xlabel=HP_LABELS.get(cat_hp, cat_hp),
                    ylabel=HP_LABELS.get(cont_hp, cont_hp),
                    xticks=xt, yticks=yt,
                    xticklabels=[str(l) for l in levels],
                    yticklabels=yl,
                    title=f"{HP_LABELS.get(cat_hp,cat_hp)} × {HP_LABELS.get(cont_hp,cont_hp)}",
                )))
        fig = _page(pairs_data, "Categorical × Continuous", display)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # --- Page 3: Categorical × Categorical --------------------------
        pairs_data = []
        for cat_x, cat_y in itertools.combinations(cat_names, 2):
            lx = cat_choices[cat_x]
            ly = cat_choices[cat_y]
            density, rate = grid_cat_cat(prim, succ, cat_x, cat_y, lx, ly)
            pairs_data.append((density, rate, dict(
                xlabel=HP_LABELS.get(cat_x, cat_x),
                ylabel=HP_LABELS.get(cat_y, cat_y),
                xticks=list(range(len(lx))),
                yticks=list(range(len(ly))),
                xticklabels=[str(l) for l in lx],
                yticklabels=[str(l) for l in ly],
                title=f"{HP_LABELS.get(cat_x,cat_x)} × {HP_LABELS.get(cat_y,cat_y)}",
            )))
        fig = _page(pairs_data, "Categorical × Categorical", display)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_adding_failed() -> pd.DataFrame:
    state_path = PRODUCTION_DIR / "adding_failed_run" / "bo_state.json"
    meta = task_meta()["adding"]
    cont_names = meta["cont_param_names"]
    rows = []
    for obs in json.load(open(state_path)):
        row = {
            "task":        "adding",
            "paradigm":    meta["paradigm"],
            "iteration":   obs["iteration"],
            "is_repeat":   obs.get("is_repeat", False),
            "performance": obs["performance"],
        }
        for k, v in obs["config"].items():
            row[k] = v
        unit_vals = obs.get("cont_unit_vals", [])
        for i, name in enumerate(cont_names):
            row[f"unit_{name}"] = unit_vals[i] if i < len(unit_vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    meta       = task_meta()
    thresholds = json.load(open(TABLES_DIR / "success_thresholds.json"))
    plot_tasks = [t for t in TASK_NAMES if t != "adding"] + ["adding"]

    for task in plot_tasks:
        if task == "adding":
            p = PRODUCTION_DIR / "adding_failed_run" / "bo_state.json"
            if not p.exists():
                print(f"  [skip] adding")
                continue
            df = load_adding_failed()
        else:
            p = PRODUCTION_DIR / task / "bo_state.json"
            if not p.exists():
                print(f"  [skip] {task}")
                continue
            df = load_task_df(task)

        out = save_task_pdf(task, df, thresholds, meta)
        print(f"  saved: {out}")


if __name__ == "__main__":
    main()
