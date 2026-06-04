"""
RL acquisition slice visualiser.

Plots UCB, N_eff, and Acquisition over learning_rate × hidden_size (and
optionally l1_reg × learning_rate) at 4 GP snapshots (10 / 40 / 70 / 100
GP primaries, i.e. after 60 / 90 / 120 / 150 total primaries).

Generates (per task):
  acq_rl_<task>_hot_lr_hs.png     — lr × hs slice, best-visited categorical combo
  acq_rl_<task>_cold_lr_hs.png    — lr × hs slice, never-GP-visited categorical combo
  acq_rl_cartpole_hot_l1_lr.png   — l1 × lr slice, cartpole hot combo only

Usage:
    python scripts/plot_acquisition_rl.py
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tasks import TASKS
from src.bo import (
    _cont_params_for_task, cat_params_for_task,
    build_XY, fit_gp,
    _cont_to_unit_val,
)

BETA      = 4.0
H         = 0.15
N_GRID    = 60
N_SOBOL   = 50
SNAPSHOTS = (60, 90, 120, 150)   # total primary counts (sobol + GP)
OUTDIR    = ROOT / "output" / "figures"


# ---------------------------------------------------------------------------
# Grid and observation helpers
# ---------------------------------------------------------------------------

def log_mid(lo, hi):
    return math.exp((math.log(lo) + math.log(hi)) / 2)


def log_range(lo, hi, n):
    return np.exp(np.linspace(math.log(lo), math.log(hi), n))


def param_range(name, cont_params):
    return next((lo, hi) for n, lo, hi in cont_params if n == name)


def build_grid(ax1_name, ax2_name, ax1_vals, ax2_vals,
               fixed_cont, fixed_cat, cont_params, cat_params):
    """(n_ax1 * n_ax2, n_dims) matrix; ax1 is outer loop (x-axis)."""
    rows = []
    for v1 in ax1_vals:
        for v2 in ax2_vals:
            cont_row = [
                _cont_to_unit_val(v1 if n == ax1_name else
                                  v2 if n == ax2_name else
                                  fixed_cont[n], lo, hi)
                for n, lo, hi in cont_params
            ]
            cat_row = [float(choices.index(fixed_cat[n]))
                       for n, choices in cat_params]
            rows.append(cont_row + cat_row)
    return np.array(rows, dtype=np.float64)


def build_obs_matrix(obs_list, cont_params, cat_params):
    n_cont = len(cont_params)
    rows = []
    for o in obs_list:
        if o.get("cont_unit_vals") and len(o["cont_unit_vals"]) >= n_cont:
            cont_row = list(o["cont_unit_vals"][:n_cont])
        else:
            cont_row = [_cont_to_unit_val(o["config"][n], lo, hi)
                        for n, lo, hi in cont_params]
        cat_row = [float(choices.index(o["config"][n]))
                   for n, choices in cat_params]
        rows.append(cont_row + cat_row)
    return np.array(rows, dtype=np.float64)


def compute_neff(grid_mat, obs_mat, cont_params, cat_params, h, lam=1.0):
    n_cont = len(cont_params)
    h2 = 2.0 * h * h

    g    = grid_mat[:, :, np.newaxis]
    o    = obs_mat[np.newaxis, :, :].transpose(0, 2, 1)
    diff = g - o

    d2 = np.zeros((len(grid_mat), len(obs_mat)), dtype=np.float64)
    for i in range(n_cont):
        d2 += diff[:, i, :] ** 2
    for j in range(len(cat_params)):
        d2 += lam * (diff[:, n_cont + j, :] != 0).astype(np.float64)
    # no normalisation by n_dims — matches UCBoverNeff.forward in bo.py

    return np.exp(-d2 / h2).sum(axis=1)


# ---------------------------------------------------------------------------
# Global range computation
# ---------------------------------------------------------------------------

def global_ranges(snap_obs, snap_all_obs, all_combos, ax1_name, ax2_name, fixed_cont,
                  cont_params, cat_params, task):
    """Compute global (vmin, vmax) for UCB, N_eff, Acquisition across all combos.
    snap_obs: primary observations only (for N_eff).
    snap_all_obs: all observations including repeats (for GP fit).
    """
    chance = getattr(task, "chance_perf", 0.0)
    max_m  = getattr(task, "max_metric",  1.0)

    lo1, hi1 = param_range(ax1_name, cont_params)
    lo2, hi2 = param_range(ax2_name, cont_params)
    ax1_vals = log_range(lo1, hi1, N_GRID)
    ax2_vals = log_range(lo2, hi2, N_GRID)

    obs_mat = build_obs_matrix(snap_all_obs, cont_params, cat_params)
    X, Y    = build_XY(snap_all_obs, cont_params, cat_params,
                       chance_perf=chance, max_metric=max_m)
    gp      = fit_gp(X, Y, len(cont_params))

    all_ucb, all_neff, all_acq = [], [], []
    for combo in all_combos:
        fixed_cat_c = {n: combo[n] for n, _ in cat_params}
        gmat = build_grid(ax1_name, ax2_name, ax1_vals, ax2_vals,
                          fixed_cont, fixed_cat_c, cont_params, cat_params)
        Xt   = torch.tensor(gmat, dtype=torch.double).unsqueeze(1)
        neff = compute_neff(gmat, obs_mat, cont_params, cat_params, H)
        with torch.no_grad():
            post  = gp.posterior(Xt)
            mean  = post.mean[:, 0, 0].numpy()
            sigma = post.variance[:, 0, 0].clamp(min=0).sqrt().numpy()
        ucb = mean + math.sqrt(BETA) * sigma
        acq = ucb / (1.0 + neff)
        all_ucb.extend(ucb); all_neff.extend(neff); all_acq.extend(acq)

    return {
        "ucb": (min(all_ucb), max(all_ucb)),
        "neff": (min(all_neff), max(all_neff)),
        "acq": (min(all_acq), max(all_acq)),
    }


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_slice(primaries, all_obs, all_combos, cont_params, cat_params, task,
               ax1_name, ax2_name, fixed_cont, fixed_cat,
               title, outfile):
    """primaries: primary-only observations (for N_eff and snapshot counting).
    all_obs: all observations including repeats (for GP fit).
    """
    lo1, hi1 = param_range(ax1_name, cont_params)
    lo2, hi2 = param_range(ax2_name, cont_params)
    ax1_vals = log_range(lo1, hi1, N_GRID)
    ax2_vals = log_range(lo2, hi2, N_GRID)

    grid_mat = build_grid(ax1_name, ax2_name, ax1_vals, ax2_vals,
                          fixed_cont, fixed_cat, cont_params, cat_params)
    X_torch  = torch.tensor(grid_mat, dtype=torch.double).unsqueeze(1)

    chance = getattr(task, "chance_perf", 0.0)
    max_m  = getattr(task, "max_metric",  1.0)

    X_grid, Y_grid = np.meshgrid(ax1_vals, ax2_vals)

    fig, axes = plt.subplots(3, len(SNAPSHOTS),
                             figsize=(4.5 * len(SNAPSHOTS), 11))
    fig.suptitle(title, fontsize=11)

    # Build an iteration→index map to slice all_obs up to each snapshot
    primary_iters = [o["iteration"] for o in primaries]

    for col, n_snap in enumerate(SNAPSHOTS):
        snap_obs = primaries[:n_snap]
        # All obs (including repeats) up to the iteration of the last primary in snap
        last_iter = snap_obs[-1]["iteration"] if snap_obs else -1
        snap_all_obs = [o for o in all_obs if o["iteration"] <= last_iter]
        n_gp = n_snap - N_SOBOL

        # Global ranges across all combos for this snapshot
        ranges = global_ranges(snap_obs, snap_all_obs, all_combos, ax1_name, ax2_name,
                               fixed_cont, cont_params, cat_params, task)

        obs_mat = build_obs_matrix(snap_all_obs, cont_params, cat_params)
        neff    = compute_neff(grid_mat, obs_mat, cont_params, cat_params, H)

        X, Y = build_XY(snap_all_obs, cont_params, cat_params,
                        chance_perf=chance, max_metric=max_m)
        gp   = fit_gp(X, Y, len(cont_params))

        with torch.no_grad():
            post  = gp.posterior(X_torch)
            mean  = post.mean[:, 0, 0].numpy()
            sigma = post.variance[:, 0, 0].clamp(min=0).sqrt().numpy()
        ucb = mean + math.sqrt(BETA) * sigma
        acq = ucb / (1.0 + neff)

        def rg(arr):
            return arr.reshape(len(ax1_vals), len(ax2_vals)).T

        grids     = [rg(ucb), rg(neff), rg(acq)]
        labels    = ["UCB", "N_eff", "Acquisition (UCB / (1+N_eff))"]
        cb_labels = ["UCB (normalised metric)", "N_eff (effective obs count)", "Acquisition"]
        vmins     = [ranges["ucb"][0], ranges["neff"][0], ranges["acq"][0]]
        vmaxs     = [ranges["ucb"][1], ranges["neff"][1], ranges["acq"][1]]

        match_obs = [o for o in snap_obs
                     if all(o["config"].get(k) == v for k, v in fixed_cat.items())]
        mx = [o["config"][ax1_name] for o in match_obs]
        my = [o["config"][ax2_name] for o in match_obs]

        for row, (label, cb_label, grid, vmin, vmax) in enumerate(
                zip(labels, cb_labels, grids, vmins, vmaxs)):
            ax = axes[row, col]
            im = ax.pcolormesh(X_grid, Y_grid, grid, cmap="viridis",
                               shading="auto", vmin=vmin, vmax=vmax)
            ax.set_xscale("log")
            ax.set_yscale("log")
            if mx:
                ax.scatter(mx, my, c="red", s=18, zorder=5, alpha=0.75,
                           linewidths=0.5, edgecolors="white")
            ax.set_xlabel(ax1_name if row == 2 else "")
            ax.set_ylabel(ax2_name if col == 0 else "")
            if row == 0:
                ax.set_title(f"n_GP = {n_gp}")
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(cb_label, fontsize=7)

    plt.tight_layout()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {outfile.name}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    "fourrooms": {
        "hot":  {"depth": 1, "activation": "relu",    "optimizer": "adam", "init_scale": 0.1},
        "cold": {"depth": 1, "activation": "sigmoid", "optimizer": "sgd",  "init_scale": 0.1},
    },
    "cartpole": {
        "hot":  {"depth": 1, "activation": "tanh",    "optimizer": "adam", "init_scale": 1.0},
        "cold": {"depth": 1, "activation": "relu",    "optimizer": "sgd",  "init_scale": 0.1},
    },
}

for task_name, combos in TASK_CONFIGS.items():
    print(f"\n=== {task_name} ===")
    state_file = ROOT / "output/experiments_rl_epsilon_test" / task_name / "bo_state.json"
    all_obs   = json.load(open(state_file))
    primaries = [o for o in all_obs if not o.get("is_repeat")]

    task        = TASKS[task_name]()
    cont_params = _cont_params_for_task(task)
    cat_params  = cat_params_for_task(task)
    all_combos  = [dict(zip([n for n, _ in cat_params], vals))
                   for vals in __import__('itertools').product(*[c for _, c in cat_params])]

    log_mids = {n: log_mid(lo, hi) for n, lo, hi in cont_params}

    for combo_label, fixed_cat in combos.items():
        fixed_cont_lr_hs = {n: v for n, v in log_mids.items()
                            if n not in ("learning_rate", "hidden_size")}
        plot_slice(
            primaries, all_obs, all_combos, cont_params, cat_params, task,
            ax1_name="learning_rate", ax2_name="hidden_size",
            fixed_cont=fixed_cont_lr_hs, fixed_cat=fixed_cat,
            title=(f"{task_name}  |  lr × hidden_size  |  {combo_label}: "
                   + ", ".join(f"{k}={v}" for k, v in fixed_cat.items())),
            outfile=OUTDIR / f"acq_rl_{task_name}_{combo_label}_lr_hs.png",
        )

    if task_name == "cartpole":
        fixed_cont_l1_lr = {n: v for n, v in log_mids.items()
                            if n not in ("l1_reg", "learning_rate")}
        plot_slice(
            primaries, all_obs, all_combos, cont_params, cat_params, task,
            ax1_name="l1_reg", ax2_name="learning_rate",
            fixed_cont=fixed_cont_l1_lr, fixed_cat=combos["hot"],
            title=(f"cartpole  |  l1_reg × lr  |  hot: "
                   + ", ".join(f"{k}={v}" for k, v in combos["hot"].items())),
            outfile=OUTDIR / "acq_rl_cartpole_hot_l1_lr.png",
        )

print("\nDone.")
