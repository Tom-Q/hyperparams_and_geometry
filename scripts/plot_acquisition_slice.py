"""
Plot 2D slices of the UCBoverNeff acquisition function over batch_size × hidden_size,
at two snapshots: immediately post-Sobol (100 obs) and after GP phase.

Other dims fixed at: relu / adam / depth=2 / init_scale=0.1 / lr=1e-3 / l1=l2=1e-6
Trained networks overlaid as crosses.

Usage:
    python plot_acquisition_slice.py [--state experiments_gp_test/spirals/bo_state.json]
                                     [--n-sobol 100] [--h 0.15] [--beta 4.0]
                                     [--out acquisition_slice.png]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks import TASKS
from src.bo import (
    _cont_params_for_task, cat_params_for_task,
    build_XY, fit_gp, UCBoverNeff,
    get_primary_observations, _cont_to_unit_val,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--state",    default="output/experiments_gp_test/spirals/bo_state.json")
    p.add_argument("--n-sobol", type=int,   default=100)
    p.add_argument("--h",       type=float, default=0.15)
    p.add_argument("--beta",    type=float, default=4.0)
    p.add_argument("--out",     default="output/figures/acquisition_slice.png")
    return p.parse_args()


def make_grid(cont_params, cat_params, fixed, n_bs=80, n_hs=80):
    """Build a (n_bs*n_hs, 1, n_dims) tensor sweeping over batch_size and hidden_size."""
    bs_lo = dict(cont_params)["batch_size"][0] if False else None  # handled below
    for name, lo, hi in cont_params:
        if name == "batch_size":
            bs_lo, bs_hi = lo, hi
        if name == "hidden_size":
            hs_lo, hs_hi = lo, hi

    bs_vals = np.exp(np.linspace(math.log(bs_lo), math.log(bs_hi), n_bs))
    hs_vals = np.exp(np.linspace(math.log(hs_lo), math.log(hs_hi), n_hs))

    # Unit values for fixed dims
    fixed_cont = []
    for name, lo, hi in cont_params:
        if name in ("batch_size", "hidden_size"):
            fixed_cont.append(None)   # placeholder, filled per grid point
        else:
            fixed_cont.append(_cont_to_unit_val(fixed[name], lo, hi))

    fixed_cat = []
    for name, choices in cat_params:
        fixed_cat.append(float(choices.index(fixed[name])))

    bs_idx = next(i for i, (n, *_) in enumerate(cont_params) if n == "batch_size")
    hs_idx = next(i for i, (n, *_) in enumerate(cont_params) if n == "hidden_size")

    rows = []
    for bs in bs_vals:
        for hs in hs_vals:
            row = list(fixed_cont)
            row[bs_idx] = _cont_to_unit_val(bs, bs_lo, bs_hi)
            row[hs_idx] = _cont_to_unit_val(hs, hs_lo, hs_hi)
            rows.append(row + fixed_cat)

    X = torch.tensor(rows, dtype=torch.double).unsqueeze(1)  # (n_bs*n_hs, 1, n_dims)
    return X, bs_vals, hs_vals


def main():
    args = parse_args()
    Path("output/figures").mkdir(parents=True, exist_ok=True)

    state_path = Path(args.state)
    if not state_path.exists():
        sys.exit(f"State file not found: {state_path}")

    obs_all  = json.loads(state_path.read_text())
    primaries = get_primary_observations(obs_all)
    print(f"Loaded {len(primaries)} primary observations")

    task        = TASKS["spirals"]()
    cont_params = _cont_params_for_task(task)
    cat_params  = cat_params_for_task(task)
    chance      = getattr(task, "chance_perf", 0.0)
    n_cont      = len(cont_params)

    fixed = {
        "learning_rate": 1e-3,
        "l1_reg":        1e-6,
        "l2_reg":        1e-6,
        "depth":         2,
        "activation":    "relu",
        "optimizer":     "adam",
        "init_scale":    0.1,
    }

    sobol_obs = primaries[:args.n_sobol]
    gp_obs    = primaries

    snapshots = [
        (f"post-Sobol  (n={len(sobol_obs)})", sobol_obs),
        (f"post-GP     (n={len(gp_obs)})",    gp_obs),
    ]

    X_grid, bs_vals, hs_vals = make_grid(cont_params, cat_params, fixed)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Acquisition slice: batch_size × hidden_size\n"
        f"fixed: relu / adam / depth=2 / init_scale=0.1 / lr=1e-3 / l1=l2=1e-6  "
        f"[h={args.h}  β={args.beta}]",
        fontsize=11,
    )

    vmin, vmax = None, None

    acq_maps = []
    for label, obs in snapshots:
        X, Y = build_XY(obs, cont_params, cat_params, chance_perf=chance)
        gp   = fit_gp(X, Y, n_cont)
        acqf = UCBoverNeff(gp, args.beta, obs, cont_params, cat_params, args.h)
        with torch.no_grad():
            vals = acqf(X_grid).detach().numpy()
        acq_map = vals.reshape(len(bs_vals), len(hs_vals))
        acq_maps.append((label, obs, acq_map))
        vmin = acq_map.min() if vmin is None else min(vmin, acq_map.min())
        vmax = acq_map.max() if vmax is None else max(vmax, acq_map.max())

    for ax, (label, obs, acq_map) in zip(axes, acq_maps):
        im = ax.pcolormesh(hs_vals, bs_vals, acq_map,
                           cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
        plt.colorbar(im, ax=ax, label="acquisition value")

        hs_obs = [o["config"]["hidden_size"] for o in obs]
        bs_obs = [o["config"]["batch_size"]  for o in obs]
        ax.scatter(hs_obs, bs_obs, marker="+", c="red", s=60,
                   linewidths=1.2, alpha=0.6, zorder=5, label="trained networks")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("hidden_size", fontsize=11)
        ax.set_ylabel("batch_size",  fontsize=11)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
