"""
Visualize acquisition function, UCB, and N_eff slices over batch_size × hidden_size.

Outputs:
  acq_slice_relu.png    — 3 metrics × 4 snapshots for relu/adam slice
  acq_slice_tanh.png    — same for tanh/adam/init_scale=1 slice
  acq_compare_h.png     — acquisition at n=200, h ∈ {0.1, 0.15, 0.2}
  acq_compare_lambda.png — N_eff at n=200, lambda ∈ {0.1, 1.0}
  acq_compare_dist.png  — acquisition at n=200, Gower vs Euclidean

Usage:
    python plot_acquisition_slice_v2.py \
        [--state experiments_gp_test/spirals/bo_state.json]
"""
import argparse, json, math, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks import TASKS
from src.bo import (
    _cont_params_for_task, cat_params_for_task,
    build_XY, fit_gp, get_primary_observations,
    _cont_to_unit_val, ORDINAL_PARAMS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ord_to_unit(val, choices):
    lo, hi = float(choices[0]), float(choices[-1])
    if lo <= 0:
        return choices.index(val) / max(1, len(choices) - 1)
    return (math.log(float(val)) - math.log(lo)) / (math.log(hi) - math.log(lo))


def build_obs_matrix(obs_list, cont_params, cat_params):
    """Return (n_obs, n_dims) numpy array in unit space."""
    n_cont = len(cont_params)
    rows = []
    for o in obs_list:
        row = []
        # Continuous dims — use stored raw unit vals if available
        if o.get("cont_unit_vals"):
            row.extend(o["cont_unit_vals"][:n_cont])
        else:
            for name, lo, hi in cont_params:
                row.append(_cont_to_unit_val(o["config"][name], lo, hi))
        # Categorical dims
        for name, choices in cat_params:
            val = o["config"][name]
            if name in ORDINAL_PARAMS:
                row.append(ord_to_unit(val, choices))
            else:
                row.append(float(choices.index(val)))
        rows.append(row)
    return np.array(rows, dtype=np.float64)


def build_grid_matrix(bs_vals, hs_vals, fixed, cont_params, cat_params):
    """Return (n_bs*n_hs, n_dims) numpy array for the slice grid."""
    n_cont = len(cont_params)
    bs_idx = next(i for i, (n, *_) in enumerate(cont_params) if n == "batch_size")
    hs_idx = next(i for i, (n, *_) in enumerate(cont_params) if n == "hidden_size")

    fixed_cont = []
    for name, lo, hi in cont_params:
        if name in ("batch_size", "hidden_size"):
            fixed_cont.append(None)
        else:
            fixed_cont.append(_cont_to_unit_val(fixed[name], lo, hi))

    fixed_cat = []
    for name, choices in cat_params:
        val = fixed[name]
        if name in ORDINAL_PARAMS:
            fixed_cat.append(ord_to_unit(val, choices))
        else:
            fixed_cat.append(float(choices.index(val)))

    rows = []
    for bs in bs_vals:
        for hs in hs_vals:
            row = list(fixed_cont)
            row[bs_idx] = _cont_to_unit_val(bs, *next(
                (lo, hi) for n, lo, hi in cont_params if n == "batch_size"))
            row[hs_idx] = _cont_to_unit_val(hs, *next(
                (lo, hi) for n, lo, hi in cont_params if n == "hidden_size"))
            rows.append(row + fixed_cat)
    return np.array(rows, dtype=np.float64)


def compute_neff(grid_mat, obs_mat, cont_params, cat_params, h, lam=1.0, normalize=True):
    """N_eff at each grid point.  grid_mat: (n_grid, n_dims), obs_mat: (n_obs, n_dims)."""
    n_cont  = len(cont_params)
    n_dims  = len(cont_params) + len(cat_params)
    h2      = 2.0 * h * h

    g = grid_mat[:, :, np.newaxis]   # (n_grid, n_dims, 1)
    o = obs_mat[np.newaxis, :, :].transpose(0, 2, 1)  # (1, n_dims, n_obs)

    diff = g - o   # (n_grid, n_dims, n_obs)

    d2 = np.zeros((len(grid_mat), len(obs_mat)), dtype=np.float64)

    # Continuous + ordinal dims: squared diff
    for i, (name, *_) in enumerate(cont_params):
        d2 += diff[:, i, :] ** 2
    for j, (name, choices) in enumerate(cat_params):
        dim = n_cont + j
        if name in ORDINAL_PARAMS:
            d2 += diff[:, dim, :] ** 2
        else:
            d2 += lam * (diff[:, dim, :] != 0).astype(np.float64)

    if normalize:
        d2 /= n_dims

    return np.exp(-d2 / h2).sum(axis=1)   # (n_grid,)


def compute_ucb_sigma(gp, X_torch, beta):
    """Return (ucb, mean, sigma) as numpy arrays, shape (n_grid,)."""
    with torch.no_grad():
        post  = gp.posterior(X_torch)
        mean  = post.mean[:, 0, 0].numpy()
        sigma = post.variance[:, 0, 0].clamp(min=0).sqrt().numpy()
    ucb = mean + math.sqrt(beta) * sigma
    return ucb, mean, sigma


def slice_distances(obs_list, fixed, cont_params, cat_params):
    """Gower distance from each obs to the slice fixed-dim values. Returns (n_obs,) array."""
    # Only the fixed dims (not batch_size / hidden_size)
    fixed_cont = {n: _cont_to_unit_val(fixed[n], lo, hi)
                  for n, lo, hi in cont_params if n not in ("batch_size", "hidden_size")}
    n_fixed = len(fixed_cont) + len(cat_params)
    dists = []
    for o in obs_list:
        d2 = 0.0
        for name, u_fixed in fixed_cont.items():
            lo, hi = next((lo, hi) for n, lo, hi in cont_params if n == name)
            u_obs = _cont_to_unit_val(o["config"][name], lo, hi)
            d2 += (u_fixed - u_obs) ** 2
        for name, choices in cat_params:
            val_obs   = o["config"][name]
            val_fixed = fixed[name]
            if name in ORDINAL_PARAMS:
                d2 += (ord_to_unit(val_fixed, choices) - ord_to_unit(val_obs, choices)) ** 2
            else:
                d2 += float(val_fixed != val_obs)
        dists.append(math.sqrt(d2 / n_fixed))
    return np.array(dists)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def make_grid_vals(cont_params, n=80):
    bs_lo, bs_hi = next((lo, hi) for n_, lo, hi in cont_params if n_ == "batch_size")
    hs_lo, hs_hi = next((lo, hi) for n_, lo, hi in cont_params if n_ == "hidden_size")
    bs_vals = np.exp(np.linspace(math.log(bs_lo), math.log(bs_hi), n))
    hs_vals = np.exp(np.linspace(math.log(hs_lo), math.log(hs_hi), n))
    return bs_vals, hs_vals


def overlay_networks(ax, obs_list, fixed, cont_params, cat_params):
    dists = slice_distances(obs_list, fixed, cont_params, cat_params)
    alphas = np.clip(np.exp(-4.0 * dists), 0.05, 1.0)
    sizes  = 20 + 60 * np.exp(-4.0 * dists)
    hs_obs = np.array([o["config"]["hidden_size"] for o in obs_list])
    bs_obs = np.array([o["config"]["batch_size"]  for o in obs_list])
    for hs, bs, a, s in zip(hs_obs, bs_obs, alphas, sizes):
        ax.plot(hs, bs, "+", color="red", alpha=float(a), markersize=float(s)**0.5 * 1.5,
                markeredgewidth=1.0, zorder=5)


def plot_main_slice(primaries, cont_params, cat_params, gps, fixed, label, outfile,
                    snapshots=(50, 100, 150, None), h=0.15, beta=4.0, chance=0.0):
    bs_vals, hs_vals = make_grid_vals(cont_params)
    grid_mat = build_grid_matrix(bs_vals, hs_vals, fixed, cont_params, cat_params)
    X_torch  = torch.tensor(grid_mat, dtype=torch.double).unsqueeze(1)
    n_grid   = len(grid_mat)

    snap_obs = [primaries[:n] if n else primaries for n in snapshots]
    snap_gps = gps   # precomputed

    metrics = ["UCB", "N_eff", "Acquisition"]
    fig, axes = plt.subplots(len(metrics), len(snapshots),
                             figsize=(4.5 * len(snapshots), 4.0 * len(metrics)))
    fig.suptitle(f"Slice: {label}  [h={h}  β={beta}]", fontsize=12)

    for col, (obs, gp) in enumerate(zip(snap_obs, snap_gps)):
        obs_mat = build_obs_matrix(obs, cont_params, cat_params)
        ucb_flat, mean_flat, sigma_flat = compute_ucb_sigma(gp, X_torch, beta)
        neff_flat = compute_neff(grid_mat, obs_mat, cont_params, cat_params, h)
        acq_flat  = ucb_flat / (1.0 + neff_flat)

        n_label = f"n={len(obs)}"
        for row, (vals, metric) in enumerate(zip(
                [ucb_flat, neff_flat, acq_flat], metrics)):
            ax  = axes[row, col]
            mat = vals.reshape(len(bs_vals), len(hs_vals))
            im  = ax.pcolormesh(hs_vals, bs_vals, mat, cmap="viridis", shading="auto")
            plt.colorbar(im, ax=ax)
            overlay_networks(ax, obs, fixed, cont_params, cat_params)
            ax.set_xscale("log"); ax.set_yscale("log")
            if row == 0:   ax.set_title(n_label, fontsize=10)
            if col == 0:   ax.set_ylabel(f"{metric}\nbatch_size", fontsize=9)
            if row == len(metrics)-1: ax.set_xlabel("hidden_size", fontsize=9)

    plt.tight_layout()
    plt.savefig(outfile, dpi=130, bbox_inches="tight")
    print(f"Saved {outfile}")
    plt.close()


def plot_comparison_h(primaries, obs_mat, cont_params, cat_params, gp200,
                      fixed, outfile, hs_list=(0.1, 0.15, 0.2), beta=4.0):
    bs_vals, hs_vals = make_grid_vals(cont_params)
    grid_mat = build_grid_matrix(bs_vals, hs_vals, fixed, cont_params, cat_params)
    X_torch  = torch.tensor(grid_mat, dtype=torch.double).unsqueeze(1)
    ucb_flat, *_ = compute_ucb_sigma(gp200, X_torch, beta)

    fig, axes = plt.subplots(1, len(hs_list), figsize=(5 * len(hs_list), 4.5))
    fig.suptitle(f"Acquisition at n=200 — varying h  [{fixed['activation']} slice]", fontsize=11)
    for ax, h in zip(axes, hs_list):
        neff = compute_neff(grid_mat, obs_mat, cont_params, cat_params, h)
        acq  = (ucb_flat / (1.0 + neff)).reshape(len(bs_vals), len(hs_vals))
        im = ax.pcolormesh(hs_vals, bs_vals, acq, cmap="viridis", shading="auto")
        plt.colorbar(im, ax=ax)
        overlay_networks(ax, primaries, fixed, cont_params, cat_params)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"h={h}", fontsize=10)
        ax.set_xlabel("hidden_size"); ax.set_ylabel("batch_size")
    plt.tight_layout()
    plt.savefig(outfile, dpi=130, bbox_inches="tight")
    print(f"Saved {outfile}")
    plt.close()


def plot_comparison_lambda(primaries, obs_mat, cont_params, cat_params,
                           fixed, outfile, h=0.15, lambdas=(0.1, 1.0)):
    bs_vals, hs_vals = make_grid_vals(cont_params)
    grid_mat = build_grid_matrix(bs_vals, hs_vals, fixed, cont_params, cat_params)

    fig, axes = plt.subplots(1, len(lambdas), figsize=(5 * len(lambdas), 4.5))
    fig.suptitle(f"N_eff at n=200 — varying lambda (cat weight)  [h={h}]", fontsize=11)
    for ax, lam in zip(axes, lambdas):
        neff = compute_neff(grid_mat, obs_mat, cont_params, cat_params,
                            h, lam=lam).reshape(len(bs_vals), len(hs_vals))
        im = ax.pcolormesh(hs_vals, bs_vals, neff, cmap="plasma", shading="auto")
        plt.colorbar(im, ax=ax, label="N_eff")
        overlay_networks(ax, primaries, fixed, cont_params, cat_params)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"lambda={lam}", fontsize=10)
        ax.set_xlabel("hidden_size"); ax.set_ylabel("batch_size")
    plt.tight_layout()
    plt.savefig(outfile, dpi=130, bbox_inches="tight")
    print(f"Saved {outfile}")
    plt.close()


def plot_comparison_dist(primaries, obs_mat, cont_params, cat_params, gp200,
                         fixed, outfile, h=0.15, beta=4.0):
    bs_vals, hs_vals = make_grid_vals(cont_params)
    grid_mat = build_grid_matrix(bs_vals, hs_vals, fixed, cont_params, cat_params)
    X_torch  = torch.tensor(grid_mat, dtype=torch.double).unsqueeze(1)
    ucb_flat, *_ = compute_ucb_sigma(gp200, X_torch, beta)

    configs = [
        ("Gower (normalised)", True,  1.0),
        ("Euclidean (no norm)", False, 1.0),
    ]
    fig, axes = plt.subplots(1, len(configs), figsize=(5 * len(configs), 4.5))
    fig.suptitle(f"Acquisition at n=200 — distance metric  [h={h}]", fontsize=11)
    for ax, (title, normalize, lam) in zip(axes, configs):
        neff = compute_neff(grid_mat, obs_mat, cont_params, cat_params,
                            h, lam=lam, normalize=normalize)
        acq  = (ucb_flat / (1.0 + neff)).reshape(len(bs_vals), len(hs_vals))
        im = ax.pcolormesh(hs_vals, bs_vals, acq, cmap="viridis", shading="auto")
        plt.colorbar(im, ax=ax)
        overlay_networks(ax, primaries, fixed, cont_params, cat_params)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("hidden_size"); ax.set_ylabel("batch_size")
    plt.tight_layout()
    plt.savefig(outfile, dpi=130, bbox_inches="tight")
    print(f"Saved {outfile}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--state",   default="output/experiments_gp_test/spirals/bo_state.json")
    p.add_argument("--h",       type=float, default=0.15)
    p.add_argument("--beta",    type=float, default=4.0)
    return p.parse_args()


def main():
    args = parse_args()
    Path("output/figures").mkdir(parents=True, exist_ok=True)
    obs_all  = json.loads(Path(args.state).read_text())
    primaries = get_primary_observations(obs_all)
    print(f"Loaded {len(primaries)} primary observations")

    task        = TASKS["spirals"]()
    cont_params = _cont_params_for_task(task)
    cat_params  = cat_params_for_task(task)
    chance      = getattr(task, "chance_accuracy", 0.0)
    n_cont      = len(cont_params)

    slices = [
        ("relu", {
            "learning_rate": 1e-3, "l1_reg": 1e-6, "l2_reg": 1e-6,
            "depth": 2, "activation": "relu", "optimizer": "adam", "init_scale": 0.1,
        }),
        ("tanh", {
            "learning_rate": 1e-3, "l1_reg": 1e-6, "l2_reg": 1e-6,
            "depth": 2, "activation": "tanh", "optimizer": "adam", "init_scale": 1.0,
        }),
    ]

    snapshots = (50, 100, 150, None)  # None = all primaries

    # Pre-fit GPs at each snapshot (shared across slices)
    print("Fitting GPs...")
    snap_counts = [n if n else len(primaries) for n in snapshots]
    gps = []
    for n in snap_counts:
        obs = primaries[:n]
        X, Y = build_XY(obs, cont_params, cat_params, chance_accuracy=chance)
        gps.append(fit_gp(X, Y, n_cont))
        print(f"  GP fitted on {n} obs")

    gp200    = gps[-1]
    obs200   = primaries
    obs_mat  = build_obs_matrix(obs200, cont_params, cat_params)

    for slice_name, fixed in slices:
        snap_obs = [primaries[:n] if n else primaries for n in snapshots]

        plot_main_slice(
            primaries, cont_params, cat_params, gps, fixed,
            label=f"{fixed['activation']} / {fixed['optimizer']} / depth={fixed['depth']} / init_scale={fixed['init_scale']}",
            outfile=f"output/figures/acq_slice_{slice_name}.png",
            snapshots=snapshots, h=args.h, beta=args.beta, chance=chance,
        )

        plot_comparison_h(
            primaries, obs_mat, cont_params, cat_params, gp200,
            fixed, outfile=f"output/figures/acq_compare_h_{slice_name}.png",
            hs_list=(0.1, 0.15, 0.2), beta=args.beta,
        )

        plot_comparison_lambda(
            primaries, obs_mat, cont_params, cat_params,
            fixed, outfile=f"output/figures/acq_compare_lambda_{slice_name}.png", h=args.h,
        )

        plot_comparison_dist(
            primaries, obs_mat, cont_params, cat_params, gp200,
            fixed, outfile=f"output/figures/acq_compare_dist_{slice_name}.png",
            h=args.h, beta=args.beta,
        )


if __name__ == "__main__":
    main()
