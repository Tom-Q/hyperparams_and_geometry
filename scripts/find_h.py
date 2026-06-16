#!/usr/bin/env python3
"""Find the N_eff bandwidth h for a given paradigm and total network count.

Criterion: p90 of Sobol-equivalent N_eff = 0.5 at N_total observations.

Because lam=1 and h is small, cross-combo N_eff contributions are negligible,
so each combo's continuous subspace is treated independently. N_total / n_combos
points are distributed uniformly (Sobol) within each combo's [0,1]^n_cont space.

Usage:
    python scripts/find_h.py                  # validate all paradigms at N=1000
    python scripts/find_h.py --paradigm rnn --n-total 200   # mnist_rnn case
"""
import argparse

import numpy as np
from scipy.stats import qmc

PARADIGMS = {
    "rl":         {"n_cont": 4, "n_combos": 24},
    "supervised": {"n_cont": 5, "n_combos": 24},
    "rnn":        {"n_cont": 5, "n_combos": 16},
}

KNOWN_H = {
    ("rl",         1000): 0.116,
    ("supervised", 1000): 0.160,
    ("rnn",        1000): 0.148,
}

N_EVAL  = 2000   # evaluation points per combo (more = smoother p90 estimate)
SEED    = 42


def sobol_points(n, d, seed=SEED):
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    return sampler.random(n)


def compute_p90_neff(n_cont, n_combos, n_total, h, n_eval=N_EVAL, seed=SEED):
    """Return p90 of N_eff across all combos under Sobol-uniform coverage."""
    n_obs = max(1, n_total // n_combos)   # observations per combo
    rng   = np.random.default_rng(seed)
    p90s  = []
    for combo_seed in rng.integers(0, 10_000, size=n_combos):
        obs  = sobol_points(n_obs,  n_cont, seed=int(combo_seed))
        eval_pts = sobol_points(n_eval, n_cont, seed=int(combo_seed) + 1)
        # N_eff(x) = sum_i exp(-||x - obs_i||^2 / 2h^2)
        diffs = eval_pts[:, None, :] - obs[None, :, :]   # (n_eval, n_obs, n_cont)
        sq_d  = (diffs ** 2).sum(axis=2)                  # (n_eval, n_obs)
        neff  = np.exp(-sq_d / (2 * h ** 2)).sum(axis=1)  # (n_eval,)
        p90s.append(np.percentile(neff, 90))
    return float(np.mean(p90s))


def find_h(n_cont, n_combos, n_total, target=0.5, tol=1e-4):
    """Binary search for h such that p90 N_eff = target."""
    lo, hi = 0.01, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        val = compute_p90_neff(n_cont, n_combos, n_total, mid)
        if val < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2, compute_p90_neff(n_cont, n_combos, n_total, (lo + hi) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paradigm", choices=list(PARADIGMS), default=None,
                    help="Paradigm to compute h for (default: validate all)")
    ap.add_argument("--n-total", type=int, default=1000,
                    help="Total number of networks (default: 1000)")
    ap.add_argument("--target", type=float, default=0.5,
                    help="Target p90 N_eff (default: 0.5)")
    args = ap.parse_args()

    if args.paradigm:
        paradigms_to_run = {args.paradigm: PARADIGMS[args.paradigm]}
    else:
        paradigms_to_run = PARADIGMS

    print(f"\n{'Paradigm':<12} {'N_total':>8} {'n_cont':>7} {'n_combos':>9} {'obs/combo':>10} {'h':>7} {'p90 N_eff':>10}")
    print("-" * 70)

    for name, cfg in paradigms_to_run.items():
        n_cont, n_combos = cfg["n_cont"], cfg["n_combos"]
        n_total = args.n_total
        h, p90 = find_h(n_cont, n_combos, n_total, target=args.target)
        known  = KNOWN_H.get((name, n_total))
        note   = ""
        if known is not None:
            note = f"  (expected {known:.3f}, diff {h - known:+.4f})"
        print(f"{name:<12} {n_total:>8} {n_cont:>7} {n_combos:>9} {n_total // n_combos:>10} {h:>7.3f} {p90:>10.4f}{note}")

    print()


if __name__ == "__main__":
    main()
