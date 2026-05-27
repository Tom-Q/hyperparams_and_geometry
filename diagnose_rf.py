"""
Replay the RF acquisition phase and report variance decomposition at each step.

For each RF iteration in bo_state.json, refit the RF and noise model on all
observations seen so far, then report var_total, var_aleatoric, var_epistemic,
and UCB at the config that was actually chosen.

Usage:
    .venv/bin/python diagnose_rf.py --state experiments_rf_test/spirals/bo_state.json --task spirals
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from tasks import TASKS
from src.bo import (
    _cont_params_for_task, cat_params_for_task,
    get_primary_observations, get_repeat_pairs,
    build_XY_rf, fit_rf, get_tree_predictions,
    fit_noise_model, _noise_features_single,
    N_SOBOL,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--task",  required=True)
    p.add_argument("--beta",  type=float, default=8.0)
    return p.parse_args()


def main():
    args   = parse_args()
    task   = TASKS[args.task]()
    cont_params = _cont_params_for_task(task)
    cat_params  = cat_params_for_task(task)

    with open(args.state) as f:
        all_obs = json.load(f)

    rf_primaries = [
        o for o in all_obs
        if not o.get("is_repeat", False) and o["iteration"] >= N_SOBOL
    ]

    if not rf_primaries:
        print("No RF-phase primary observations found.")
        return

    print(f"{'iter':>5}  {'n_obs':>5}  {'n_pairs':>7}  "
          f"{'var_total':>10}  {'var_aleat':>10}  {'var_epist':>10}  "
          f"{'ucb':>7}  {'mean':>7}  {'actual':>7}")
    print("-" * 85)

    for obs in rf_primaries:
        iteration = obs["iteration"]

        # Observations available when this suggestion was made
        obs_so_far = [o for o in all_obs if o["iteration"] < iteration]

        primary_so_far = get_primary_observations(obs_so_far)
        repeat_pairs   = get_repeat_pairs(obs_so_far)
        noise_coeffs   = fit_noise_model(repeat_pairs, cont_params, cat_params)

        X, Y = build_XY_rf(obs_so_far, cont_params, cat_params)
        rf   = fit_rf(X, Y)

        # Evaluate at the config that was actually chosen
        from src.bo import _cont_to_unit, _cat_to_indices
        row = _cont_to_unit(obs["config"], cont_params) + _cat_to_indices(obs["config"], cat_params)
        X_query = np.array([row])

        tree_preds = get_tree_predictions(rf, X_query)
        mean_pred  = float(tree_preds.mean())
        var_total  = float(tree_preds.var(ddof=1))

        if noise_coeffs is not None:
            feats        = _noise_features_single(obs["config"], cont_params, cat_params)
            var_aleat    = float(np.exp(feats @ noise_coeffs))
            var_epist    = max(0.0, var_total - var_aleat)
        else:
            var_aleat  = float("nan")
            var_epist  = var_total

        ucb = mean_pred + args.beta * np.sqrt(var_epist)

        print(f"{iteration:>5}  {len(obs_so_far):>5}  {len(repeat_pairs):>7}  "
              f"{var_total:>10.4f}  {var_aleat:>10.4f}  {var_epist:>10.4f}  "
              f"{ucb:>7.4f}  {mean_pred:>7.4f}  {obs['mean_metric']:>7.4f}")

    # Summary stats over RF phase
    print()
    print("Summary across RF iterations:")
    rows = []
    for obs in rf_primaries:
        iteration    = obs["iteration"]
        obs_so_far   = [o for o in all_obs if o["iteration"] < iteration]
        repeat_pairs = get_repeat_pairs(obs_so_far)
        noise_coeffs = fit_noise_model(repeat_pairs, cont_params, cat_params)
        X, Y = build_XY_rf(obs_so_far, cont_params, cat_params)
        rf   = fit_rf(X, Y)
        from src.bo import _cont_to_unit, _cat_to_indices
        row  = _cont_to_unit(obs["config"], cont_params) + _cat_to_indices(obs["config"], cat_params)
        tree_preds = get_tree_predictions(rf, np.array([row]))
        var_total  = float(tree_preds.var(ddof=1))
        if noise_coeffs is not None:
            feats     = _noise_features_single(obs["config"], cont_params, cat_params)
            var_aleat = float(np.exp(feats @ noise_coeffs))
            var_epist = max(0.0, var_total - var_aleat)
        else:
            var_aleat = float("nan")
            var_epist = var_total
        rows.append((var_total, var_aleat, var_epist))

    vt = [r[0] for r in rows]
    va = [r[1] for r in rows if not np.isnan(r[1])]
    ve = [r[2] for r in rows]

    print(f"  var_total  : mean={np.mean(vt):.4f}  median={np.median(vt):.4f}  min={np.min(vt):.4f}  max={np.max(vt):.4f}")
    if va:
        print(f"  var_aleat  : mean={np.mean(va):.4f}  median={np.median(va):.4f}  min={np.min(va):.4f}  max={np.max(va):.4f}")
    print(f"  var_epist  : mean={np.mean(ve):.4f}  median={np.median(ve):.4f}  min={np.min(ve):.4f}  max={np.max(ve):.4f}")
    n_zero = sum(1 for v in ve if v < 1e-6)
    print(f"  var_epist ~ 0 (< 1e-6): {n_zero}/{len(ve)} iterations")


if __name__ == "__main__":
    main()
