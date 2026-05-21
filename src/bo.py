"""
Stratified round-robin Bayesian optimisation with MixedSingleTaskGP.

Design
------
The GP sees ALL observations with both continuous AND categorical dims encoded.
Round-robin over the task's categorical combos guarantees balanced coverage.

For each iteration:
  1. Pick the least-visited categorical combo (round-robin).
  2. If total observations < N_SOBOL: draw a Sobol point for the continuous dims.
  3. Otherwise: fit MixedSingleTaskGP on all data, fix the selected combo's
     categorical dims, optimise the acquisition over continuous dims only.

Input tensor layout (N × (N_CONT + N_CAT)):
  dims 0 .. N_CONT-1 : continuous, normalised to [0,1] via log-transform
  dims N_CONT ..      : categorical indices (float, treated as categorical by the GP)
"""

import json
import math
from itertools import product as iproduct
from pathlib import Path

import numpy as np
import torch
from botorch.acquisition import qUpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.optim import optimize_acqf_mixed
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

N_SOBOL = 20


# ---------------------------------------------------------------------------
# Per-task space helpers
# ---------------------------------------------------------------------------

def _cont_params_for_task(task):
    """Return list of (name, raw_lo, raw_hi) for continuous dims."""
    lo, hi = task.hidden_size_range
    return [
        ("hidden_size",   lo,   hi),
        ("learning_rate", 1e-5, 1e-1),
        ("l1_reg",        1e-6, 1e-1),
        ("l2_reg",        1e-6, 1e-1),
    ]


def cat_params_for_task(task):
    """Return ordered list of (name, choices) for categorical dims."""
    space = task.categorical_space()
    return [(name, choices) for name, choices in space.items()]


def _all_combos_for_task(task):
    cat_params = cat_params_for_task(task)
    return [
        dict(zip([n for n, _ in cat_params], vals))
        for vals in iproduct(*[choices for _, choices in cat_params])
    ]


def _make_bounds(cont_params, cat_params):
    lo, hi = [], []
    for _ in cont_params:
        lo.append(0.0)
        hi.append(1.0)
    for _, choices in cat_params:
        lo.append(0.0)
        hi.append(float(len(choices) - 1))
    return torch.tensor([lo, hi], dtype=torch.double)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _cont_to_unit(config, cont_params):
    row = []
    for name, raw_lo, raw_hi in cont_params:
        v  = math.log(config[name])
        lo = math.log(raw_lo)
        hi = math.log(raw_hi)
        row.append((v - lo) / (hi - lo))
    return row


def _unit_to_cont(unit_row, cont_params):
    result = {}
    for i, (name, raw_lo, raw_hi) in enumerate(cont_params):
        u = unit_row[i] if not hasattr(unit_row[i], 'item') else unit_row[i].item()
        u = float(np.clip(u, 0.0, 1.0))
        log_val = u * (math.log(raw_hi) - math.log(raw_lo)) + math.log(raw_lo)
        result[name] = math.exp(log_val)
    result["hidden_size"] = max(1, int(round(result["hidden_size"])))
    return result


def _cat_to_indices(config, cat_params):
    return [float(choices.index(config[name])) for name, choices in cat_params]


def encode_config(config, cont_params, cat_params):
    row = _cont_to_unit(config, cont_params) + _cat_to_indices(config, cat_params)
    return torch.tensor(row, dtype=torch.double)


def build_XY(observations, cont_params, cat_params):
    X = torch.stack([encode_config(o["config"], cont_params, cat_params)
                     for o in observations])
    Y = torch.tensor([[o["mean_metric"]] for o in observations], dtype=torch.double)
    return X, Y


# ---------------------------------------------------------------------------
# GP fitting
# ---------------------------------------------------------------------------

def fit_gp(X, Y, n_cont):
    cat_dims = list(range(n_cont, X.shape[1]))
    model = MixedSingleTaskGP(X, Y, cat_dims=cat_dims)
    mll   = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


# ---------------------------------------------------------------------------
# Acquisition optimisation
# ---------------------------------------------------------------------------

def _fixed_features_for_combo(combo, cat_params, n_cont):
    return {
        n_cont + j: float(choices.index(combo[name]))
        for j, (name, choices) in enumerate(cat_params)
    }


def suggest_continuous_for_combo(gp, combo, bounds, cat_params, n_cont, beta=8.0):
    acqf = qUpperConfidenceBound(model=gp, beta=beta)
    candidate, _ = optimize_acqf_mixed(
        acq_function        = acqf,
        bounds              = bounds,
        fixed_features_list = [_fixed_features_for_combo(combo, cat_params, n_cont)],
        q                   = 1,
        num_restarts        = 10,
        raw_samples         = 128,
    )
    unit_cont = candidate.squeeze(0)[:n_cont]
    return unit_cont


# ---------------------------------------------------------------------------
# Sobol fallback
# ---------------------------------------------------------------------------

def sobol_continuous(seed, n_cont):
    engine = SobolEngine(dimension=n_cont, scramble=True, seed=seed)
    u = engine.draw(1).double().squeeze(0)
    return u


# ---------------------------------------------------------------------------
# Round-robin combo selection
# ---------------------------------------------------------------------------

def _combo_key(combo):
    return json.dumps(combo, sort_keys=True)


def build_run_counts(observations, all_combos, cat_params):
    keys   = [_combo_key(c) for c in all_combos]
    counts = [0] * len(all_combos)
    for obs in observations:
        cat_dict = {name: obs["config"][name] for name, _ in cat_params}
        k = _combo_key(cat_dict)
        if k in keys:
            counts[keys.index(k)] += 1
    return counts


def next_combo(run_counts, all_combos):
    idx = min(range(len(all_combos)), key=lambda i: run_counts[i])
    return all_combos[idx], idx


# ---------------------------------------------------------------------------
# Top-level: suggest next full config
# ---------------------------------------------------------------------------

def suggest_next(observations, task, beta=8.0):
    """
    Select least-visited categorical combo, then:
      - Sobol sample for continuous dims if total obs < N_SOBOL
      - Otherwise: MixedSingleTaskGP suggestion with that combo fixed
    Returns (config dict, combo_idx, mode_str).
    """
    cont_params = _cont_params_for_task(task)
    cat_params  = cat_params_for_task(task)
    all_combos  = _all_combos_for_task(task)
    n_cont      = len(cont_params)
    bounds      = _make_bounds(cont_params, cat_params)

    run_counts          = build_run_counts(observations, all_combos, cat_params)
    combo, combo_idx    = next_combo(run_counts, all_combos)

    if len(observations) < N_SOBOL:
        u    = sobol_continuous(seed=len(observations), n_cont=n_cont)
        cont = _unit_to_cont(u, cont_params)
        mode = "sobol"
    else:
        X, Y = build_XY(observations, cont_params, cat_params)
        gp   = fit_gp(X, Y, n_cont)
        u    = suggest_continuous_for_combo(gp, combo, bounds, cat_params, n_cont, beta)
        cont = _unit_to_cont(u, cont_params)
        mode = "gp"

    config = {**combo, **cont}
    if "depth" in config:
        config["depth"] = int(config["depth"])
    if "batch_size" in config:
        config["batch_size"] = int(config["batch_size"])
    if "n_rnn_layers" in config:
        config["n_rnn_layers"] = int(config["n_rnn_layers"])
    return config, combo_idx, mode


def get_all_combos(task):
    return _all_combos_for_task(task)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def save_state(path, observations):
    def _default(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        raise TypeError(type(obj))
    with open(path, "w") as f:
        json.dump(observations, f, indent=2, default=_default)


def load_state(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return json.load(f)
