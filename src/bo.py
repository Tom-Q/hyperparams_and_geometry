"""
Saturating Bayesian optimisation with MixedSingleTaskGP.

Acquisition (GP phase):
    A(x) = [μ(x) + sqrt(β)·σ(x)] / (1 + N_eff(x))

N_eff measures the effective local sample density; once a region saturates the
acquisition value drops and the optimiser naturally moves elsewhere.

GP target: normalised accuracy y = (raw - chance) / (1 - chance), clamped [0, 1].

Input tensor layout (N × (N_CONT + N_CAT)):
  dims 0 .. N_CONT-1 : continuous, log-normalised to [0, 1]
  dims N_CONT ..      : categorical indices (float, treated as categorical by the GP)
"""

import json
import math
from itertools import product as iproduct
from pathlib import Path

import numpy as np
import torch
from botorch.acquisition import AnalyticAcquisitionFunction, qUpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

N_SOBOL = 100

# Params promoted from categorical to continuous (treated as log-continuous by the GP).
# They are filtered out of cat_params and added to cont_params automatically.
CONT_FROM_CAT = {"hidden_size", "batch_size"}

# Remaining ordinal categoricals (numeric, log-scale ordering in N_eff).
ORDINAL_PARAMS = {"depth", "n_rnn_layers"}


# ---------------------------------------------------------------------------
# Per-task space helpers
# ---------------------------------------------------------------------------

def _cont_params_for_task(task):
    """Return list of (name, raw_lo, raw_hi) for continuous dims.

    Includes learning_rate, l1_reg, l2_reg, plus any CONT_FROM_CAT params
    present in the task's categorical_space (hidden_size, batch_size).
    """
    l1_hi = getattr(task, "l1_range_hi", 1e-2)
    l2_hi = getattr(task, "l2_range_hi", 1e-2)
    params = [
        ("learning_rate", 1e-5, 1e-1),
        ("l1_reg",        1e-6, l1_hi),
        ("l2_reg",        1e-6, l2_hi),
    ]
    cat_space = task.categorical_space()
    if "hidden_size" in cat_space:
        choices = cat_space["hidden_size"]
        params.append(("hidden_size", float(min(choices)), float(max(choices))))
    if "batch_size" in cat_space:
        choices = cat_space["batch_size"]
        params.append(("batch_size", float(min(choices)), float(max(choices))))
    return params


def cat_params_for_task(task):
    """Return ordered list of (name, choices) for categorical dims.

    Params in CONT_FROM_CAT are excluded — they are handled as continuous.
    """
    space = task.categorical_space()
    return [(name, choices) for name, choices in space.items()
            if name not in CONT_FROM_CAT]


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

def _cont_to_unit_val(value, raw_lo, raw_hi):
    """Map a single continuous value to [0, 1] on log scale."""
    return (math.log(value) - math.log(raw_lo)) / (math.log(raw_hi) - math.log(raw_lo))


def _cont_to_unit(config, cont_params):
    return [_cont_to_unit_val(config[name], raw_lo, raw_hi)
            for name, raw_lo, raw_hi in cont_params]


def _unit_to_cont(unit_row, cont_params):
    result = {}
    for i, (name, raw_lo, raw_hi) in enumerate(cont_params):
        u = unit_row[i].item() if hasattr(unit_row[i], 'item') else unit_row[i]
        u = float(np.clip(u, 0.0, 1.0))
        log_val = u * (math.log(raw_hi) - math.log(raw_lo)) + math.log(raw_lo)
        val = math.exp(log_val)
        if name in CONT_FROM_CAT:
            val = max(1, round(val))
        result[name] = val
    return result


def _cat_to_indices(config, cat_params):
    return [float(choices.index(config[name])) for name, choices in cat_params]


def _ord_to_unit(value, choices):
    """Log-normalise an ordinal categorical value to [0, 1]."""
    if len(choices) == 1:
        return 0.0
    try:
        logs = [math.log(c) for c in choices]
        lo, hi = logs[0], logs[-1]
        if hi == lo:
            return 0.0
        return (math.log(value) - lo) / (hi - lo)
    except (TypeError, ValueError):
        idx = choices.index(value)
        return idx / (len(choices) - 1)


def encode_config(obs, cont_params, cat_params):
    """Encode an observation dict to a unit-space tensor.

    Uses obs["cont_unit_vals"] (raw optimizer/Sobol u values) when present,
    so the GP sees the actual explored point rather than the snapped integer value.
    Falls back to recomputing from obs["config"] for legacy observations.
    """
    cont_row = obs.get("cont_unit_vals") or _cont_to_unit(obs["config"], cont_params)
    cat_row  = _cat_to_indices(obs["config"], cat_params)
    return torch.tensor(cont_row + cat_row, dtype=torch.double)


def get_primary_observations(observations):
    return [o for o in observations if not o.get("is_repeat", False)]


# ---------------------------------------------------------------------------
# Accuracy normalisation
# ---------------------------------------------------------------------------

def _normalise_metric(raw, chance_accuracy):
    """Normalise raw metric to [0, 1] relative to chance. Clamped."""
    denom = max(1e-6, 1.0 - chance_accuracy)
    return float(np.clip((raw - chance_accuracy) / denom, 0.0, 1.0))


def build_XY(observations, cont_params, cat_params, chance_accuracy=0.0):
    X = torch.stack([encode_config(o, cont_params, cat_params)
                     for o in observations])
    Y = torch.tensor(
        [[_normalise_metric(o["mean_metric"], chance_accuracy)] for o in observations],
        dtype=torch.double,
    )
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
# N_eff: effective local sample count
# ---------------------------------------------------------------------------

def compute_n_eff(x_query, observations, cont_params, cat_params, h=0.2):
    """
    Compute N_eff(x_query) = Σ_i K(x_query, x_i) over primary observations,
    using Gower distance: each dimension contributes a value in [0, 1] and the
    total is averaged over all dimensions.

    Continuous/ordinal dims: squared unit distance.
    Unordered categorical dims: binary (0 same, 1 different).

    x_query: config dict. Repeat observations are excluded.
    """
    if not observations:
        return 0.0
    n_dims = len(cont_params) + len(cat_params)
    h2     = 2.0 * h * h
    total  = 0.0
    for obs in observations:
        if obs.get("is_repeat"):
            continue
        xi = obs["config"]
        d2 = 0.0
        for name, raw_lo, raw_hi in cont_params:
            u  = _cont_to_unit_val(x_query[name], raw_lo, raw_hi)
            ui = _cont_to_unit_val(xi[name],      raw_lo, raw_hi)
            d2 += (u - ui) ** 2
        for name, choices in cat_params:
            if name in ORDINAL_PARAMS:
                o  = _ord_to_unit(x_query[name], choices)
                oi = _ord_to_unit(xi[name],      choices)
                d2 += (o - oi) ** 2
            else:
                d2 += float(x_query[name] != xi[name])
        total += math.exp(-(d2 / n_dims) / h2)
    return total


# ---------------------------------------------------------------------------
# Sobol fallback
# ---------------------------------------------------------------------------

def sobol_continuous(seed, n_cont):
    engine = SobolEngine(dimension=n_cont, scramble=True, seed=seed)
    return engine.draw(1).double().squeeze(0)


# ---------------------------------------------------------------------------
# Round-robin combo selection (used during Sobol phase)
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


def next_combo(run_counts, all_combos, rng=None):
    min_count = min(run_counts)
    tied = [i for i, c in enumerate(run_counts) if c == min_count]
    if rng is None:
        rng = np.random.default_rng()
    idx = int(rng.choice(tied))
    return all_combos[idx], idx


# ---------------------------------------------------------------------------
# UCB / (1 + N_eff) acquisition function (differentiable, for gradient optimisation)
# ---------------------------------------------------------------------------

class UCBoverNeff(AnalyticAcquisitionFunction):
    """
    A(x) = [μ(x) + sqrt(β)·σ(x)] / (1 + N_eff(x))

    Computes the full Gower distance between candidate x and every primary
    observation directly from x — no precomputed per-combo quantities needed.
    Safe to use with optimize_acqf_mixed across all categorical combos at once.

    Continuous and ordinal dims: squared unit distance.
    Unordered categorical dims: binary (0 same, 1 different).
    All contributions divided by n_dims (Gower normalisation).
    """

    def __init__(self, model, beta, primary_observations, cont_params, cat_params, h):
        super().__init__(model=model)
        self.register_buffer("sqrt_beta", torch.tensor(beta ** 0.5, dtype=torch.double))
        self.h2     = 2.0 * h * h
        self.n_cont = len(cont_params)
        self.n_dims = len(cont_params) + len(cat_params)

        # Continuous obs unit values: (n_obs, n_cont)
        # Use stored raw unit values when available (avoids snapping back to integer grid).
        def _cont_units(o):
            if "cont_unit_vals" in o:
                return o["cont_unit_vals"][:len(cont_params)]
            return [_cont_to_unit_val(o["config"][name], lo, hi) for name, lo, hi in cont_params]
        self.register_buffer("obs_cont", torch.tensor(
            [_cont_units(o) for o in primary_observations], dtype=torch.double))

        # For each cat dim: store obs values and (for ordinal) a unit-value lookup table.
        ord_obs_cols, unord_obs_cols = [], []
        self.ord_cat_dims   = []   # positions within cat_params
        self.unord_cat_dims = []

        for j, (name, choices) in enumerate(cat_params):
            obs_ints = [choices.index(o["config"][name]) for o in primary_observations]
            if name in ORDINAL_PARAMS:
                self.ord_cat_dims.append(j)
                k = len(self.ord_cat_dims) - 1
                lut = torch.tensor([_ord_to_unit(c, choices) for c in choices], dtype=torch.double)
                self.register_buffer(f"_ord_lut_{k}", lut)
                ord_obs_cols.append(torch.tensor([lut[i].item() for i in obs_ints], dtype=torch.double))
            else:
                self.unord_cat_dims.append(j)
                unord_obs_cols.append(torch.tensor(obs_ints, dtype=torch.double))

        if ord_obs_cols:
            self.register_buffer("obs_ord",   torch.stack(ord_obs_cols,   dim=1))  # (n_obs, n_ord)
        if unord_obs_cols:
            self.register_buffer("obs_unord", torch.stack(unord_obs_cols, dim=1))  # (n_obs, n_unord)

    def forward(self, X):
        # X: (*batch, 1, n_dims)
        mean, sigma = self._mean_and_sigma(X)
        mean  = mean.squeeze(-1)
        sigma = sigma.squeeze(-1)
        ucb   = mean + self.sqrt_beta * sigma

        x = X[..., 0, :]  # (*batch, n_dims)

        # Continuous: squared distance in unit space — (*batch, n_obs)
        x_cont  = x[..., :self.n_cont]
        d2 = ((x_cont.unsqueeze(-2) - self.obs_cont) ** 2).sum(-1)

        # Ordinal: squared distance between unit values (integer index → unit via LUT)
        for k, j in enumerate(self.ord_cat_dims):
            lut    = getattr(self, f"_ord_lut_{k}")             # (n_choices,)
            x_unit = lut[x[..., self.n_cont + j].long()]       # (*batch,)
            d2     = d2 + (x_unit.unsqueeze(-1) - self.obs_ord[:, k]) ** 2

        # Unordered: binary (detached — categorical dims are fixed during opt)
        for k, j in enumerate(self.unord_cat_dims):
            x_idx  = x[..., self.n_cont + j]                   # (*batch,)
            binary = (x_idx.unsqueeze(-1).round() != self.obs_unord[:, k].round()).float().detach()
            d2     = d2 + binary

        n_eff = torch.exp(-(d2 / self.n_dims) / self.h2).sum(-1)  # (*batch,)
        return ucb / (1.0 + n_eff)


# ---------------------------------------------------------------------------
# Saturating acquisition: optimise UCBoverNeff jointly over all combos
# ---------------------------------------------------------------------------

def _suggest_saturating(gp, primary_observations, cont_params, cat_params, beta, h=0.2):
    """
    Build one UCBoverNeff acquisition function, then call optimize_acqf_mixed
    with the full list of categorical combos.  BoTorch handles the joint
    optimisation — one batched GP evaluation for all combos, gradient ascent
    over continuous dims with categoricals fixed per combo.

    Returns the continuous unit row and combo dict of the global argmax.
    """
    n_cont = len(cont_params)
    n_dims = n_cont + len(cat_params)

    all_combos = [
        dict(zip([nm for nm, _ in cat_params], vals))
        for vals in iproduct(*[ch for _, ch in cat_params])
    ]

    acqf = UCBoverNeff(gp, beta, primary_observations, cont_params, cat_params, h)

    bounds = torch.zeros(2, n_dims, dtype=torch.double)
    bounds[1, :n_cont] = 1.0
    for j, (_, choices) in enumerate(cat_params):
        bounds[1, n_cont + j] = float(len(choices) - 1)

    fixed_features_list = [
        {n_cont + j: float(choices.index(combo[name])) for j, (name, choices) in enumerate(cat_params)}
        for combo in all_combos
    ]

    candidate, _ = optimize_acqf_mixed(
        acq_function       = acqf,
        bounds             = bounds,
        q                  = 1,
        num_restarts       = 3,
        raw_samples        = 32,
        fixed_features_list = fixed_features_list,
        options            = {"maxiter": 20},
    )

    best_cont = candidate.squeeze(0)[:n_cont].detach()
    cat_idx   = candidate.squeeze(0)[n_cont:].round().long().tolist()
    best_combo = {name: choices[cat_idx[j]] for j, (name, choices) in enumerate(cat_params)}

    return best_cont, best_combo


# ---------------------------------------------------------------------------
# Top-level: suggest next full config
# ---------------------------------------------------------------------------

def suggest_next(observations, task, beta=4.0, h=0.2):
    """
    Sobol phase (n_primary < N_SOBOL):
        Round-robin over all combos, quasi-random continuous dims.

    GP phase (n_primary >= N_SOBOL):
        Fit GP on normalised accuracy, evaluate A(x) = UCB(x) / (1 + N_eff(x))
        on a Sobol grid of candidates across all combos, return the argmax.

    Returns (config dict, combo_idx_in_all_combos, mode_str, cont_unit_vals).

    cont_unit_vals is a list of floats — the raw [0,1] unit values for the
    continuous dims before rounding. Callers should store this in the
    observation dict so the GP sees the actual explored point, not the
    snapped integer value.
    """
    cont_params = _cont_params_for_task(task)
    cat_params  = cat_params_for_task(task)
    all_combos  = _all_combos_for_task(task)
    n_cont      = len(cont_params)
    chance      = getattr(task, "chance_accuracy", 0.0)

    primary_obs = get_primary_observations(observations)
    n_primary   = len(primary_obs)
    rng         = np.random.default_rng(n_primary)

    if n_primary < N_SOBOL:
        run_counts       = build_run_counts(primary_obs, all_combos, cat_params)
        combo, combo_idx = next_combo(run_counts, all_combos, rng)
        u    = sobol_continuous(seed=n_primary, n_cont=n_cont)
        cont = _unit_to_cont(u, cont_params)
        cont_unit_vals = [float(v) for v in u]
        mode = "sobol"
    else:
        X, Y = build_XY(observations, cont_params, cat_params, chance_accuracy=chance)
        gp   = fit_gp(X, Y, n_cont)

        best_unit, best_combo = _suggest_saturating(
            gp, primary_obs, cont_params, cat_params, beta, h=h,
        )
        combo     = best_combo
        combo_idx = next(i for i, c in enumerate(all_combos)
                         if _combo_key(c) == _combo_key(combo))
        cont = _unit_to_cont(best_unit, cont_params)
        cont_unit_vals = [float(v) for v in best_unit]
        mode = "gp-saturating"

    config = {**combo, **cont}
    if "depth"        in config: config["depth"]        = int(config["depth"])
    if "n_rnn_layers" in config: config["n_rnn_layers"] = int(config["n_rnn_layers"])
    return config, combo_idx, mode, cont_unit_vals


def get_all_combos(task):
    return _all_combos_for_task(task)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def save_state(path, observations, s3_bucket=None, task_name=None):
    def _default(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        raise TypeError(type(obj))
    with open(path, "w") as f:
        json.dump(observations, f, indent=2, default=_default)
    if s3_bucket and task_name:
        import boto3
        boto3.client("s3").upload_file(str(path), s3_bucket, f"{task_name}/bo_state.json")


def load_state(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Legacy helpers (used by run_spirals_culling_test*.py; not part of main path)
# ---------------------------------------------------------------------------

def _make_bounds_legacy(cont_params, cat_params):
    return _make_bounds(cont_params, cat_params)


def suggest_continuous_for_combo(gp, combo, bounds, cat_params, n_cont, beta=8.0):
    acqf = qUpperConfidenceBound(model=gp, beta=beta)
    from botorch.optim import optimize_acqf_mixed
    candidate, _ = optimize_acqf_mixed(
        acq_function        = acqf,
        bounds              = bounds,
        fixed_features_list = [_fixed_features_for_combo(combo, cat_params, n_cont)],
        q                   = 1,
        num_restarts        = 10,
        raw_samples         = 128,
    )
    return candidate.squeeze(0)[:n_cont]


def _fixed_features_for_combo(combo, cat_params, n_cont):
    return {
        n_cont + j: float(choices.index(combo[name]))
        for j, (name, choices) in enumerate(cat_params)
    }


def _combo_ucb_max(gp, combo, cont_params, cat_params, beta, n_candidates=1000):
    """Max UCB over a Sobol grid for a fixed combo (legacy culling scripts)."""
    n_cont = len(cont_params)
    engine = SobolEngine(dimension=n_cont, scramble=True, seed=0)
    unit_cont = engine.draw(n_candidates).double()

    cat_indices = torch.tensor(
        [float(choices.index(combo[name])) for name, choices in cat_params],
        dtype=torch.double,
    )
    cat_part = cat_indices.unsqueeze(0).expand(n_candidates, -1)
    X_cand = torch.cat([unit_cont, cat_part], dim=1)

    with torch.no_grad():
        posterior = gp.posterior(X_cand)
        mean      = posterior.mean.squeeze(-1)
        variance  = posterior.variance.squeeze(-1).clamp_min(0)
        ucb       = mean + math.sqrt(beta) * variance.sqrt()

    return float(ucb.max())


def get_active_combos(gp, all_combos, cont_params, cat_params, success_threshold, beta):
    """Legacy: return combos whose UCB upper bound exceeds success_threshold."""
    active = [
        c for c in all_combos
        if _combo_ucb_max(gp, c, cont_params, cat_params, beta) >= success_threshold
    ]
    return active if active else all_combos
