"""
Stratified round-robin Bayesian optimisation with Random Forest surrogate.

Design
------
Round-robin over the task's categorical combos guarantees balanced coverage.
The RF sees ALL observations (primary + repeats); only primary observations
count toward round-robin and the Sobol threshold.

Every other iteration (in run_bo.py) repeats the most recent primary config
to give a direct local noise estimate: (y1-y2)^2/2 ~ sigma^2_aleatoric.
A log-linear OLS model generalises these estimates to any config.

Epistemic uncertainty = max(0, between-tree variance - aleatoric estimate).
Acquisition: UCB = mean + beta * sqrt(sigma^2_epistemic), maximised by
evaluating ~10,000 Sobol candidates per combo and returning the argmax.

Input encoding (same for Sobol and RF):
  dims 0..N_CONT-1 : continuous, log-transformed and normalised to [0,1]
  dims N_CONT..    : categorical indices (float)
"""

import json
import math
from itertools import product as iproduct
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from torch.quasirandom import SobolEngine

N_SOBOL         = 100     # total observations (primary + repeats) before switching to RF
N_RF_CANDIDATES = 50_000  # acquisition grid size
MIN_NOISE_PAIRS = 5       # repeat pairs needed before fitting noise model

# Continuous and categorical dimensions used as noise model features.
# Update here if parameters are added or renamed.
NOISE_CONT_FEATURES = ["learning_rate", "hidden_size"]
NOISE_CAT_FEATURES  = ["batch_size"]


# ---------------------------------------------------------------------------
# Per-task space helpers
# ---------------------------------------------------------------------------

def _cont_params_for_task(task):
    """Return list of (name, raw_lo, raw_hi) for continuous dims."""
    lo, hi = task.hidden_size_range
    l1_hi = getattr(task, "l1_range_hi", 1e-2)
    l2_hi = getattr(task, "l2_range_hi", 1e-2)
    return [
        ("hidden_size",   lo,   hi),
        ("learning_rate", 1e-5, 1e-1),
        ("l1_reg",        1e-6, l1_hi),
        ("l2_reg",        1e-6, l2_hi),
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
    # BUG: integer encoding treats all categoricals as ordered. This is wrong for
    # unordered categoricals (e.g. activation function). Should use one-hot encoding
    # for unordered dims. If this branch is ever revived, fix encoding before use.
    return [float(choices.index(config[name])) for name, choices in cat_params]


def build_XY_rf(observations, cont_params, cat_params):
    rows, ys = [], []
    for o in observations:
        row = _cont_to_unit(o["config"], cont_params) + _cat_to_indices(o["config"], cat_params)
        rows.append(row)
        ys.append(o["mean_metric"])
    return np.array(rows, dtype=float), np.array(ys, dtype=float)


# ---------------------------------------------------------------------------
# Observation splitting
# ---------------------------------------------------------------------------

def get_primary_observations(observations):
    return [o for o in observations if not o.get("is_repeat", False)]


def get_repeat_pairs(observations):
    """Return list of (config, y_primary, y_repeat) for completed pairs."""
    pairs = []
    for obs in observations:
        if obs.get("is_repeat", False):
            ref_idx = obs.get("repeat_of")
            if ref_idx is not None and ref_idx < len(observations):
                primary = observations[ref_idx]
                pairs.append((primary["config"], primary["mean_metric"], obs["mean_metric"]))
    return pairs


# ---------------------------------------------------------------------------
# RF fitting
# ---------------------------------------------------------------------------

def fit_rf(X, Y, n_estimators=100):
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=2,  # prevents zero-variance single-sample leaves
        random_state=0,
    )
    rf.fit(X, Y.ravel())
    return rf


def get_tree_predictions(rf, X_query):
    """Returns shape (n_trees, n_query)."""
    return np.array([t.predict(X_query) for t in rf.estimators_])


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------

def _noise_features_single(config, cont_params, cat_params):
    """[1, <cont noise features>, <cat noise features>?] for one config."""
    unit_vals  = _cont_to_unit(config, cont_params)
    cont_names = [name for name, _, _ in cont_params]
    feats = [1.0] + [unit_vals[cont_names.index(n)] for n in NOISE_CONT_FEATURES]
    for name, choices in cat_params:
        if name in NOISE_CAT_FEATURES:
            lo, hi = math.log(min(choices)), math.log(max(choices))
            feats.append((math.log(config[name]) - lo) / (hi - lo) if hi > lo else 0.0)
    return np.array(feats, dtype=float)


def _noise_features_batch(unit_cont, combo, cont_params, cat_params):
    """Shape (n_cand, n_feats) for vectorised noise prediction."""
    n          = len(unit_cont)
    cont_names = [name for name, _, _ in cont_params]
    cols = [np.ones(n)] + [unit_cont[:, cont_names.index(n)] for n in NOISE_CONT_FEATURES]
    for name, choices in cat_params:
        if name in NOISE_CAT_FEATURES:
            lo, hi = math.log(min(choices)), math.log(max(choices))
            u_batch = (math.log(combo[name]) - lo) / (hi - lo) if hi > lo else 0.0
            cols.append(np.full(n, u_batch))
    return np.column_stack(cols)


def fit_noise_model(repeat_pairs, cont_params, cat_params):
    """OLS log-linear noise model on repeat pairs.

    Returns coefficient array or None if fewer than MIN_NOISE_PAIRS pairs.
    """
    if len(repeat_pairs) < MIN_NOISE_PAIRS:
        return None
    A, b = [], []
    for config, y1, y2 in repeat_pairs:
        noise = max(1e-10, (y1 - y2) ** 2 / 2)
        A.append(_noise_features_single(config, cont_params, cat_params))
        b.append(math.log(noise))
    coeffs, _, _, _ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    return coeffs


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------

def suggest_continuous_for_combo_rf(rf, noise_coeffs, combo, cont_params,
                                    cat_params, beta, n_candidates=N_RF_CANDIDATES, seed=0):
    n_cont = len(cont_params)

    engine = SobolEngine(dimension=n_cont, scramble=True, seed=seed)
    unit_cont = engine.draw(n_candidates).numpy()  # (n_cand, n_cont)

    # BUG: same integer encoding issue as _cat_to_indices — wrong for unordered categoricals.
    cat_indices = np.array([float(choices.index(combo[name])) for name, choices in cat_params])
    cat_part = np.empty((n_candidates, len(cat_indices)))
    cat_part[:] = cat_indices  # broadcast single row to all candidates
    X_cand = np.hstack([unit_cont, cat_part])

    tree_preds = get_tree_predictions(rf, X_cand)  # (n_trees, n_cand)
    mean      = tree_preds.mean(axis=0)
    var_total = tree_preds.var(axis=0, ddof=1)

    if noise_coeffs is not None:
        noise_feats  = _noise_features_batch(unit_cont, combo, cont_params, cat_params)
        var_aleatoric = np.exp(noise_feats @ noise_coeffs)
        var_epistemic = np.maximum(0.0, var_total - var_aleatoric)
    else:
        var_epistemic = var_total

    ucb      = mean + beta * np.sqrt(var_epistemic)
    best_idx = int(np.argmax(ucb))
    return _unit_to_cont(unit_cont[best_idx], cont_params)


# ---------------------------------------------------------------------------
# Sobol fallback
# ---------------------------------------------------------------------------

def sobol_continuous(seed, n_cont):
    engine = SobolEngine(dimension=n_cont, scramble=True, seed=seed)
    return engine.draw(1).double().squeeze(0)


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


def next_combo(run_counts, all_combos, rng=None):
    min_count = min(run_counts)
    tied = [i for i, c in enumerate(run_counts) if c == min_count]
    if rng is None:
        rng = np.random.default_rng()
    idx = int(rng.choice(tied))
    return all_combos[idx], idx


# ---------------------------------------------------------------------------
# Top-level: suggest next full config
# ---------------------------------------------------------------------------

def suggest_next(observations, task, beta=8.0):
    """
    Select least-visited categorical combo (by primary observations), then:
      - Sobol sample for continuous dims if total obs < N_SOBOL
      - Otherwise: RF surrogate with epistemic UCB acquisition
    Returns (config dict, combo_idx, mode_str).
    """
    cont_params  = _cont_params_for_task(task)
    cat_params   = cat_params_for_task(task)
    all_combos   = _all_combos_for_task(task)

    primary_obs = get_primary_observations(observations)
    n_primary   = len(primary_obs)

    run_counts       = build_run_counts(primary_obs, all_combos, cat_params)
    rng              = np.random.default_rng(n_primary)
    combo, combo_idx = next_combo(run_counts, all_combos, rng)

    if len(observations) < N_SOBOL:
        u    = sobol_continuous(seed=n_primary, n_cont=len(cont_params))
        cont = _unit_to_cont(u, cont_params)
        mode = "sobol"
    else:
        repeat_pairs  = get_repeat_pairs(observations)
        noise_coeffs  = fit_noise_model(repeat_pairs, cont_params, cat_params)

        X, Y = build_XY_rf(observations, cont_params, cat_params)
        rf   = fit_rf(X, Y)

        cont = suggest_continuous_for_combo_rf(
            rf, noise_coeffs, combo, cont_params, cat_params, beta,
            seed=len(observations),
        )
        mode = "rf"

    config = {**combo, **cont}
    if "depth"        in config: config["depth"]        = int(config["depth"])
    if "batch_size"   in config: config["batch_size"]   = int(config["batch_size"])
    if "n_rnn_layers" in config: config["n_rnn_layers"] = int(config["n_rnn_layers"])
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
