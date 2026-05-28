# Bayesian Optimisation Design

## Research goal

The objective is not to find the best network. It is to produce many interesting,
good-enough networks across the hyperparameter space for downstream representational
geometry analysis. The BO must therefore balance accuracy (networks that actually
learn) with broad coverage of the hyperparameter space.

---

## Alternatives considered

### Option A — Two GPs: one for accuracy, one for RDM sensitivity C(x)

Acquisition:
```
A(x) = [μ_acc(x) + sqrt(β)·σ_acc(x)] × C_GP(x) / (1 + N_eff(x))
```

The second GP is fit on observed C(x) values (local RDM dissimilarity in the
hyperparameter neighbourhood of each past network) and predicts C in unexplored
regions.

**Advantages:** principled estimate of RDM sensitivity even before sampling a region;
the GP's uncertainty over C drives exploration toward potentially interesting areas.

**Disadvantages:** two GPs to fit, tune, and maintain; C(x) requires storing and
loading RDMs for all past networks at each step. The empirical C estimates used as
GP targets improve as more networks are added (denser neighbourhoods give better
estimates), so refitting each iteration is necessary — but this is no different from
any Bayesian updating and not a fundamental problem.

---

### Option B — Single GP over the composite interestingness score I(x)

```
I(x) = accuracy_over_chance(x) × C(x) / (1 + N_eff(x))
```

Recompute I(x_i) for all past observations at each iteration (reflecting the
current dataset), then refit a single GP on (X, I(X)).

**Advantages:** one model; automatically learns correlations between hyperparameters
and "interestingness."

**Disadvantages:** the GP conflates accuracy uncertainty with C uncertainty — it
cannot disentangle why a region had high I (good accuracy? high RDM sensitivity?).
This makes the model harder to interpret and potentially miscalibrated. I(x) is
a composite of multiple terms whose relative contributions change as the dataset
grows, requiring a full refit each iteration.

---

### Option C — Single GP for accuracy, N_eff computed directly  ✓ chosen

```
A(x) = [μ(x) + sqrt(β)·σ(x)] / (1 + N_eff(x))
```

The GP models normalised accuracy only — a fixed, stationary function. C(x) is
dropped (it would be good to have, but not essential to the core goal). N_eff(x)
is computed directly from existing observations at acquisition time; no model needed.

**Advantages:** GP has a clean, stable target; no full refit cost from non-stationarity;
N_eff is exact (not estimated); mechanistically transparent.

**Disadvantages:** does not explicitly reward RDM-sensitive regions; C(x) = 0 for
unexplored regions, but the GP's σ term already drives exploration there.

**Why C(x) was dropped:** The primary goal is coverage of the learnable hyperparameter
space with good-enough networks. Actively seeking RDM-sensitive regions is desirable
but secondary. Option A can be revisited if downstream geometry analysis reveals that
coverage is not informative enough.

---

## Chosen approach: full specification

### Accuracy normalisation

Each task defines a `chance_accuracy`. The GP target is:

```
y = (raw_accuracy - chance_accuracy) / (1 - chance_accuracy)
```

Clamped to [0, 1]. This ensures consistent GP behaviour across tasks with different
output sizes (binary at 50% vs 10-way at 10%).

### Acquisition function

```
A(x) = [μ(x) + sqrt(β)·σ(x)] / (1 + N_eff(x))
```

- `μ(x)`, `σ(x)`: posterior mean and std of the normalised-accuracy GP at x.
- `β`: exploration weight. Start at 4.0 (to be tuned after initial runs).
- `N_eff(x)`: effective local sample count (see below).

When N_eff(x) ≈ 0 (unexplored region), A(x) = raw UCB. As a region saturates,
N_eff grows and A(x) shrinks, pushing the search to less-visited areas.

### N_eff

```
N_eff(x) = Σ_i  exp(-d²(x, x_i) / 2h²)
```

Parameters: **h = 0.2**, **λ = 0.1** (see rationale below).

#### Distance metric

```
d²(x, x_i) = d²_cont + d²_ord + λ · d²_unord
```

**Continuous dims** (learning_rate, l1_reg, l2_reg) — log-normalised to [0, 1]:
```
d²_cont = Σ_j  (ũ_j(x) - ũ_j(x_i))²
```

**Ordinal categoricals** (hidden_size, batch_size, depth, n_rnn_layers) — treated
as pseudo-continuous using their log-normalised ordinal index in [0, 1]:
```
d²_ord = Σ_k  (ô_k(x) - ô_k(x_i))²
```
where ô_k = (rank - 0) / (n_choices - 1), computed on log-scale of the underlying
values (e.g. hidden_size ∈ {16, 256} → ô ∈ {0, 1}).

**Unordered categoricals** (activation, optimizer, init_scale) — binary mismatch:
```
d²_unord = Σ_k  1[c_k(x) ≠ c_k(x_i)]
```

#### Parameter rationale

**h = 0.2**: with a budget of ~1000 networks per task and 144 combos (~7 per combo
on average), the expected N_eff ≥ 1 threshold is reached after ~8 networks per
combo (~1150 total). This means the denominator becomes meaningfully large only
when a combo has been genuinely saturated, not prematurely.

**λ = 0.1**: allows moderate cross-combo bleed. One unordered categorical mismatch
contributes exp(-0.1 / 0.08) ≈ 0.29 to K — meaningful but not dominant. Three
mismatches contribute exp(-0.3 / 0.08) ≈ 0.02 — negligible. This reflects that
sampling relu+lr=0.001 ten times is a reason to deprioritise sigmoid+lr=0.001
relative to sigmoid+lr=0.0001, even though the two activations are distinct
experiments. Both the GP (via its categorical kernel) and N_eff (via λ) convey
this information, but through complementary channels: the GP measures accuracy
uncertainty, N_eff measures resource expenditure.

### What replaces round-robin and culling

Round-robin and UCB-based combo culling are dropped. The N_eff denominator provides
natural coverage pressure: once any region (continuous or categorical) is saturated,
its acquisition value drops and the optimiser naturally moves elsewhere. The GP's σ
term handles exploration into truly unexplored regions (where N_eff ≈ 0).

The Sobol phase is retained for initial coverage. During the Sobol phase N_eff is
not used for selection (Sobol points are pre-determined); N_eff only enters in the
GP phase.

### Repeat infrastructure

The P P P P R repeat pattern (every 4th primary triggers a repeat) is retained
unchanged. Repeats are not counted in N_eff (only primary observations are used).

---

## Implementation plan

### 1. Add `chance_accuracy` to each task

In `tasks/base.py`, add:
```python
chance_accuracy: float   # e.g. 0.5 for binary, 0.1 for 10-way
```

For classification tasks, chance_accuracy = 1 / output_size. For regression/RL
tasks (adding, cartpole, fourrooms), define an appropriate baseline (e.g. 0.0).

### 2. Normalise accuracy in `src/bo.py`

In `build_XY`, apply normalisation before building Y:
```python
def build_XY(observations, cont_params, cat_params, chance_accuracy=0.0):
    X = torch.stack([encode_config(o["config"], cont_params, cat_params)
                     for o in observations])
    raw = [o["mean_metric"] for o in observations]
    norm = [(r - chance_accuracy) / max(1e-6, 1 - chance_accuracy) for r in raw]
    Y = torch.tensor([[y] for y in norm], dtype=torch.double)
    return X, Y
```

Pass `task.chance_accuracy` through from `suggest_next`.

### 3. Add ordinal encoding helpers in `src/bo.py`

Add a mapping from categorical param name → whether it is ordinal:
```python
ORDINAL_PARAMS = {"hidden_size", "batch_size", "depth", "n_rnn_layers"}
```

Add `_ord_to_unit(value, choices)` that log-normalises the ordinal index:
```python
def _ord_to_unit(value, choices):
    idx = choices.index(value)
    if len(choices) == 1:
        return 0.0
    # log-scale: use log of the value if numeric, else just rank
    try:
        logs = [math.log(c) for c in choices]
        lo, hi = logs[0], logs[-1]
        return (math.log(value) - lo) / (hi - lo)
    except (TypeError, ValueError):
        return idx / (len(choices) - 1)
```

### 4. Add `compute_n_eff` in `src/bo.py`

```python
def compute_n_eff(x_query, observations, cont_params, cat_params,
                  h=0.2, lam=0.1):
    """
    x_query: dict (config) or unit-row tensor for a candidate point.
    Returns scalar N_eff.
    """
    if not observations:
        return 0.0
    total = 0.0
    for obs in observations:
        if obs.get("is_repeat"):
            continue
        d2 = 0.0
        xi = obs["config"]
        # continuous
        for name, raw_lo, raw_hi in cont_params:
            u  = _cont_to_unit_val(x_query[name], raw_lo, raw_hi)
            ui = _cont_to_unit_val(xi[name],      raw_lo, raw_hi)
            d2 += (u - ui) ** 2
        # categoricals
        for name, choices in cat_params:
            if name in ORDINAL_PARAMS:
                o  = _ord_to_unit(x_query[name], choices)
                oi = _ord_to_unit(xi[name],      choices)
                d2 += (o - oi) ** 2
            else:
                if x_query[name] != xi[name]:
                    d2 += lam
        total += math.exp(-d2 / (2 * h * h))
    return total
```

### 5. Custom acquisition function

Implement `UCBoverNeff` as a BoTorch `AnalyticAcquisitionFunction` (or a thin
wrapper that evaluates on a candidate set):

```python
class UCBoverNeff(AnalyticAcquisitionFunction):
    def __init__(self, model, beta, n_eff_values):
        # n_eff_values: precomputed tensor of N_eff for each candidate
        ...
    def forward(self, X):
        mean, sigma = self._mean_and_sigma(X)
        ucb = mean + self.beta.sqrt() * sigma
        return ucb / (1 + self.n_eff_values)
```

Since N_eff depends on the candidate x (not just on training data), it must be
evaluated jointly with the acquisition. The cleanest approach: enumerate a large
Sobol grid of candidates (e.g. 2048 per active combo), compute N_eff for each,
compute UCB for each, divide, and take the argmax. This avoids gradient-based
optimisation through N_eff (which would require autodiff through the distance loop).

### 6. Modify `suggest_next` in `src/bo.py`

Replace the current round-robin + culling logic with:

```
Sobol phase (n_primary < N_SOBOL):
    unchanged — quasi-random with round-robin for initial coverage

GP phase (n_primary >= N_SOBOL):
    1. Fit GP on normalised accuracy
    2. Generate Sobol candidate grid (2048 × n_combo candidates, one grid per combo)
    3. For each candidate: compute N_eff and UCB, compute A = UCB / (1 + N_eff)
    4. Select candidate with highest A
    5. Decode to config dict
```

### 7. Update `run_bo.py`

- Remove beta argument from round-robin / culling calls (no longer needed there)
- Pass `task.chance_accuracy` through to `suggest_next`
- Keep repeat logic unchanged
- Keep diagnostic prints but update to show N_eff statistics instead of culling counts

### 8. Clean up

- Remove `get_active_combos`, `_combo_ucb_max`, `build_run_counts`, `next_combo`
  from `src/bo.py` (no longer used in main path; keep if still used by culling
  test scripts)
- Remove `--beta` argument from `run_bo.py` or repurpose as the UCB beta

---

## Open questions

- **Beta value**: with N_eff providing coverage pressure, a fixed β = 4.0 may be
  appropriate (less need for a decay schedule). To be decided after initial runs.
- **Sobol phase round-robin**: keeping round-robin during Sobol ensures all combos
  visited at least once before GP takes over. This is still desirable for the same
  reason as before.
- **C(x) revisit**: if geometry analysis later shows that the sampling is not
  sufficiently concentrated in hyperparameter-sensitive regions, Option A (second GP
  for C) can be added on top of this architecture without changing the core GP or
  N_eff logic.
- **h and λ tuning**: h=0.2 and λ=0.1 are principled starting points but should
  be validated empirically on spirals before committing to full 9-task runs.
