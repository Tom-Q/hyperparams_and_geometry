# Bayesian Optimisation Design

## Research goal

The objective is not to find the best network. It is to produce many successful
networks (above accuracy threshold) with broad coverage of the hyperparameter space,
for downstream representational geometry analysis. The BO must therefore balance
finding working configurations with coverage — not converge on a single optimum.

---

## Alternatives considered

### Option A — Two GPs: one for accuracy, one for RDM sensitivity C(x)

```
A(x) = [μ_acc(x) + sqrt(β)·σ_acc(x)] × C_GP(x) / (1 + N_eff(x))
```

**Advantages:** principled estimate of RDM sensitivity even before sampling a region.

**Disadvantages:** two GPs to fit and maintain; C(x) requires storing and loading
RDMs for all past networks at each step. C(x) estimates improve as more networks
are added, so refitting each iteration is necessary.

---

### Option B — Single GP over composite interestingness score I(x)

```
I(x) = accuracy_over_chance(x) × C(x) / (1 + N_eff(x))
```

**Advantages:** one model; learns correlations between hyperparameters and interestingness.

**Disadvantages:** conflates accuracy uncertainty with C uncertainty; I(x) is
non-stationary as the dataset grows, making calibration harder.

---

### Option C — Single GP for accuracy, N_eff computed directly  ✓ chosen

```
A(x) = [μ(x) + sqrt(β)·σ(x)] / (1 + N_eff(x))
```

The GP models normalised accuracy only. N_eff(x) is computed directly from
existing observations at acquisition time — no second model needed.

**Advantages:** GP has a clean, stable target; N_eff is exact; mechanistically
transparent.

**Disadvantages:** does not explicitly reward RDM-sensitive regions. This can be
revisited if downstream geometry analysis reveals that coverage alone is not
informative enough.

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
- `β = 4.0` (default).
- `N_eff(x)`: effective local sample count (see below).

When N_eff(x) ≈ 0 (unexplored region), A(x) = raw UCB. As a region saturates,
N_eff grows and A(x) shrinks, naturally pushing the search elsewhere.

### N_eff

```
N_eff(x) = Σ_i  exp(-d²(x, x_i) / 2h²)
```

Default **h = 0.15** for standard runs (higher h = broader saturation footprint).

#### Distance metric

Euclidean distance in unit space — no per-dimension normalisation:

```
d²(x, x_i) = d²_cont + d²_ord + d²_unord
```

**Continuous dims** (learning_rate, l1_reg, l2_reg, hidden_size, batch_size) —
log-normalised to [0, 1]:
```
d²_cont = Σ_j  (u_j(x) - u_j(x_i))²
```

**Ordinal categoricals** (depth, n_rnn_layers) — mapped to [0, 1] via log-scale
of the underlying values:
```
d²_ord = Σ_k  (ô_k(x) - ô_k(x_i))²
```

**Unordered categoricals** (activation, optimizer, init_scale, cell_type, gamma)
— binary mismatch, weight 1.0 per dimension:
```
d²_unord = Σ_k  1[c_k(x) ≠ c_k(x_i)]
```

Note: hidden_size and batch_size are treated as continuous (not categorical), so
they contribute to d²_cont. The raw pre-rounding unit values are stored in each
observation (`cont_unit_vals`) so the GP sees the actual explored location, not
the snapped integer value.

### What replaces round-robin

Round-robin is used only during the Sobol phase to ensure all categorical combos
are visited at least once before the GP takes over. In the GP phase, the acquisition
function provides natural coverage pressure: once a region saturates (N_eff grows),
A(x) drops and the optimiser moves elsewhere. No explicit culling or exclusion.

### Repeat infrastructure

Every 4th primary observation triggers a noise-estimation repeat of a previous
config (P P P P R pattern). Repeats are not counted in N_eff — only primary
observations contribute.

---

## Implementation notes

### Optimisation

`optimize_acqf_mixed` (BoTorch) is called with all categorical combos in
`fixed_features_list`, so continuous dims are optimised jointly across all combos
in one call. Parameters: 3 restarts, 32 raw samples, 20 L-BFGS-B iterations.

### cont_unit_vals

hidden_size and batch_size are continuous in the model but must be rounded to
integers when training a network. The raw optimiser output (pre-rounding) is stored
as `cont_unit_vals` in each observation. This prevents the GP from treating all
observations that round to the same integer as a single point.

### Sobol phase

N_SOBOL = 100. Sobol quasi-random sequences fill the continuous space; round-robin
ensures all categorical combos are visited before the GP phase starts.
