# Random Forest Surrogate Design Notes

## Why random forest over GP

The GP approach has a fundamental issue for our use case: the UCB acquisition
function uses posterior standard deviation as an exploration bonus. In regions with
inherently high outcome variance (high lr, small batch, small hidden size), the
posterior std stays elevated even after many observations because the variance is
irreducible. This permanently attracts exploration to noisy regions, independent of
how much we actually know about them.

Random forests handle this more naturally: tree disagreement (total variance) can be
decomposed into epistemic and aleatoric components using repeat observations, giving
a cleaner exploration signal.

Additional advantages of forests:
- Categorical inputs handled natively via tree splits — no kernel, no encoding assumptions
- Non-smooth landscapes handled without functional form assumptions
- Scales well with N (no O(N³) matrix inversion)
- Acquisition optimisation via grid search over random candidates — no gradient needed

## Design

### Encoding
Raw values passed directly to the forest:
- Continuous dims (hidden_size, lr, l1, l2): raw log-scale values
- Categorical dims: raw values (trees split on thresholds naturally; no one-hot needed)
- batch_size: raw integer (1, 8, 64) — the tree will find the right splits

### Repeat observations
Every other iteration trains the same config twice (both in Sobol phase and forest
phase). The repeat gives a direct local noise estimate:

    σ²_aleatoric(x) ≈ (y1 - y2)² / 2

These repeat pairs are spread across wherever we sample, giving an approximately
unbiased noise estimate proportional to our sampling distribution.

### Noise model
A log-linear OLS regression is fit on the repeat pairs:

    log σ²(x) = a0 + a1·log(lr) + a2·log(batch_size) + a3·log(hidden_size)

This generalises the local noise estimates to any config. Fit after each new repeat
pair is collected.

### Epistemic uncertainty
For a candidate config x, evaluate all trees individually to get predictions
y1_tree, ..., yn_tree. Then:

    σ²_total(x)      = variance across tree predictions
    σ²_aleatoric(x)  = noise model prediction at x
    σ²_epistemic(x)  = max(0, σ²_total(x) - σ²_aleatoric(x))

### Acquisition function
UCB over epistemic uncertainty only:

    UCB(x) = forest_mean(x) + β × sqrt(σ²_epistemic(x))

Optimised by evaluating on a large grid of random candidates (~10,000) and picking
the best, within the round-robin selected categorical combo.

## Known issue: self-reinforcing bias

The noise model informs the acquisition function, which determines where new samples
(and therefore repeats) are placed. If the noise model overestimates aleatoric noise
in some region early on:
1. Acquisition deprioritises that region
2. Fewer samples → fewer repeats → noise estimate stays poorly calibrated there
3. Region remains deprioritised indefinitely

This is partially mitigated by:
- The Sobol phase providing an unbiased initialisation (uniform sampling with repeats)
- The round-robin forcing categorical coverage regardless of acquisition values
- The continuous space within each combo may not be noisy enough for this to matter
  much in practice

A theoretically clean solution would involve occasional forced uniform exploration
(e.g. epsilon-greedy acquisition), but this is not currently planned. The Sobol
phase with repeats should provide enough calibration for the forest phase to start
well, and degradation of calibration during the forest phase is an open empirical
question.

## Open questions
- How quickly does the noise model calibration degrade during the forest phase?
- Is β=8.0 still appropriate, or does the cleaner epistemic uncertainty estimate
  allow a lower β?
- How many trees are needed for stable variance estimates? (100 is a standard default)
