# BO Design Notes: Sobol + GP for Hyperparameter Search

## The goal

We are not trying to find optimal hyperparameters. We want a diverse collection of
successful networks across the full categorical space (72 combos) so we can study
how hyperparameters relate to representational geometry. This is a fundamentally
different objective from standard BO.

## Current setup

- Round-robin over 72 categorical combos (batch_size × depth × activation ×
  optimizer × init_scale), with randomised tie-breaking to ensure diverse coverage
  from the first iteration
- Sobol sequence for continuous dims (hidden_size, lr, l1, l2) for the first
  N_SOBOL iterations
- MixedSingleTaskGP (BoTorch) over all observations jointly — one GP across all
  combos, not one per combo — for remaining iterations
- Acquisition: qUCB with β=8.0

## What the empirical results show (spirals, 200 iterations)

- Sobol phase (100 iterations): 3% success rate (3/100 above threshold)
- GP phase (100 iterations): 22% success rate (22/100 above threshold)
- The GP demonstrably improves on Sobol — this is real signal, not noise
- The GP has learned the lr-batch_size relationship correctly (lower lr suggested
  for smaller batches)
- High-lr failures are spread evenly across batch sizes, not concentrated at bs=1

## The stochasticity bias problem

qUCB acquires at mean + β × std. Regions with inherently high outcome variance
(e.g. high lr, small batch, small hidden size) maintain high posterior std even
after many observations, because the variance is irreducible. With high β, the
acquisition function is permanently attracted to these regions.

The standard fix is a noise-aware (homoscedastic) GP that estimates a global noise
parameter σ², separating observation noise from function uncertainty. With multiple
reps at the same config, σ² can be estimated directly: y1 - y2 = ε1 - ε2 ~
N(0, 2σ²), so (y1-y2)²/2 is an unbiased estimate of σ² that doesn't depend on f.

However, noise is heteroscedastic in our setting — variance depends on the config
(high lr + small batch is more variable than low lr + large batch). A single global
σ² partially corrects the bias but doesn't eliminate it, since some regions will
still have above-average noise and attract disproportionate exploration.

A fully heteroscedastic GP would model σ²(x) as a second latent function, but this
requires considerably more data and complexity than we currently have.

In practice, the empirical data from spirals does not show a strong stochasticity
bias — the GP's lr suggestions are lower for smaller batches (correct direction) and
success rates improved substantially over Sobol. The bias may be present but is
competing with and losing to the performance signal.

## The exploration-exploitation tension

- High β → more exploration → bias toward high-variance regions
- Low β → more exploitation → clustering around known good combos (relu+depth=2)
- Neither is right for our goal of diverse successful networks

The round-robin handles categorical diversity by construction. The GP's job is to
find good continuous params per combo. For combos with Sobol successes, it refines
well. For combos with no successes, it relies on cross-combo generalisation.

## Cross-combo generalisation

The GP fits one joint model over all combos. A success for relu+adam at lr=0.003,
H=256 informs the GP's beliefs about lr and H for sigmoid+sgd too, discounted by
how different those categorical values are in the kernel. This is the key advantage
of a joint GP over separate per-combo models.

The 3%→22% improvement from Sobol to GP phase is partly attributable to this
cross-combo transfer. The GP is not "blind" for combos with no successes — it has
sub-threshold accuracy values (e.g. 0.35, 0.42, 0.49) that still carry information
about which continuous params are better or worse within those combos.

## Open questions

- How much does the GP's cross-combo transfer degrade for combos that are genuinely
  different from the successful ones (e.g. sigmoid+sgd vs relu+adam)? This is
  empirically testable by checking whether GP suggestions for hard combos improve
  over iterations.
- Is β=8.0 well-calibrated? The scale depends on the posterior std, which we
  haven't inspected directly.
- Would 2 reps per config help enough to justify halving the number of unique configs
  explored? Probably not unless outcome variance turns out to be high — and the
  spirals data suggests variance is low enough that the GP is finding signal.

## Current conclusion

The setup is more reasonable than a naive analysis suggests. The main risks are:

1. Hard combos (sigmoid+sgd) may not get enough signal for the GP to refine, even
   with cross-combo transfer
2. H=256 hitting the ceiling repeatedly suggests the hidden size range should be
   widened for some tasks
3. The stochasticity bias is theoretically present but empirically modest for spirals

The most actionable improvement is ensuring the Sobol phase produces enough signal
across diverse combos before the GP takes over — either by increasing N_SOBOL or by
tightening the continuous bounds so the Sobol hits are more frequent.
