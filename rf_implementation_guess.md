# RF Implementation — What I Think It Does

The top-level loop in run_bo.py alternates between primary and repeat observations
in a P P R P P R pattern. When a repeat is due, the most recent primary config is
re-trained and stored with an is_repeat flag and a pointer back to its primary.
When a new suggestion is needed, suggest_next is called with the full observation
list. It separates primary observations from repeats, uses the primary count to
seed the Sobol sequence and for round-robin tie-breaking, and uses total
observation count to decide whether we are still in the Sobol phase or the RF
phase.

The RF is trained on all observations — primary and repeats — encoded as
unit-normalised log-scale continuous values concatenated with raw categorical
indices. The round-robin selects the categorical combo with the fewest primary
observations, breaking ties randomly. In the Sobol phase the continuous dims are
filled from a quasi-random Sobol sequence seeded by the primary count, giving
space-filling coverage. In the RF phase, 50,000 Sobol candidates are drawn for the
continuous dims, the selected categorical combo is appended as fixed columns, and
all 50,000 feature vectors are passed through every tree individually to get a
matrix of predictions of shape (n_trees, n_candidates). The mean and
between-tree variance are computed across trees for each candidate.

The noise model is fit on repeat pairs only. For each pair the aleatoric noise
estimate is (y1-y2)^2/2, and a log-linear OLS is fit regressing log(noise) on
unit-normalised learning rate, hidden size, and log-normalised batch size. This
gives a predicted aleatoric variance for any candidate config. Epistemic variance
is then total variance minus predicted aleatoric, clipped at zero. UCB is mean
plus beta times the square root of epistemic variance, and the candidate with the
highest UCB is returned and decoded back to a config dict.

The suggested continuous values are decoded from unit space back to raw scale via
log-inverse, with hidden_size rounded to the nearest integer. The returned config
merges the decoded continuous values with the categorical combo chosen by
round-robin, with integer casting applied to depth, batch_size, and n_rnn_layers.
