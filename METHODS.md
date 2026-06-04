# Hyperparameters and Geometry: Project Overview and Methods

**Authors:** Thomas Colin  
**Collaborators:** Kirsten, Clay  
**Status:** Active development — RL and RNN test runs completed/in progress

---

## 1. Project Overview

This project is a follow-up to Devolder, Colin & Holroyd (submitted 2026), which ran a fully crossed grid search over 576 hyperparameter conditions on single-hidden-layer MLPs trained on a dual-task MNIST variant, with multiple networks per condition. That study examined the relationship between hyperparameters and representational geometry (measured via RDMs), and found complex, entangled relationships that could not be fully disentangled — motivating a more systematic approach.

The present project extends this in three directions:

1. **Depth.** Networks now have 1–2 hidden layers (with width halving across layers).
2. **Scale.** Rather than a fixed grid, we use Bayesian optimisation (BO) to search continuous hyperparameter dimensions (learning rate, regularisation, hidden size) more efficiently, while maintaining stratified coverage of categorical dimensions.
3. **Dynamics.** Activations are saved at logarithmically-spaced training checkpoints, enabling analysis of *how* representational geometry evolves during learning, not just its final state.

The analysis pipeline is: train many networks under varied hyperparameters → save hidden-layer activations on a fixed stimulus set at multiple training steps → compute RDMs → second-order RSA across tasks, checkpoints, layers, and hyperparameter configurations.

---

## 2. Task Suite

Nine tasks are implemented across three paradigms. All share the same BO infrastructure; tasks differ in their data, model type, and categorical hyperparameter space.

| Task | Key | Paradigm | Input | Output | Hidden range | Threshold | Stimuli (N) | Stimuli structure |
|---|---|---|---|---|---|---|---|---|
| MNIST dual-task | `mnist_dual` | Supervised MLP | 785-dim (784 pixels + task bit) | 1 (BCE) | 4–1024 | ≥ 0.90 val acc | 200 | 10 digits × 10 exemplars × 2 task bits |
| MNIST 10-way | `mnist_10way` | Supervised MLP | 784-dim | 10 (CE) | 4–1024 | ≥ 0.90 val acc | 100 | 10 digits × 10 exemplars |
| Fashion-MNIST 10-way | `fashion_10way` | Supervised MLP | 784-dim | 10 (CE) | 4–1024 | ≥ 0.85 val acc | 100 | 10 classes × 10 exemplars |
| Spirals (3-arm) | `spirals` | Supervised MLP | 2-dim | 3 (CE) | 16–256 | ≥ 0.85 val acc | 198 | 3 arms × 66 evenly spaced noiseless points |
| 8-bit Parity | `parity` | Supervised MLP | 8-dim | 1 (BCE) | 16–256 | ≥ 0.95 val acc | 118 | Up to 20 patterns per Hamming weight 0–8 |
| MNIST row-by-row | `mnist_rnn` | RNN | 28-dim per step, 28 steps | 10 (CE) | 16–256 | ≥ 0.90 val acc | 100 | 10 digits × 10 exemplars (as sequences) |
| Adding problem | `adding` | RNN | 2-dim per step, 50 steps | 1 (MSE) | 16–256 | MSE < 0.02 | 100 | 100 fixed sequences (seed 200) |
| CartPole-v1 | `cartpole` | RL (Q-learning) | 4-dim state | 2 Q-values | 16–256 | ≥ 195 mean return | 196 | 14×14 grid over (pole angle, pole angular velocity) |
| FourRooms | `fourrooms` | RL (Q-learning) | 61-dim RBF | 4 Q-values | 16–256 | ≥ 0.80 mean return | 61 | All non-wall cells, RBF-encoded |

### Notes on specific tasks

**MNIST dual-task.** The core task from Colin et al. (2025). A single image is presented with a 1-bit task indicator appended to the 784-pixel input (785 total). Task bit 0 = even/odd; task bit 1 = digit < 5. The output is a single logit trained with BCEWithLogitsLoss. The RDM stimulus set samples 10 exemplars per digit from the held-out test set; each exemplar appears twice, once with each task bit, yielding 200 stimuli.

**Spirals.** Three-arm Archimedean spiral in 2D (1000 training points per arm, 200 val per arm). Noise is additive Gaussian with σ = 0.1 × radius. RDM stimuli are 66 noiseless, evenly-spaced points per arm — no randomness involved in the stimulus set.

**Parity.** All 256 possible 8-bit patterns are used for both training and validation (the function is deterministic, so the task is memorisation). RDM stimuli are stratified: up to 20 patterns per Hamming weight (number of 1-bits), giving 118 stimuli total. Note that train and val sets are identical; val accuracy therefore tracks train accuracy throughout.

**Adding problem.** Each sequence consists of T=50 steps; each step is a (value, flag) pair where value ∈ [0,1] and exactly 2 flags are 1. The target is the sum of the two flagged values. Success threshold is MSE < 0.02 (a network that always predicts the mean of ~1.0 achieves MSE ≈ 0.17, so this is a meaningful threshold).

**CartPole.** Online Q-learning via Gymnasium's CartPole-v1. The RDM stimulus set is a 14×14 grid over pole angle × pole angular velocity with cart position and velocity fixed at 0.

**FourRooms.** Custom gridworld implementation (no Gymnasium dependency). An 11×11 grid with four interconnected rooms; goal is a fixed cell at (9, 9). State is encoded as a 61-dimensional RBF feature vector (one Gaussian per free cell, σ = 1.5). Reward: +1 on goal, −0.01 per step. The RDM stimulus set is every free cell, RBF-encoded (61 stimuli); metadata stores (row, col) for each.

---

## 3. Hyperparameters

Hyperparameters are divided into **categorical** (discrete, fully enumerable) and **continuous** (real-valued, BO-optimised on a log scale). The categorical space differs slightly by paradigm.

### 3.1 Categorical hyperparameters

`hidden_size` and `batch_size` are treated as **continuous** by the BO (see Section 3.2),
so they do not appear in the categorical space.

#### Supervised MLP tasks

| Parameter | Values | Notes |
|---|---|---|
| `depth` | 1, 2 | Number of hidden layers |
| `activation` | sigmoid, tanh, relu | Applied to all hidden layers |
| `optimizer` | sgd, adam | See Section 5.2 for details |
| `init_scale` | 0.1, 1.0 | Multiplier applied after standard init; see Section 5.3 |

This gives **2 × 3 × 2 × 2 = 24 categorical combinations**.

#### RL tasks

Same categorical parameters as supervised MLP except `init_scale` ∈ {0.1, 1.0}
and no `batch_size` (online Q-learning). Discount factor `gamma` is fixed at 0.99.
→ **2 × 3 × 2 × 2 = 24 categorical combinations**.

#### RNN tasks

| Parameter | Values |
|---|---|
| `cell_type` | rnn (Elman), gru |
| `n_rnn_layers` | 1, 2 |
| `optimizer` | sgd, adam |
| `init_scale` | 0.1, 1.0 |

This gives **2 × 2 × 2 × 2 = 16 categorical combinations** for RNN tasks.

### 3.2 Continuous hyperparameters

All continuous hyperparameters are optimised on a log scale, encoded as unit-normalised
log values in [0, 1] internally. `hidden_size` and `batch_size` are continuous in the
BO but rounded to integers when training a network; the pre-rounding values are stored
in `bo_state.json` as `cont_unit_vals`.

| Parameter | Range | Paradigms | Notes |
|---|---|---|---|
| `hidden_size` | [16, 256] | all | Rounded to nearest integer |
| `batch_size` | [1, 64] | supervised, RNN | Rounded to nearest integer |
| `learning_rate` | [1×10⁻⁵, 1×10⁻¹] | all | Passed directly to optimizer |
| `l1_reg` | [1×10⁻⁶, 1×10⁻¹] | all | Explicit L1 penalty on weight matrices only |
| `l2_reg` | [1×10⁻⁶, 1×10⁻²] | all | Passed as `weight_decay` to optimizer |

RL tasks have no `batch_size` continuous dim (online Q-learning, batch size = 1 implicitly).

---

## 4. Bayesian Optimisation

### 4.1 Overview

The acquisition function is **UCB-over-N_eff**:

```
A(x) = [μ(x) + sqrt(β)·σ(x)] / (1 + N_eff(x))
```

where μ(x) and σ(x) are the GP posterior mean and std of normalised accuracy, and
N_eff(x) is an effective local sample count that saturates regions already well-explored.
This naturally balances exploitation (GP UCB) with coverage (N_eff denominator) without
requiring explicit round-robin or culling in the GP phase.

### 4.2 Phases

**Sobol phase** (first N_SOBOL = 200 primary observations): quasi-random Sobol sequence for
continuous dims, with round-robin over categorical combos to ensure all are visited
at least once before the GP takes over.

**GP phase** (N_SOBOL onwards): fit a GP, optimise A(x) jointly over all categorical
combos and continuous dims, select the global argmax.

### 4.3 Gaussian Process

**Model:** `MixedSingleTaskGP` (BoTorch) — one GP over all observations jointly,
not one per categorical combo. Cross-combo information sharing is handled by the
mixed kernel.

**Input encoding:**
- Continuous dims (including hidden_size, batch_size): log-transformed, normalised to [0,1]
- Ordinal categoricals (depth, n_rnn_layers): mapped to [0,1] via log-scale rank
- Unordered categoricals (activation, optimizer, init_scale, etc.): integer indices

**Target:** normalised metric `y = (raw - chance_perf) / (max_metric - chance_perf)`, clamped to [0,1]. `chance_perf` and `max_metric` are task-specific attributes.

**MLL fitting:** `ExactMarginalLogLikelihood` via `fit_gpytorch_mll` (L-BFGS-B).

### 4.4 N_eff

```
N_eff(x) = Σ_i  exp(-d²(x, x_i) / 2h²)
```

summed over **all observations** (primaries and repeats). Distance is Euclidean
in unit space: squared distance for continuous and ordinal dims, binary (0/1) for
unordered categoricals. Because the binary categorical penalty is 1 and h is small,
exp(−1/2h²) ≈ 0, so observations in different categorical combos contribute
negligibly — each combo's continuous subspace is effectively saturated independently.

**h is paradigm-specific**, chosen so that the 90th percentile of Sobol-equivalent
N_eff reaches 0.5 at 1000 total observations:

| Paradigm | h |
|----------|-------|
| RL | 0.116 |
| Supervised | 0.160 |
| RNN | 0.148 |

See `output/h_selection.md` for full derivation.

### 4.5 Acquisition optimisation

`optimize_acqf_mixed` (BoTorch) with all categorical combos in `fixed_features_list`.
Continuous dims are optimised jointly via gradient ascent with categoricals fixed
per combo. Parameters: 3 restarts, 32 raw samples, 20 L-BFGS-B iterations.

### 4.6 Scoring

`performance` is the raw validation metric (no penalty or thresholding). The GP
sees the full gradient from chance-level up through successful networks.
`success_threshold` is used only for console reporting.

### 4.7 State persistence

After every iteration, the full observation history is written to `bo_state.json`
and uploaded to S3 (cloud runs). Runs are resumable: on restart, the script loads
existing observations and continues from where it left off.

### 4.8 Repeat infrastructure

Every 4th primary observation triggers a noise-estimation repeat of a previous config
(P P P P R pattern, ~20% repeats). Repeats are included in N_eff (they represent real
observations of a region) and in the GP fit, but are not selected by the acquisition
function — they exist solely to estimate intra-config variance.

---

## 5. Model Architectures

### 5.1 MLP (supervised and RL tasks)

A fully connected feedforward network with the following width schedule:

```
input → H → H//2 → H//4 → ... → output
```

where H is `hidden_size` and each successive layer halves the width. The number of hidden layers equals `depth` (1 or 2). No dropout.

**Special case:** If `hidden_size < 8`, `depth` is capped at 2, because `H // 4` would be less than 2 units (degenerate). The effective depth after this cap is stored as `effective_depth` in the saved metadata.

Activation functions are applied to all hidden layers; no activation on the output layer.

### 5.2 RNN

A stacked Elman RNN or GRU with:
- `n_rnn_layers` stacked recurrent layers (1 or 2)
- `hidden_size` units in each layer
- A linear readout head from the final hidden state at the last time step

For Elman RNN, the PyTorch default nonlinearity (`tanh`) is used. LSTM is not included (removed for architectural simplicity and to keep the categorical space balanced).

---

## 6. Training

### 6.1 Supervised training loop

Each network is trained for up to `max_epochs` epochs. All supervised and RNN tasks use the global default of **100 epochs** — no per-task overrides.

**Batch loading:** standard PyTorch DataLoader with shuffling each epoch. Validation is done with batch size 512, no shuffling.

**Loss functions:**

| Paradigm | Loss |
|---|---|
| Binary classification (mnist_dual, parity) | BCEWithLogitsLoss |
| Multi-class classification (mnist_10way, fashion_10way, spirals) | CrossEntropyLoss |
| Regression (adding) | MSELoss |

**Regularisation:**

- **L2:** passed as `weight_decay` to the optimizer. Applied by the optimizer to all parameters.
- **L1:** added explicitly to the loss at each step. Applied only to weight matrices (`param.ndim > 1`), not biases.

Total loss per step: `criterion(logits, targets) + l1_coef × Σ |W_ij|`

### 6.2 RL training loop

Online Q-learning (no replay buffer — the non-iid nature of the training signal is
intentional). Each network trains for up to **100,000 environment steps**. Training
stops early if the success threshold is reached.

**Epsilon-greedy exploration:** ε decays linearly from 0.5 to 0 over 100,000 steps.
Policy is evaluated every 5,000 steps (mean return over 20 episodes).

**Metric stored:** mean episode return at the best evaluation point.
For the GP, returns are normalised: `y = (raw - chance_perf) / (max_metric - chance_perf)`.

**Adding task metric:** `train_rnn` returns *negative* MSE (so higher = better, consistent
with all other tasks). `chance_perf = −0.1667` (negated MSE of a naive predictor always
outputting the mean), `max_metric = 0.0` (perfect predictor). `success_threshold = −0.02`
corresponds to MSE < 0.02.

### 6.3 Optimizers

| Optimizer | Parameters |
|---|---|
| SGD | `lr=learning_rate`, `momentum=0.9`, `weight_decay=l2_reg` |
| Adam | `lr=learning_rate`, `weight_decay=l2_reg` (betas at PyTorch defaults: 0.9, 0.999) |

### 6.4 Weight initialisation

All linear layers (including the RNN readout head) are initialised with:
- **ReLU networks:** Kaiming normal (`fan_in` mode)
- **Sigmoid / tanh / RNN networks:** Xavier normal

After standard initialisation, all weights are **scaled by `init_scale`** (multiplicative).
Biases are always initialised to zero.

`init_scale` ∈ {0.1, 1.0} across all tasks. `init_scale = 0.1` produces near-zero
initial weights; `init_scale = 1.0` uses the standard initialisation directly.

### 6.5 Early stopping

- **Minimum epochs:** 15. Early stopping is not considered before this.
- **Patience:** 10 epochs without improvement in validation *loss* (threshold 1×10⁻⁴). Note: early stopping watches `val_loss`, not `val_acc`.
- **Best model:** tracked separately by `val_acc` (or the task's `metric_name`). The best checkpoint is saved to `model_best.pt`.

### 6.6 Activation checkpoints

Hidden-layer activations are saved on the fixed RDM stimulus set at **log₄-spaced training steps**: steps 1, 4, 16, 64, 256, 1024, 4096, … up to and including the final training step. This gives approximately equal coverage per order of magnitude of training progress.

Additionally, activations are always saved at the **final step** (current weights at end of training) and by **reloading `model_best.pt`** and saving those as `best.npz`.

**MLP:** post-activation outputs of each hidden layer are saved as `layer_0`, `layer_1`, ... Each array has shape `(N_stimuli, hidden_size_of_that_layer)`.

**RNN:** hidden states at a task-specific subset of time steps (to limit storage). Arrays are keyed `t_0`, `t_5`, etc. For MNIST-RNN: steps [0, 5, 11, 17, 22, 27]. For Adding: steps [0, 4, 9, 19, 34, 49].

---

## 7. Output Files

For each trained network, the following files are written under `output/experiments/<task>/run_NNNN_rR/`:

| File | Contents |
|---|---|
| `metadata.json` | Task name, full config (including `effective_depth`), best epoch/step, best metric, final epoch/step, final metric |
| `history.json` | Per-epoch: epoch number, global step, train loss, val loss, val acc |
| `model_best.pt` | PyTorch state dict at the epoch of peak val acc |
| `step_XXXXXXX.npz` | Activations on RDM stimuli at global step XXXXXXX (one file per checkpoint) |
| `best.npz` | Activations from `model_best.pt` weights |
| `final.npz` | Activations from end-of-training weights |

At the task level, `output/experiments/<task>/bo_state.json` stores the full observation history:

```json
[
  {
    "iteration": 0,
    "config": { ... },
    "cont_unit_vals": [0.42, 0.71, ...],
    "val_accs": [0.923],
    "performance": 0.923,
    "is_repeat": false,
    "repeat_of": null
  },
  ...
]
```

`cont_unit_vals` stores the raw pre-rounding unit values for continuous dims (used by
the GP to see the actual explored location, not the snapped integer). `performance` is
the raw validation metric passed to the GP. `val_accs` are per-repetition values.

---

## 8. Reproducibility Notes

- **RDM stimuli** are generated from fixed seeds (`seed=42` throughout, except the Adding task which uses `seed=200` for stimuli to decouple from the training data seed). Stimuli are identical across all runs for a given task.
- **Sobol initialisation** uses `seed = len(observations)` at the time of the call. Given a fixed run order and no interruptions, this is fully deterministic. After interruption and resume, the seed correctly reflects completed observations, preserving the sequence.
- **Training data** is generated/loaded with `seed=42` for train splits and `seed=43` for val splits (where applicable). MNIST and Fashion-MNIST are downloaded from standard sources; the train/val split uses `sklearn.model_selection.train_test_split` with `random_state=seed` and stratification by label.
- **No global random seed is set** during training. Results across runs of the same config will vary (this is intentional — the BO runs 2 repetitions per config by default to separate stochastic from hyperparameter-driven variance).
- `effective_depth` (the actual number of hidden layers used, after the small-network cap) is recorded in `metadata.json` alongside the requested `depth`.
