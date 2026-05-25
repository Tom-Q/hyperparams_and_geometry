# Hyperparameters and Geometry: Project Overview and Methods

**Authors:** Thomas Colin  
**Collaborators:** Kirsten, Clay  
**Status:** Active development — smoke testing BO pipeline

---

## 1. Project Overview

This project is a follow-up to Devolder, Colin & Holroyd (submitted 2026), which ran a fully crossed grid search over 576 hyperparameter conditions on single-hidden-layer MLPs trained on a dual-task MNIST variant, with multiple networks per condition. That study examined the relationship between hyperparameters and representational geometry (measured via RDMs), and found complex, entangled relationships that could not be fully disentangled — motivating a more systematic approach.

The present project extends this in three directions:

1. **Depth.** Networks now have 1–3 hidden layers (with width halving across layers).
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
| MNIST row-by-row | `mnist_rnn` | RNN | 28-dim per step, 28 steps | 10 (CE) | 16–512 | ≥ 0.90 val acc | 100 | 10 digits × 10 exemplars (as sequences) |
| Adding problem | `adding` | RNN | 2-dim per step, 50 steps | 1 (MSE) | 16–512 | MSE < 0.02 | 100 | 100 fixed sequences (seed 200) |
| CartPole-v1 | `cartpole` | RL (Q-learning) | 4-dim state | 2 Q-values | 4–256 | ≥ 195 mean return | 196 | 14×14 grid over (pole angle, pole angular velocity) |
| FourRooms | `fourrooms` | RL (Q-learning) | 61-dim RBF | 4 Q-values | 8–512 | ≥ 0.80 mean return | 61 | All non-wall cells, RBF-encoded |

### Notes on specific tasks

**MNIST dual-task.** The core task from Colin et al. (2025). A single image is presented with a 1-bit task indicator appended to the 784-pixel input (785 total). Task bit 0 = even/odd; task bit 1 = digit < 5. The output is a single logit trained with BCEWithLogitsLoss. The RDM stimulus set samples 10 exemplars per digit from the held-out test set; each exemplar appears twice, once with each task bit, yielding 200 stimuli.

**Spirals.** Three-arm Archimedean spiral in 2D (1000 training points per arm, 200 val per arm). Noise is additive Gaussian with σ = 0.1 × radius. RDM stimuli are 66 noiseless, evenly-spaced points per arm — no randomness involved in the stimulus set.

**Parity.** All 256 possible 8-bit patterns are used for both training and validation (the function is deterministic, so the task is memorisation). RDM stimuli are stratified: up to 20 patterns per Hamming weight (number of 1-bits), giving 118 stimuli total. L1 and L2 regularisation are capped at 0.01 for this task (vs. 0.1 globally) since heavy regularisation prevents the high-order weight interactions parity requires. Note that train and val sets are identical; val accuracy therefore tracks train accuracy throughout.

**Adding problem.** Each sequence consists of T=50 steps; each step is a (value, flag) pair where value ∈ [0,1] and exactly 2 flags are 1. The target is the sum of the two flagged values. Success threshold is MSE < 0.02 (a network that always predicts the mean of ~1.0 achieves MSE ≈ 0.17, so this is a meaningful threshold).

**CartPole.** Online Q-learning via Gymnasium's CartPole-v1. The RDM stimulus set is a 14×14 grid over pole angle × pole angular velocity with cart position and velocity fixed at 0.

**FourRooms.** Custom gridworld implementation (no Gymnasium dependency). An 11×11 grid with four interconnected rooms; goal is a fixed cell at (9, 9). State is encoded as a 61-dimensional RBF feature vector (one Gaussian per free cell, σ = 1.5). Reward: +1 on goal, −0.01 per step. The RDM stimulus set is every free cell, RBF-encoded (61 stimuli); metadata stores (row, col) for each.

---

## 3. Hyperparameters

Hyperparameters are divided into **categorical** (discrete, fully enumerable) and **continuous** (real-valued, BO-optimised on a log scale). The categorical space differs slightly by paradigm.

### 3.1 Categorical hyperparameters

#### Supervised MLP and RL tasks

| Parameter | Values | Notes |
|---|---|---|
| `batch_size` | 1, 8, 64 | Online, mini-batch, and larger mini-batch |
| `depth` | 1, 2 | Number of hidden layers; see architecture note below |
| `activation` | sigmoid, tanh, relu | Applied to all hidden layers |
| `optimizer` | sgd, adam | See Section 5.2 for details |
| `init_scale` | 0.1, 1.0 | Multiplier applied after standard init; see Section 5.3 |

This gives **3 × 2 × 3 × 2 × 2 = 72 categorical combinations** for supervised MLP tasks.

RL tasks omit `batch_size` (not applicable to online Q-learning) and instead include `gamma` (discount factor: 0.9, 0.99).

#### RNN tasks

| Parameter | Values |
|---|---|
| `batch_size` | 8, 64 |
| `cell_type` | rnn (Elman), gru |
| `n_rnn_layers` | 1, 2 |
| `optimizer` | sgd, adam |
| `init_scale` | 0.01, 1.0 |

This gives **2 × 2 × 2 × 2 × 2 = 32 categorical combinations** for RNN tasks.

### 3.2 Continuous hyperparameters

All four continuous hyperparameters are optimised on a log scale. The BO encodes them as a unit-normalised log value in [0, 1] internally.

| Parameter | Range | Scale | Notes |
|---|---|---|---|
| `hidden_size` | [4, 1024] (supervised); task-specific otherwise | log | Rounded to nearest integer after decoding; see architecture note |
| `learning_rate` | [1×10⁻⁵, 1×10⁻¹] | log | Passed directly to optimizer |
| `l1_reg` | [1×10⁻⁶, 1×10⁻²] | log | Coefficient on explicit L1 penalty; applied to weight matrices only |
| `l2_reg` | [1×10⁻⁶, 1×10⁻²] | log | Passed as `weight_decay` to optimizer |

`hidden_size` ranges are task-specific and set conservatively to avoid very large networks on simple tasks (e.g., spirals and parity cap at 256).

---

## 4. Bayesian Optimisation

### 4.1 Strategy

The BO uses a **stratified round-robin** approach to ensure balanced coverage of the categorical space:

1. At each iteration, select the categorical combo with the fewest observations so far (ties broken by combo index, i.e., lexicographic order on the first pass).
2. Given the selected categorical combo, choose continuous hyperparameters either via Sobol sampling (first 20 total observations) or GP acquisition (thereafter).

This decouples categorical and continuous search: every categorical combo is visited at roughly the same rate, while the GP refines the continuous values per combo as data accumulates.

### 4.2 Gaussian Process

We use **`MixedSingleTaskGP`** from BoTorch, which handles mixed continuous/categorical inputs natively. The GP sees all observations jointly — it is not separate per categorical combo — which allows it to share information across combos.

**Input encoding:**
- Continuous dimensions: log-transformed and normalised to [0, 1]
- Categorical dimensions: integer indices (0, 1, 2, ...) treated as categorical by the GP kernel

**Acquisition function:** q-Upper Confidence Bound (qUCB) with β = 8.0 (default; tunable via `--beta`). A high β favours exploration.

**Acquisition optimisation:** `optimize_acqf_mixed` from BoTorch. The categorical dimensions are fixed to the selected combo (via `fixed_features_list`); optimisation runs over the 4 continuous dimensions only. Parameters: 10 restarts, 128 raw samples.

**MLL fitting:** `ExactMarginalLogLikelihood`, fit via `fit_gpytorch_mll` (BoTorch default L-BFGS-B).

### 4.3 Initialisation

The first **N_SOBOL = 20** configurations use a Sobol sequence for the continuous dimensions (scrambled, with `seed = len(observations)` at the time of suggestion — deterministic given run order). This ensures space-filling coverage before the GP is fitted.

### 4.4 Scoring

The metric used as the BO objective is `mean_metric`: the mean raw validation metric across `runs_per_config` repetitions of a given configuration. Raw values are passed directly to the GP with no penalty or thresholding, so the GP sees the full gradient from chance-level performance up through successful networks. The `success_threshold` is used only for reporting (flagging runs as OK or FAILED in the console output).

### 4.5 State persistence

After every iteration, the full observation history is written to `bo_state.json` in the task's output directory. This makes runs **resumable after interruption**: on restart, the script loads existing observations and continues from where it left off. The Sobol seed is deterministic given `len(observations)`, so the sequence is reproducible.

---

## 5. Model Architectures

### 5.1 MLP (supervised and RL tasks)

A fully connected feedforward network with the following width schedule:

```
input → H → H//2 → H//4 → ... → output
```

where H is `hidden_size` and each successive layer halves the width. The number of hidden layers equals `depth` (1, 2, or 3). No dropout.

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

Each network is trained for up to `max_epochs` epochs. Task-specific epoch limits (where set) override the global default of 100.

| Task | Max epochs |
|---|---|
| Spirals | 300 |
| All others (supervised MLP) | 100 (global default) |

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

### 6.2 Optimizers

| Optimizer | Parameters |
|---|---|
| SGD | `lr=learning_rate`, `momentum=0.9`, `weight_decay=l2_reg` |
| Adam | `lr=learning_rate`, `weight_decay=l2_reg` (betas at PyTorch defaults: 0.9, 0.999) |

### 6.3 Weight initialisation

All linear layers (including the RNN readout head) are initialised with:
- **ReLU networks:** Kaiming normal (`fan_in` mode)
- **Sigmoid / tanh / RNN networks:** Xavier normal

After standard initialisation, all weights are **scaled by `init_scale`** (multiplicative). `init_scale` ∈ {0.01, 1.0}. Biases are always initialised to zero.

This means `init_scale = 0.01` produces near-zero initial weights (strong regularisation effect at initialisation), while `init_scale = 1.0` uses the standard initialisation directly.

### 6.4 Early stopping

- **Minimum epochs:** 15. Early stopping is not considered before this.
- **Patience:** 10 epochs without improvement in validation *loss* (threshold 1×10⁻⁴). Note: early stopping watches `val_loss`, not `val_acc`.
- **Best model:** tracked separately by `val_acc` (or the task's `metric_name`). The best checkpoint is saved to `model_best.pt`.

### 6.5 Activation checkpoints

Hidden-layer activations are saved on the fixed RDM stimulus set at **log₄-spaced training steps**: steps 1, 4, 16, 64, 256, 1024, 4096, … up to and including the final training step. This gives approximately equal coverage per order of magnitude of training progress.

Additionally, activations are always saved at the **final step** (current weights at end of training) and by **reloading `model_best.pt`** and saving those as `best.npz`.

**MLP:** post-activation outputs of each hidden layer are saved as `layer_0`, `layer_1`, ... Each array has shape `(N_stimuli, hidden_size_of_that_layer)`.

**RNN:** hidden states at a task-specific subset of time steps (to limit storage). Arrays are keyed `t_0`, `t_5`, etc. For MNIST-RNN: steps [0, 5, 11, 17, 22, 27]. For Adding: steps [0, 4, 9, 19, 34, 49].

---

## 7. Output Files

For each trained network, the following files are written under `experiments/<task>/run_NNNN_rR/`:

| File | Contents |
|---|---|
| `metadata.json` | Task name, full config (including `effective_depth`), best epoch/step, best metric, final epoch/step, final metric |
| `history.json` | Per-epoch: epoch number, global step, train loss, val loss, val acc |
| `model_best.pt` | PyTorch state dict at the epoch of peak val acc |
| `step_XXXXXXX.npz` | Activations on RDM stimuli at global step XXXXXXX (one file per checkpoint) |
| `best.npz` | Activations from `model_best.pt` weights |
| `final.npz` | Activations from end-of-training weights |

At the task level, `experiments/<task>/bo_state.json` stores the full observation history:

```json
[
  {
    "iteration": 0,
    "config": { ... },
    "val_accs": [0.923],
    "mean_metric": 0.923
  },
  ...
]
```

`mean_metric` is the scored mean (with penalty applied for failed runs). `val_accs` are the raw values before scoring.

---

## 8. Reproducibility Notes

- **RDM stimuli** are generated from fixed seeds (`seed=42` throughout, except the Adding task which uses `seed=200` for stimuli to decouple from the training data seed). Stimuli are identical across all runs for a given task.
- **Sobol initialisation** uses `seed = len(observations)` at the time of the call. Given a fixed run order and no interruptions, this is fully deterministic. After interruption and resume, the seed correctly reflects completed observations, preserving the sequence.
- **Training data** is generated/loaded with `seed=42` for train splits and `seed=43` for val splits (where applicable). MNIST and Fashion-MNIST are downloaded from standard sources; the train/val split uses `sklearn.model_selection.train_test_split` with `random_state=seed` and stratification by label.
- **No global random seed is set** during training. Results across runs of the same config will vary (this is intentional — the BO runs 2 repetitions per config by default to separate stochastic from hyperparameter-driven variance).
- `effective_depth` (the actual number of hidden layers used, after the small-network cap) is recorded in `metadata.json` alongside the requested `depth`.
