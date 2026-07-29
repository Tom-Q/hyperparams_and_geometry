# CIFAR-10 BO Experiment Plan

Goal: train ~200 CNN+FC networks on CIFAR-10 using saturating Bayesian optimisation
(same BO infrastructure as existing tasks) to extend the representational geometry
analysis to a harder image classification task and deeper FC architectures (up to 3
hidden layers).

---

## Architecture

**Conv frontend (fixed, not varied):**
```
Conv2d(3→32, kernel=3, padding=1) → ReLU → MaxPool(2×2)   # output: 32×16×16
Conv2d(32→64, kernel=3, padding=1) → ReLU → MaxPool(2×2)  # output: 64×8×8
Flatten → 4096
```
Always ReLU in conv layers, always the same architecture across all runs.

**FC backend (varied):**
```
Linear(4096, H) → act
[Linear(H, H) → act]  × (n_fc_layers - 1)
Linear(H, 10)          ← output, no activation
```
Where `H = hidden_size`, `act` = relu or tanh (uniform across all FC layers).
So n_fc_layers=1 is a single hidden layer: 4096→H→10.

---

## Hyperparameter space

| HP | Type | Values / Range | Notes |
|----|------|----------------|-------|
| n_fc_layers | ordinal categorical | 1, 2, 3 | replaces `depth`; log-ordered in N_eff |
| activation | unordered categorical | relu, tanh | sigmoid excluded (too slow on CIFAR-10) |
| hidden_size | continuous (from cat) | [64, 512] log-scale | uniform across all FC layers |
| batch_size | continuous (from cat) | [16, 128] log-scale | wider low end for coverage |
| learning_rate | continuous | [1e-4, 1e-2] log-scale | Adam range; 1e-5 is too slow |
| l2_reg | continuous | [1e-6, 1e-3] log-scale | smaller max than default (1e-2); light regularisation |
| optimizer | fixed | adam | always Adam |
| l1_reg | fixed | 0.0 | excluded from BO; not a parameter |
| init_scale | fixed | 1.0 | did not matter on other tasks; fixed |

**Categorical combos**: n_fc_layers × activation = 3 × 2 = **6 combos**

**Continuous BO dimensions**: lr, l2, hidden_size, batch_size = **4 dims**

**N_SOBOL**: to be decided (≈36 seems reasonable given 6 categorical combos; 6 per combo).

**Total planned runs**: ~200 (primary + ~20% repeats = ~167 primary iterations).

---

## Key differences from existing tasks

- l1_reg is **excluded** from BO (fixed at 0.0) — existing tasks have it as a continuous dim
- learning_rate range is narrower: [1e-4, 1e-2] vs [1e-5, 1e-1]
- l2_reg range is narrower: [1e-6, 1e-3] vs [1e-6, 1e-2]
- hidden_size range: [64, 512] vs [16, 256]
- batch_size range: [16, 128] vs [1, 64]
- depth is replaced by n_fc_layers [1, 2, 3] (three levels instead of two)
- activation: relu and tanh only (no sigmoid)
- optimizer: fixed to adam (not a categorical HP)
- init_scale: fixed at 1.0 (not a categorical HP)
- Model class: CNNMLP (not MLP); train_supervised.py uses task.build_model(config)

---

## Stimulus set for RDMs

100 images: 10 exemplars × 10 CIFAR-10 classes, sampled from the **test set**
(never seen during training). Same approach as `fashion_10way.get_rdm_stimuli`.

CIFAR-10 classes (in dataset order): airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck.

`get_rdm_stimuli` returns:
- `inputs`: `(100, 3, 32, 32)` float32 array, values in [0, 1]
- `metadata`: `{"classes": array of 100 ints, shape (100,)}`

The 4D input shape is handled transparently by `stimuli_to_tensor` and
`save_activations_mlp` (both are shape-agnostic). No changes to `rdm.py`.

---

## Activations saved

FC hidden layers only: `layer_0`, `layer_1`, ..., `layer_{n_fc_layers-1}`.
Each is shape `(100, hidden_size)`.

**Not saved**: conv feature maps. The CNN frontend is treated as fixed
preprocessing; the RDM analysis studies the FC geometry, consistent with how
the existing MLP tasks work.

Checkpoint schedule (reusing existing utils):
- Log4 steps: 1, 4, 16, 64, ... up to total_steps
- Epoch checkpoints: 0.25, 1, 4, 16, 64
- Performance thresholds: 2.5%, 5%, 10%, 20%, 40%, 60%, 80%, 85%, 90%, 95%
  of the way from chance (0.1) to perfect (1.0)
- `best/` and `final/` at end of training

---

## Data handling

- Download via `torchvision.datasets.CIFAR10(download=True)`
- `CIFAR10.data` is `(N, 32, 32, 3)` uint8 numpy (channels-last); must be
  transposed to `(N, 3, 32, 32)` and divided by 255 → float32 in [0, 1]
- `CIFAR10.targets` is a Python list (not a tensor); use `np.array(targets)` for stratify
- Train/val split: 87.5% / 12.5% stratified (same ratio as `fashion_10way`)
- Test set: used for `get_rdm_stimuli` only, not for early stopping

---

## Directory structure

```
production/cifar10/
  bo_state.json                 # BO state (same format as existing tasks)
  run_0000_r0/
    metadata.json               # config, best_metric, best_epoch, timing
    history.json                # per-epoch val_acc, val_loss
    model_best.pt
    best/          *.npz        # layer_0 … layer_{n-1}, shape (100, H)
    final/         *.npz
    step_0000001/  *.npz
    epoch_1/       *.npz
    perf_0p4/      *.npz
    ...
  run_0001_r0/
    ...
```

Run with: `python run_bo.py --task cifar10 --n-iter 200 2>&1 | tee production/cifar10/run.log`

---

## New files

| File | Role |
|------|------|
| `src/model_cnn_mlp.py` | `CNNMLP` model class |
| `tasks/cifar10.py` | `Cifar10Task` task class |

### Modified files

| File | Change |
|------|--------|
| `tasks/__init__.py` | Add Cifar10Task |
| `tasks/base.py` | Add `build_model(config)` hook (default returns None) and `cont_param_ranges()` hook (default returns None) |
| `src/bo.py` | `_cont_params_for_task` calls `task.cont_param_ranges()` when not None; add `n_fc_layers` to ORDINAL_PARAMS |
| `src/train_supervised.py` | Use `task.build_model(config)` when it returns non-None; use `config.get("l1_reg", 0.0)` |
| `run_bo.py` | Make `l1` in cont_str conditional on `"l1_reg" in config` |

### `src/model_cnn_mlp.py`
- Fixed conv frontend (hardcoded, always ReLU)
- Uniform-width FC backend
- `get_layer_activations(x)` returns `{layer_0: ..., layer_1: ...}` — same
  interface as `MLP.get_layer_activations`, so `save_activations_mlp` works
  unchanged
- `_init_weights`: kaiming for relu FC layers, xavier for tanh; init_scale
  fixed at 1.0 (not a parameter)

### `tasks/cifar10.py`
- Inherits from `Task`; `input_size = (3, 32, 32)` as documentation
- `categorical_space()` returns n_fc_layers, hidden_size, batch_size, activation
- `cont_param_ranges()` returns [(lr, 1e-4, 1e-2), (l2, 1e-6, 1e-3)]
- `build_model(config)` returns a `CNNMLP` instance
- `success_threshold = 0.60` tentative — to be confirmed after first results
- `make_loss()` returns `nn.CrossEntropyLoss()`

---

## Storage estimate

Per checkpoint: 100 stimuli × 512 units × 3 FC layers × 4 bytes = 614 KB raw,
~80 KB compressed. With ~23 checkpoints per run: ~1.8 MB per network. At 200
runs: **~360 MB total**. Comfortable.

---

## Future work (not in this plan)

- Extend `10_compute_rdms.py` to handle CIFAR-10 (4D stimulus arrays, new
  task key)
- Integrate CIFAR-10 into the main analysis pipeline (scripts 11–25)
- Decide whether to add CIFAR-10 to `TASK_NAMES` in `analysis_utils.py`
