# Activation Saving — Implementation Plan

## Checkpoint schedule (all paradigms)

Powers of 4 in **gradient steps**: 1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, …
up to and including the final step. No cap. For supervised/RNN, a gradient step = one
optimizer update (within an epoch). For RL, a gradient step = one environment step
(online Q-learning, batch=1).

---

## File structure per run

```
experiments/<task>/run_<id>/
    metadata.json       ← config, seeds, best/final epoch+step, test metric
    history.json        ← per-epoch or per-eval performance log
    model_best.pt       ← weights at best validation epoch/step
    step_1.npz
    step_4.npz
    step_16.npz
    ...                 ← one file per powers-of-4 checkpoint
    best.npz            ← activations from model_best.pt
    final.npz           ← activations from end-of-training weights
```

---

## NPZ format (activation files)

Each `.npz` file contains one array per hidden layer:

- **MLP / RL:** `layer_0`, `layer_1`, … each shape `(N_stimuli, hidden_size)`
- **RNN:** `layer_0`, `layer_1`, … each shape `(N_stimuli, n_time_steps, hidden_size)`,
  where `n_time_steps` is indexed by `task.rdm_time_indices`

Saved with `np.savez_compressed`.

---

## metadata.json

```json
{
  "task":         "mnist_dual",
  "paradigm":     "supervised",
  "config":       { "hidden_size": 256, "depth": 2, "..." : "..." },
  "seed":         42,
  "best_epoch":   23,
  "best_step":    21551,
  "best_metric":  0.9823,
  "final_epoch":  33,
  "final_step":   30953,
  "final_metric": 0.9341,
  "test_metric":  0.9801
}
```

- `best_epoch` / `best_step`: epoch number and global gradient step at the end of
  the best validation epoch
- `final_epoch` / `final_step`: last epoch trained and its corresponding global step
- `test_metric`: test-set accuracy (supervised only) evaluated on `model_best.pt`
  after training ends; absent for RL tasks

For RL, `best_epoch` is absent; `best_step` is the environment step at which the
best mean return was recorded.

---

## history.json

Full performance log throughout training.

**Supervised / RNN** — one record per epoch:
```json
[
  { "epoch": 1,  "step": 937,   "train_loss": 0.45, "val_loss": 0.32, "val_acc": 0.891 },
  { "epoch": 2,  "step": 1874,  "train_loss": 0.31, "val_loss": 0.28, "val_acc": 0.912 },
  { "epoch": 3,  "step": 2811,  "train_loss": 0.27, "val_loss": 0.27, "val_acc": 0.921 }
]
```

For MSE tasks (Adding problem), `val_acc` is replaced by `val_mse`.

**RL** — one record per `eval_interval` steps:
```json
[
  { "step": 1000,  "mean_return": 12.4 },
  { "step": 2000,  "mean_return": 18.7 },
  { "step": 3000,  "mean_return": 24.1 }
]
```

---

## Training loop behaviour

### Supervised / RNN

1. Track `global_step` (incremented after each optimizer update)
2. After each update: if `global_step` in checkpoint set → save `step_{k}.npz`
3. After each epoch:
   - Compute `train_loss`, `val_loss`, `val_acc` (or `val_mse`)
   - Append record to history list
   - If val metric improved: save `model_best.pt`, record `best_epoch` + `best_step`
   - Early stopping check (patience on `val_loss`, unchanged)
4. At end of training:
   - Load `model_best.pt` → evaluate on test set → record `test_metric`
   - Load `model_best.pt` → forward pass on RDM stimuli → save `best.npz`
   - Current weights → forward pass on RDM stimuli → save `final.npz`
   - Write `metadata.json` and `history.json`

### RL

1. Track `global_step` (environment steps)
2. After each step: if `global_step` in checkpoint set → save `step_{k}.npz`
3. Every `eval_interval = 1000` steps:
   - Run 10 evaluation episodes (ε=0), compute `mean_return`
   - Append record to history list
   - If `mean_return` improved: save `model_best.pt`, record `best_step`
4. At end of training:
   - Load `model_best.pt` → forward pass on RDM stimuli → save `best.npz`
   - Current weights → forward pass on RDM stimuli → save `final.npz`
   - Write `metadata.json` and `history.json`

RL parameters:
- `eval_interval = 1000` steps (both CartPole and FourRooms)
- `max_steps = 500_000` (CartPole), `2_000_000` (FourRooms)

---

## Model weights

Keep `model_best.pt` after training. Size is 50 KB–5 MB depending on hidden_size.
Useful for: post-hoc probing with new stimuli, probing classifiers, behavioural
experiments.

No test-set evaluation for RL (no held-out set; environment is generative).

---

## Code changes required

| File | Change |
|---|---|
| `src/utils.py` | Remove any cap on `log4_checkpoints` |
| `src/model_mlp.py` | Add `get_layer_activations(x)` returning list of per-layer hidden outputs |
| `src/model_rnn.py` | Verify `get_step_activations` returns per-layer, per-step hidden states |
| `src/rdm.py` | New `save_activations(model, stimuli, path, task)` writing per-layer NPZ |
| `src/train_supervised.py` | `global_step` counter; within-epoch checkpoint saving; best model saving; end-of-training best/final/test pass; write metadata + history |
| `src/train_rnn.py` | Same as supervised |
| `src/train_rl.py` | Step-based eval every 1000 steps; best model saving; end-of-training best/final pass; write metadata + history |
