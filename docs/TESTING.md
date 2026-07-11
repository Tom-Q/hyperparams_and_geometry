# Testing plan (pre-AWS verification)

Three checks, in order. None of these touch `src/`, `tasks/`, or `run_bo.py` —
they are read-only verification scripts living under `tests/` / `scripts/`.

---

## 1. Per-task spec verification (local, no training)

For each task, generate data via `get_data()` / `get_rdm_stimuli()` and assert
properties derived **independently from the spec/docstring**, not from the
implementation. Pure data inspection — runs in seconds, no GPU.

| Task | What to check |
|---|---|
| **adding** | `x.shape == (N, 25, 2)`; values `x[:,:,0] ∈ [0,1]`; flags `x[:,:,1]` are 0/1 with exactly 2 ones per row; `y == (x[:,:,0]*x[:,:,1]).sum(axis=1)` recomputed independently; `chance_perf=-0.1667` ≈ `-Var[v1+v2]` for two `U(0,1)` draws (Monte Carlo check) |
| **mnist_rnn** | `x.shape == (N, 14, 56)`; each step = 2 MNIST rows (28×2=56), pixel values in `[0,1]`; reshape back to `(N,28,28)` and visually/statistically confirm it's a valid digit image; `rdm_stimuli` = 100 = 10 digits × 10 exemplars, `digits` metadata balanced 10/digit |
| **mnist_dual** | `x.shape == (N, 785)`; last column (task bit) ∈ {0,1}; label = even/odd (task bit 0) or <5/≥5 (task bit 1), recomputed independently from MNIST labels; `rdm_stimuli` = 200 = 10 digits × 10 exemplars × 2 task bits |
| **mnist_10way / fashion_10way** | `x.shape == (N, 784)`, values in `[0,1]`; `y ∈ {0..9}`; `rdm_stimuli` = 100 = 10 classes × 10 exemplars, `digits`/`classes` balanced |
| **parity** | all `2^8=256` patterns enumerated exactly once; `label == popcount(pattern) % 2` recomputed independently; `rdm_stimuli` stratified by `n_ones` (≤20 per level, 9 levels) |
| **spirals** | `x.shape == (N,2)`, 3 classes balanced (1000/class train, 200/class val); recompute spiral arm assignment from `(x1,x2)` angle/radius and confirm it matches `y`; `rdm_stimuli` = 198 = 3×66 noise-free points lying exactly on the 3 spiral curves |
| **cartpole** | `rdm_stimuli` = 196 = 14×14 grid over `(pole_angle ∈ [-0.2,0.2], pole_vel ∈ [-2,2])`, other dims = 0; `env_factory()` produces a `CartPole-v1` env with `obs.shape==(4,)`, `action_space.n==2` |
| **fourrooms** | grid layout matches the hardcoded `GRID` (4 rooms, walls); `N_RBF == len(FREE_CELLS)`; `_rbf_encode` output sums/peaks correctly at the encoded cell; `reset()` never starts on a wall or the goal; `step()` reward always `-0.01`, `done` iff goal reached or `max_steps` hit |
| **all RNN tasks** | `rdm_time_indices` are valid indices `< n_steps` and increasing |
| **all tasks** | `categorical_space()` keys match what `make_optimizer`/`MLP`/`RNNModel` expect (`activation`, `optimizer`, `cell_type`, `n_rnn_layers`, `depth`, `init_scale`, etc.); continuous ranges (`LEARNING_RATE`, `L1_REG`, `L2_REG`) are sane (`low < high`, `low > 0`) |

**Output**: one script per task under `tests/spec/`, or one combined script that
prints a PASS/FAIL line per assertion per task. No model training involved.

---

## 2. Local smoke test: activation saving (small training run)

Goal: verify RDM activation recording/storage end-to-end, which has **not**
been checked before (training itself is known to work from the 180-network
CPU test run).

- Pick one task per paradigm (e.g. `spirals` for supervised/MLP, `adding` for
  RNN, `cartpole` for RL).
- Run `run_bo.py --n-iter 1 --no-repeats --max-epochs <small>` **without**
  `--no-save-activations` (i.e. activation saving ON, the AWS default).
- A checker script then walks the resulting `run_*` directory and verifies:
  - `metadata.json` / `history.json` exist with expected keys
    (`training_time_s`, `best_epoch`/`best_step`, `config`, etc.)
  - Checkpoint directories exist as predicted by `log4_checkpoints`,
    `epoch_checkpoints`, `perf_checkpoint_thresholds` (`step_*`, `epoch_*`,
    `perf_*`, `final/`, `best/`)
  - Each `.npz` has the expected keys:
    - MLP: `layer_0`, `layer_1`, ... matching `depth`
    - RNN: `t_<i>` for each `i` in `task.rdm_time_indices`, nothing else
  - Shapes are `(N_stimuli, hidden_size)` for every key
  - Values are finite, not all-zero (sanity that activations were actually
    captured)
  - `model_best.pt` loads and its `final`/`best` activations differ (since
    weights changed between best and final checkpoint, unless they coincide)

---

## 3. AWS smoke test (one script, one EC2 instance)

- One script that, on a single EC2 instance, runs `run_bo.py --n-iter 1
  --no-repeats` (activations ON) for **every task** sequentially (9 runs),
  using the real AWS entrypoint/paths/S3 sync config.
- Confirms: environment setup (deps, GPU/CPU device selection per task),
  data download/caching (MNIST/FashionMNIST), disk space for activations,
  S3 sync of `bo_state.json`, and that all 9 tasks complete without error.
- Re-uses the checker from #2 against each of the 9 `run_*` directories.
- This is the last gate before launching the full BO run at scale.
