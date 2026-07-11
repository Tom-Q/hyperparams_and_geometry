# RDM Computation Plan

## Key decisions

| Decision | Choice | Rationale |
|---|---|---|
| Distance metric | Cosine distance | Confirmed from Devolder et al. (2026); measures directional similarity, insensitive to activation magnitude |
| RSA comparison | Spearman rank correlation | Confirmed from Devolder et al. (2026); robust to non-linearities in dissimilarity scale |
| Storage format | HDF5 (one file per task) | Supports efficient access by network (trajectory analysis) and by checkpoint (cross-network analysis); handles partial writes; resumable |
| Numerical precision | float32 | Halves storage vs. float64 with negligible precision loss for downstream correlations |
| Representation stored | Upper triangle only (no diagonal) | Diagonal is always 0; lower triangle is redundant |

---

## Distance metric

For each pair of stimuli (i, j), compute:

```
d(i,j) = 1 − cos(a_i, a_j) = 1 − (a_i · a_j) / (‖a_i‖ ‖a_j‖)
```

where a_i is the activation vector for stimulus i. Use `sklearn.metrics.pairwise.cosine_distances` (vectorized, handles the full stimulus matrix in one call).

**Note:** cosine distance is sensitive to near-zero activation vectors (undefined cosine for zero vectors). If any activation vector has norm < 1e-8, flag the network and exclude from analysis. This can occur for sigmoid/tanh networks with very small init_scale or heavy regularization.

---

## What activations to use

### MLP (supervised and RL tasks)

Each `.npz` file contains `layer_0`, `layer_1`, ... with shapes `(N_stimuli, hidden_size_of_that_layer)`.

- Compute one RDM **per layer** independently.
- For depth=1 networks: one RDM (layer_0).
- For depth=2 networks: two RDMs (layer_0 at width H, layer_1 at width H//2).
- **Do not concatenate layers.** Concatenation mixes representational spaces of different dimensionality; cosine distance would weight them unequally. Keeping layers separate lets the data answer which layer produces more reliable/informative RDMs (Finding #1.4).
- **Default for cross-network analyses:** last hidden layer (layer_0 for depth=1, layer_1 for depth=2). Report sensitivity to this choice where relevant.

### RNN tasks (mnist_rnn, adding)

Each `.npz` file contains hidden states at **all** time steps (despite METHODS.md listing a subset — the implementation saves the full sequence). Keys follow the format `layer_{rnn_layer}_t_{timestep}`:
- mnist_rnn: `layer_0_t_0` … `layer_0_t_13` and (for n_rnn_layers=2) `layer_1_t_0` … `layer_1_t_13` — 14 steps × up to 2 layers = up to 28 keys per checkpoint
- adding: `layer_0_t_0` … `layer_0_t_24` and `layer_1_t_0` … `layer_1_t_24` — 25 steps × up to 2 layers = up to 50 keys per checkpoint

- Compute one RDM per (rnn_layer × time_step) combination. Store all of them.
- **Default for cross-network analyses:** final time step, last RNN layer (t_13/layer_X for mnist_rnn, t_24/layer_X for adding, where X = n_rnn_layers − 1). Analogous to the last hidden layer in MLPs and directly comparable to the previous paper which used the final hidden state.
- Within-sequence temporal analysis (how the RDM evolves across time steps within a single forward pass) is a secondary analysis using the stored per-timestep RDMs.

### RL tasks (cartpole, fourrooms)

Same structure as supervised MLPs. RL tasks have no `batch_size` continuous dim and no epoch checkpoints, but otherwise identical activation structure.

---

## Checkpoints to process

Process all checkpoint types for completeness. Analyses will specify which subset they use.

| Checkpoint type | Files | Used in |
|---|---|---|
| Best / final | `best.npz`, `final.npz` | All static analyses (Findings #1, #2) |
| Performance milestones | `perf_0p025.npz` … `perf_0p95.npz` | Crystallization (3.1), trajectory mapping (3.4), early prediction (4.1) |
| Epoch milestones | `epoch_0p25.npz` … `epoch_64.npz` | RDM vs. loss prediction (4.2); supervised and RNN only |
| Step checkpoints | `step_1.npz`, `step_4.npz`, … | Critical period / change rate (3.2) |

**For the initial implementation**, prioritise `best.npz` / `final.npz` so Findings #1 and #2 can begin immediately. Add remaining checkpoints in a second pass.

---

## Storage schema

One HDF5 file per task: `analysis/rdms/{task}_rdms.h5`

```
{task}_rdms.h5
│
├── meta/
│   ├── n_stimuli          (int scalar)
│   ├── n_pairs            (int scalar — upper triangle size)
│   └── stimulus_labels    (string array — task-specific labels, e.g. "digit_0_exemplar_3")
│
└── runs/
    └── {run_id}/          # e.g. "run_0042_r0"
        ├── iteration      (int)
        ├── is_repeat      (bool)
        ├── performance    (float)
        ├── depth          (int — 1 or 2)
        ├── [hp config attrs: optimizer, activation, lr, ...]
        │
        └── {checkpoint}/  # e.g. "best", "final", "perf_0p1", "epoch_4", "step_64"
            ├── layer_0    (float32, shape: n_pairs)   # always present
            ├── layer_1    (float32, shape: n_pairs)   # depth=2 only
            └── t_{k}      (float32, shape: n_pairs)   # RNN only, one per time step
```

`run_id` matches the directory name under `output/experiments/{task}/`. HDF5 attributes on the group store the HP config so it's queryable without loading the RDMs.

---

## Implementation

### Script: `analysis/10_compute_rdms.py`

**Arguments:** `--task` (required), `--checkpoints` (default: `best final`), `--overwrite` (flag).

**Logic per network:**
1. Load `metadata.json` to get HP config and depth.
2. For each requested checkpoint: locate the `.npz` file; skip with a warning if missing.
3. For each layer / time step key in the `.npz`:
   - Load activation matrix `(N_stimuli, D)`.
   - Check for near-zero vectors; flag if any found.
   - Compute cosine distance matrix `(N_stimuli, N_stimuli)`.
   - Extract upper triangle (excluding diagonal).
   - Write to HDF5 under `runs/{run_id}/{checkpoint}/layer_X`.
4. Write HP metadata as HDF5 attributes on the `runs/{run_id}` group.

**Resumability:** before processing a network, check whether `runs/{run_id}/{checkpoint}/layer_0` already exists in the HDF5. Skip if present (unless `--overwrite`). This allows the script to be killed and restarted safely.

**Parallelism:** the bottleneck is disk I/O (loading `.npz` files). Use `concurrent.futures.ThreadPoolExecutor` with ~4 threads to overlap I/O. HDF5 writes are serialized (h5py is not thread-safe for writes without locking). Alternatively, process tasks in parallel as separate processes.

### Script: `analysis/10b_rdm_summary.py`

After computing RDMs, run a quick summary script that prints:
- Per-task: how many networks have RDMs for each checkpoint type.
- Any networks flagged for near-zero activations.
- File sizes of the HDF5 files.

---

## Storage estimate

Measured from actual files (all checkpoints, all layers/timesteps, float32 upper triangles):

| Task | Runs | Ckpts | Keys/ckpt | N_stimuli | Est. total |
|---|---|---|---|---|---|
| adding_failed_run | 989 | 25 | 50 (25 steps × 2 layers) | 100 | ~24.5 GB |
| mnist_dual | 884 | 26 | 2 | 200 | ~3.7 GB |
| spirals | 1000 | 19 | 2 | 198 | ~3.0 GB |
| cartpole | 1000 | 15 | 2 | 196 | ~2.3 GB |
| mnist_rnn | 200 | 13 | 28 (14 steps × 2 layers) | 100 | ~1.4 GB |
| parity | 1000 | 25 | 1 | 118 | ~0.7 GB |
| mnist_10way | 1000 | 27 | 1 | 100 | ~0.5 GB |
| fashion_10way | 1000 | 24 | 1 | 100 | ~0.5 GB |
| fourrooms | 1000 | 17 | 1 | 61 | ~0.15 GB |
| **Total** | | | | | **~37 GB** |

37 GB is well within the 200 GB available. Store everything — no need to subset checkpoints or time steps. The adding task dominates due to 50 RDMs per checkpoint (25 time steps × 2 layers).

---

## Degenerate networks

Some networks produce NaN or near-zero activation vectors (norm < 1e-8), making cosine distance undefined. This can occur due to numerical instability during training (exploding/vanishing activations), or full weight collapse under heavy regularization.

**How degenerate entries are stored:**
- The HDF5 dataset for a degenerate (run, checkpoint, key) is written as a zero-length array (`shape=(0,)`) with attribute `degenerate=True`.
- This distinguishes "degenerate" from "checkpoint not reached" (which simply has no dataset).
- All degenerate paths are also listed in `meta/flagged` (string dataset) and counted in `meta/n_flagged` for easy inspection.

**Downstream handling:**
```python
ds = h5[f"runs/{run_id}/{checkpoint}/{key}"]
if ds.attrs.get("degenerate", False) or len(ds) == 0:
    continue  # skip this entry
rdm = ds[:]
```

Analyses should exclude degenerate entries. A network whose `best.npz` or `final.npz` last-hidden-layer key is degenerate should be excluded entirely from that analysis. Degenerate entries at intermediate checkpoints (step, epoch, perf) only exclude that specific timepoint.

## Validation checks

Before running analyses, verify:
1. **Symmetry sanity:** sample a few RDMs, confirm upper triangle values are in [0, 2] (cosine distance range). Values outside this indicate a bug.
2. **Self-distance:** diagonal of the cosine distance matrix should be 0. (Not stored, but verify during computation.)
3. **Repeat consistency:** for each repeat pair (same HP config, different seed), compute RDM correlation. Distribution should be higher than random pairs — if not, something is wrong with the activation loading.
4. **Layer ordering:** for depth=2 networks, confirm layer_0 has shape `(N_stimuli, H)` and layer_1 has shape `(N_stimuli, H//2)` where H is `hidden_size` from the config.

---

## Z-scoring

**Store raw cosine distances. Apply z-scoring only within the PCA script.**

Z-scoring an RDM vector (subtract mean, divide by std) removes the information about overall dissimilarity magnitude — how spread out the representations are on average. This destroys real information: ReLU networks genuinely have higher cosine distances than tanh/sigmoid networks, and that is a scientifically meaningful difference.

For most analyses, z-scoring is irrelevant:
- **Spearman RSA** (noise ceiling, category structure, cross-network correlations): Spearman is rank-based; any monotonic transform including z-scoring leaves the result identical.
- **Mean dissimilarity, participation ratio**: require raw values; z-scoring would destroy the signal.
- **UMAP of networks** (distance = 1 − Spearman): rank-based, unaffected.

For PCA specifically, z-scoring is necessary: without it, PC1 captures "ReLU networks have uniformly higher cosine distances" — a magnitude effect, not a geometric one. The previous paper z-scored before PCA for exactly this reason. Apply the same normalisation within the PCA script (Finding #2.3) to ensure comparability.

---

## Relation to previous paper

Devolder et al. (2026) used:
- Cosine distance RDMs ✓ (same)
- Spearman rank correlation for RSA ✓ (same)
- Single hidden layer only (depth=1 by design)
- One task (MNIST dual), one stimulus set (100 stimuli: 5 exemplars × 10 digits × 2 task contexts)
- RDMs row-wise z-scored before PCA only

For direct comparison: use `best.npz`, last hidden layer, mnist_dual task. The stimulus set here is 200 stimuli (10 exemplars × 10 digits × 2 task contexts) vs. 100 in the previous paper (5 exemplars). Results should be qualitatively comparable but not numerically identical.
