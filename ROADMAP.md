# Roadmap

## 1. Additional task: Elman RNN (task TBD)

Add a 10th task using a vanilla RNN (Elman) architecture. The task should require genuine temporal integration and be distinct from the adding problem. Candidates:

- **Delayed XOR**: two binary inputs separated by a variable delay; output their XOR. Tests whether the network learns to hold a single bit across time.
- **Temporal parity**: binary sequence of length T; output parity of the whole sequence. Similar to the 12-bit parity task but in the temporal domain.
- **Copy task**: reproduce a short sequence after a blank delay period.

The task should be defined before this is implemented. Once defined, add `tasks/elman_task.py` following the existing pattern, and confirm it trains.

---

## 2. Per-layer activation saving (MLP)

Currently `save_activations_mlp` saves one activation set per checkpoint, but the MLP already exposes `get_layer_activations()` which returns all hidden layers. We want representations at each depth, not just the final hidden layer.

- Update `save_activations_mlp` in `src/rdm.py` to save per-layer activations (keyed by layer index).
- Storage format: one `.npz` per checkpoint, containing `layer_0`, `layer_1`, ... arrays of shape `(N_stimuli, H_l)`.
- The RNN already saves per-timestep; depth is the analogous axis for MLPs.
- Affects all 7 supervised + RL tasks. Implement before any full BO run to avoid needing to re-collect data.

---

## 3. Smoke test the BO

Before committing to full runs, verify the BO machinery works end-to-end:

- Run `run_task.py --task mnist_dual --n-iter 30 --runs-per-config 1` and inspect `bo_state.json`.
- Check that categorical combos are visited in round-robin order.
- After N_SOBOL=20 iterations, check that the GP is actually fitting (MLL should decrease, suggested configs should cluster around good combos).
- Check that the state file survives interruption and correctly resumes.
- Check that `training_curves.npz` and `activations_step_*.npz` files are written with the right shapes.

---

## 4. Smoke tests with bad/mediocre configs

The BO should explore the full performance distribution, not just good configs. Manually run a few known-bad configs per task to verify they train, fail gracefully, and are correctly scored as penalty values:

- Very small hidden_size + sigmoid activation (near-zero gradients)
- SGD with high learning rate (divergence)
- Batch size 1 with deep network (noisy gradients, slow)

Confirm the penalty value is correctly returned and stored in `bo_state.json`.

---

## 5. Timing estimates

Before scheduling cloud runs, get a per-task timing estimate:

- Time one full iteration (train + eval + RDM save) for each task at a representative config.
- Estimate total compute for a full BO run (e.g. 200 iterations × 2 runs per config).
- This determines instance type and cloud budget.

---

## 6. Cloud compute setup

- Choose cloud provider and instance type (GPU vs CPU — most tasks are small enough for CPU).
- Write a launch script that installs dependencies, clones the repo, and calls `run_task.py` with the right arguments.
- Decide on data persistence: upload `experiments/<task>/` to cloud storage (S3/GCS) after each iteration, or at the end of the run.
- One instance per task in parallel, or sequential tasks on one instance?

---

## 7. Smoke test cloud compute

- Run 5 iterations of one task on the cloud instance.
- Verify that output files (training curves, activations, bo_state.json) are written and retrievable.
- Verify the run can be interrupted and resumed correctly.

---

## 8. Verify activation saving

Before analysis, manually inspect the saved activations:

- Load `activations_step_*.npz` for a trained network and check shapes.
- Confirm stimuli order is consistent across checkpoints and across runs.
- For RNN tasks, confirm time-step indexing matches `rdm_time_indices`.
- For MLP tasks (once per-layer saving is added), confirm layer indexing.
- Compute a quick RDM by hand and sanity-check it against what you'd expect (e.g. for MNIST, digit-0 reps should cluster together).

---

## 9. Analysis

### RDM generation
- For each network × checkpoint (× layer for MLP, × timestep for RNN): compute pairwise dissimilarity matrix (1 − Pearson correlation across units is standard; consider also Euclidean).
- Aggregate across runs per config (mean RDM or mean dissimilarity).

### Second-order RSA
- Compare RDMs across: tasks, checkpoints, layers/timesteps, hyperparameter configs.
- Use Kendall's τ_a or Spearman correlation as the second-order similarity metric.
- Key questions:
  - Do tasks with similar structure (e.g. MNIST dual vs MNIST 10-way) produce similar representational geometries?
  - How does geometry evolve with training (checkpoint axis)?
  - How does depth affect geometry (layer axis)?
  - Which hyperparameters predict geometry, and which predict performance?

### Visualization
- MDS or t-SNE of RDMs in second-order space.
- Dendrogram / hierarchical clustering of RDMs.
- Geometry-performance scatterplots.

### Open questions to resolve during analysis
- What is the right dissimilarity metric (correlation distance vs Euclidean vs cosine)?
- How to handle variable network size (different H across networks) — representational geometry is size-invariant by construction if using correlation distance, but worth verifying.
- How to handle the layer axis for networks of different depths.

---

## Open issues

- **Degenerate hyperparameter combinations**: sigmoid activation + deep network may produce vanishing gradients on some tasks. Monitor whether these configs consistently fail and whether the penalty value correctly represents them.
- **BO convergence**: after a full run, check whether the GP acquisition function has actually converged (i.e. the suggested configs are repeating or have tight confidence intervals). If not, more iterations may be needed.
- **Reproducibility**: the Sobol initialisation uses `seed=len(observations)` which is deterministic given the run order. Document this so results can be reproduced.
- **Multiple runs per config**: currently 2 runs per config by default. Is variance across runs large enough to warrant this cost? Could be assessed after initial data collection.
