# Roadmap

## 1. Smoke test the BO

Before committing to full runs, verify the BO machinery works end-to-end:

- Run `run_task.py --task mnist_dual --n-iter 30 --runs-per-config 1` and inspect `bo_state.json`.
- Check that categorical combos are visited in round-robin order.
- After N_SOBOL=20 iterations, check that the GP is actually fitting (MLL should decrease, suggested configs should cluster around good combos).
- Check that the state file survives interruption and correctly resumes.
- Check that `metadata.json`, `history.json`, and `step_*.npz` files are written with the right shapes.

---

## 2. Smoke tests with bad/mediocre configs

The BO should explore the full performance distribution, not just good configs. Manually run a few known-bad configs per task to verify they train, fail gracefully, and are correctly scored as penalty values:

- Very small hidden_size + sigmoid activation (near-zero gradients)
- SGD with high learning rate (divergence)
- Batch size 1 with deep network (noisy gradients, slow)

Confirm the penalty value is correctly returned and stored in `bo_state.json`.

---

## 3. Timing estimates

Before scheduling cloud runs, get a per-task timing estimate:

- Time one full iteration (train + eval + RDM save) for each task at a representative config.
- Estimate total compute for a full BO run (e.g. 200 iterations × 2 runs per config).
- This determines instance type and cloud budget.

---

## 4. Cloud compute setup

- Choose cloud provider and instance type (GPU vs CPU — most tasks are small enough for CPU).
- Write a launch script that installs dependencies, clones the repo, and calls `run_task.py` with the right arguments.
- Decide on data persistence: upload `experiments/<task>/` to cloud storage after each iteration, or at the end of the run.
- One instance per task in parallel, or sequential tasks on one instance?

---

## 5. Smoke test cloud compute

- Run 5 iterations of one task on the cloud instance.
- Verify that output files (`metadata.json`, `history.json`, `step_*.npz`, `best.npz`, `final.npz`) are written and retrievable.
- Verify the run can be interrupted and resumed correctly.

---

## 6. Verify activation saving

Before analysis, manually inspect the saved activations:

- Load `step_*.npz` for a trained network and check shapes.
- Confirm stimuli order is consistent across checkpoints and across runs.
- For RNN tasks, confirm time-step indexing matches `rdm_time_indices`.
- For MLP tasks, confirm layer indexing (`layer_0`, `layer_1`, …).
- Compute a quick RDM by hand and sanity-check it against what you'd expect (e.g. for MNIST, digit-0 reps should cluster together).

---

## 7. Analysis

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
