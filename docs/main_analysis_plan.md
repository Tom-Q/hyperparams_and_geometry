# Main Analysis Plan

## Technical foundations

### RDM computation
For each network and each checkpoint, compute the Representational Dissimilarity Matrix over the fixed stimulus set:
- **Distance metric:** cosine distance (1 − cosine similarity) between activation vectors across stimulus pairs.
- **Dual metric (not yet implemented):** A planned extension is to store both cosine distance and Pearson correlation distance (1 − Pearson r between two stimulus activation vectors across units) per network in the HDF5, using keys e.g. `layer_0` (cosine, current) and `layer_0_pearson`. Pearson correlation distance is mathematically equivalent to cosine distance on across-units mean-centred activations, and is the more common choice in ANN representational geometry literature (Kornblith, Raghu et al.) because it handles scale/offset differences across nonlinearities without explicit pre-processing. Practically, it fixes the sigmoid compression issue: sigmoid networks with large hidden sizes produce all-positive vectors that cluster in the same orthant, giving near-zero cosine distances even for well-trained networks; Pearson centering removes this shared offset. Compute cost of adding Pearson is negligible (mean-center then cosine); storage roughly doubles; downstream scripts would need a metric parameter to select which RDMs to load.
- **Layer convention:**
  - depth=1: one RDM from layer_0
  - depth=2: two RDMs, one from layer_0 (H units) and one from layer_1 (H//2 units)
  - When a single RDM per network is needed: use the **last hidden layer** (layer_0 for depth=1, layer_1 for depth=2). This is most comparable across architectures and closest to the output.
- **Checkpoint convention for static analyses (Findings #1 and #2):** use `best.npz` (supervised/RNN) or `final.npz` (RL) — peak-performance weights.

### RNN task temporal RDMs

For adding and mnist_rnn, the main analysis uses a single **temporal RDM** per network rather than separate per-timestep RDMs. This encodes the full temporal trajectory into one matrix, making these tasks directly comparable to single-RDM tasks throughout Findings #1 and #2.

**Adding — 600×600 temporal RDM.** Rows are (stimulus, phase) pairs: 100 stimuli × 6 phases, stimulus-major ordering. Each row is the average of the unit-normalised hidden state over all timesteps in that phase for that stimulus. Entry [(stim_i, phase_p), (stim_j, phase_q)] = 1 − (mean unit vector i,p) · (mean unit vector j,q), which equals the average cosine distance over all cross-timestep pairs in those two phases. **Note: valid only for cosine distance** (bilinear in unit vectors); does not generalise to other metrics.

The 6 phases (0-indexed): 0 = before flag1, 1 = at flag1, 2 = between flags, 3 = at flag2, 4 = after flag2, 5 = final step. Phases 0, 2, and 4 are absent for some stimuli (no valid timesteps), making those rows NaN. The NaN mask is fixed across all networks (depends only on the fixed stimulus set). Downstream scripts strip those pairs at load time — no NaN reaches any analysis function.

**mnist_rnn — 1400×1400 temporal RDM.** Rows are (stimulus, timestep) pairs: 100 stimuli × 14 timesteps, stimulus-major. No NaN — every stimulus has a valid hidden state at every timestep.

Both are stored under HDF5 key `temporal` (written by scripts 10b and 10c respectively), replacing the old `layer_{L}_t_{T}` key lookup for all main analyses. The per-timestep keys remain in the HDF5 for the temporal-dynamics sub-analysis (script 11b).

### Canonical performance field and success classification

**Always use `obs["performance"]` from `bo_state.json`** — never `best_metric` or `best_rolling` from `metadata.json`.

The thresholds in `success_thresholds.json` were computed from `obs["performance"]`. This field is the raw BO metric, which is consistent across tasks in sign convention (higher = better for the GP):

- Supervised tasks: `obs["performance"]` = validation accuracy (same as `best_metric` in metadata — those two happen to agree, but use the BO state as the source of truth anyway)
- RL tasks (`cartpole`, `fourrooms`): `obs["performance"]` = rolling mean return
- Adding: `obs["performance"]` = **negative MSE** (`train_rnn` negates MSE before returning it to the BO). `metadata.json` stores raw positive MSE as `best_metric` — a different quantity. Using `best_metric` directly gives wrong sign and breaks the threshold comparison entirely.

**Loading pattern** (use this in every analysis script that needs performance):
```python
def load_bo_perf(task):
    state_path = PROD_DIR / task / "bo_state.json"
    observations = json.load(open(state_path))
    return {f"run_{obs['iteration']:04d}_r0": obs["performance"]
            for obs in observations}
```
Run dir names follow the format `run_{iteration:04d}_r0` (zero-padded iteration, `_r0` suffix for the primary repetition).

**Success classification:**
- successful: `performance >= success_thresholds.json[task]["upper"]`
- near-chance: `performance <= success_thresholds.json[task]["lower"]`
- partial: between the two

### Network selection
- All analyses in Finding #1 use **primary networks only** (no repeats), across all performance categories unless specified.
- Finding #2 uses **successful networks only** (performance ≥ upper threshold) to ensure HP effects on representations aren't confounded by learning failure.
- The repeat pairs (~20% of observations, same config different seed) are reserved for the stochastic-vs-HP variance decomposition.

### Computational strategy
- All-pairs RDM comparisons: vectorise by stacking upper-triangle vectors into an N×D matrix; compute correlation matrix in batch (numpy). For very large N, subsample ~200 networks — sufficient for stable estimates.
- Per-task, not pooled across tasks. Each task is its own analysis unit.

---

## Finding #1 — Validity of RSA

**Core question:** Are RDMs reliable enough to support RSA conclusions?

### 1.1 Inter-network agreement (noise ceiling)
For each task, take a sample of successful networks. For each network, correlate its RDM with the mean RDM of all *other* networks in the sample (leave-one-out). The distribution and mean of these correlations = the noise ceiling.

- High mean, tight distribution → networks doing the same task converge on similar representations → RSA is reliable.
- Low mean or wide distribution → representations are idiosyncratic → RSA conclusions are fragile.

**Output:** one violin/box plot per task, mean correlation value per task as a summary statistic.

### 1.2 Stochastic vs. HP-driven variance
The repeat infrastructure gives us pairs of networks trained on the *exact same hyperparameter config* but different random seeds. We have ~200 such pairs per task.

- **Within-config variance** (stochastic): RDM correlation between repeat pairs.
- **Between-config variance** (HP-driven): RDM correlation between randomly sampled different-config pairs.

Both are distributions of pairwise correlations. The gap between them tells us how much of the total variance in RDMs is due to HPs vs. training stochasticity.

**Output:** paired distribution plot (within-config vs. between-config), per task. Summary: variance ratio.

### 1.3 Category structure
For each network in each classification task, compute the correlation between the network's RDM and a block-diagonal "category model" RDM where same-category stimuli have dissimilarity 0 and different-category stimuli have dissimilarity 1.

Task-specific models:
- mnist_dual / mnist_10way / fashion_10way: block by digit/class identity (100×100)
- mnist_rnn: digit block model, expanded to 1400×1400 temporal format — entry for pair [(stim_i, t_a), (stim_j, t_b)] = 0 if digit_i == digit_j, else 1. Phase/timestep is invisible to the model; it tests whether stimulus identity is encoded consistently across all timesteps.
- spirals: block by arm identity (100×100)
- parity: by Hamming weight (graded model: |Hamming(a) − Hamming(b)| / 8)
- cartpole / fourrooms: geometric models (angle gradient, distance-to-goal gradient)
- adding: see dedicated sub-analysis below.

This tells us: do representations organise stimuli the way the task structure demands? Does category structure increase with performance? Does it vary by HP?

**Output:** correlation with category model, plotted as a function of normalised performance (scatter per task). Separate values for successful / partial / near-chance networks.

#### Adding temporal category models [multi-step analysis]

The 600×600 adding temporal RDM encodes both temporal structure (phases) and stimulus-specific content. Three candidate model families, tested in sequence:

**Step 1 — Phase identity model**
Entry = 0 if both rows share the same phase (p == q), else 1, regardless of stimulus identity. Tests whether the dominant structure in the RDM is temporal (transitions between task phases) rather than stimulus-specific. Computed and correlated across all networks.

**Step 2 — Stimulus value models**
For each candidate value dimension (value1, value2, sum = value1 + value2):
- Entry for pair [(stim_i, phase_p), (stim_j, phase_q)] = |v_i − v_j| / v_max, normalised to [0, 1].
- The model treats stimulus identity as the only signal and ignores phase.
- Run on a sample of networks; inspect per-phase correlations — i.e. restrict to pairs where both rows are in the **same phase** and compute Spearman r separately per phase. This reveals which value dimension is encoded at each stage of the task.

**Step 3 — Phase × value interaction model (conditional on Step 2)**
If Step 2 reveals a clear pattern (e.g., early phases track value1, late phases track sum), build a composite model that assigns a different value dimension to each phase. The specific form is determined by the Step 2 findings. This model is only constructed if the per-phase signal is clear enough to support a principled design.

**Analysis flow:** compute Step 1 correlations first (gives phase-transition signal). Then Step 2 per-phase analysis (identifies what is encoded when). Then, based on findings, optionally design and test Step 3. A sample of ~50 successful networks is sufficient for Steps 1–2; Step 3 runs on the full set.

### 1.4 Layer comparison (depth=2 networks)
For networks with depth=2: compute noise ceiling and category structure separately for layer_0 and layer_1.

- Are layer_1 RDMs more reliable (higher noise ceiling)?
- Do they show stronger category structure?
- Within the same network: how correlated are layer_0 and layer_1 RDMs?

**Output:** paired bar plots (layer_0 vs layer_1) for noise ceiling and category-model correlation, per task. Scatter of within-network RDM correlation (layer_0 vs layer_1).

### 1.5 Effective dimensionality
For each network, compute the participation ratio of the activation covariance matrix:
```
PR = (Σ λ_i)² / Σ λ_i²
```
where λ_i are the eigenvalues of the N_stimuli × N_stimuli covariance of activations. PR = 1 means one-dimensional; PR = N_stimuli means uniform.

**Implementation note:** compute the covariance in *stimulus space* (shape N_stimuli × N_stimuli), not feature space. This keeps the matrix size manageable across all tasks (max 200×200 for mnist_dual) and is consistent with the RDM being defined over stimuli. For tasks with hidden_size > N_stimuli (possible with hidden_size up to 256 and e.g. fourrooms at 61 stimuli), computing in feature space would give spuriously high rank; stimulus space avoids this.

How does effective dimensionality vary across tasks? Across performance levels? Does it correlate with noise ceiling (lower dimensionality → more reliable)? Does hidden_size actually affect how many dimensions are used, or do networks collapse regardless?

**Output:** dimensionality distributions per task (violin plots), scatter vs. performance, scatter vs. hidden_size.

### 1.6 Cross-task RSA (MNIST family)
For tasks that share stimuli — mnist_dual, mnist_10way, mnist_rnn — compute cross-task RDM correlations: take a network from task A and correlate its RDM with a network from task B.

- Within-task correlations (from 1.1) serve as the positive reference.
- Cross-task correlations: do mnist_dual networks and mnist_10way networks produce similar RDMs, even though they were trained on different objectives?
- Do cross-task correlations exceed what you'd expect by chance (permutation test)?

This gives intuition about false positive risk: how likely is it that two networks trained on related-but-different tasks produce spuriously similar RDMs?

**Output:** correlation matrix (task × task) for the MNIST family, compared against within-task noise ceiling.

---

## Finding #2 — How hyperparameters influence representations

**Core question:** Which HPs and latent HP variables shape the geometry of learned representations?

All analyses below use successful networks only and the last-hidden-layer RDM.

**Per-network RDM summary statistics** (computed once, used throughout):
- `reliability`: correlation with group mean RDM (from 1.1)
- `category_corr`: correlation with category model RDM (from 1.3)
- `dimensionality`: participation ratio (from 1.5)
- `mean_dissimilarity`: mean of upper triangle of the RDM

### 2.1 Direct HP effects
For each HP, relate it to each RDM summary statistic:
- Continuous HPs (lr, l1, l2, hidden_size, batch_size): Spearman correlation.
- Categorical HPs (optimizer, activation, depth, init_scale, cell_type, n_rnn_layers): mean ± SE per level, one-way ANOVA F-statistic.

Simple, exhaustive, interpretable. Produces one correlation/effect-size number per (HP, RDM property) cell.

**Output:** HP × RDM-property heatmap of correlation / effect size, per paradigm.

### 2.2 Latent variable analysis
Three composite HP variables, each defined as a standardised linear combination:

| Composite | HPs that load on it | Sign |
|---|---|---|
| **Stability** | lr (low), batch_size (large), optimizer=adam, init_scale=0.1, l2_reg (high) | more = more stable learning |
| **Capacity** | hidden_size (large), depth=2 | more = larger model |
| **Regularization** | l1_reg + l2_reg | more = stronger weight penalty |

For each composite, compute a scalar score per network (continuous HPs z-scored, categoricals coded ±1), then correlate with each RDM summary statistic. This tests whether theoretically motivated latent variables predict representation geometry better than individual HPs.

**Output:** composite × RDM-property scatter plots with regression line, per task.

### 2.3 PCA on RDMs
Flatten each network's RDM upper triangle into a vector; stack into an N × D matrix; run PCA. Do networks cluster into distinct RDM types in PC space, or is it a continuum? Do HP values explain position along PCs?

This is the same approach that was tried in the predecessor study on a smaller dataset. With ~700 successful networks per task, the question is whether more data reveals the structure that was missed before.

**Output:** PC1 vs PC2 scatter coloured by each HP (one plot per HP), per task. Variance explained per PC.

### 2.4 CCA
CCA (Canonical Correlation Analysis) finds the linear combination of HPs and the linear combination of RDM properties that are maximally correlated — without having to specify in advance which HP drives which property.

Input: [stability score, capacity score, regularization score, plus key individual HPs] × [reliability, category_corr, dimensionality, mean_dissimilarity]. Output: the first canonical pair (HP blend ↔ RDM blend) and its correlation.

**Caveat:** CCA requires at least several hundred networks to be stable given the number of input features. Most tasks have ~700 successful primaries — sufficient. mnist_rnn is capped at ~160 primaries total (~120 successful), which is borderline; treat CCA results for that task as exploratory. If 2.1–2.3 already tell a clear story, CCA is optional.

### 2.5 Layer comparison for HP effects
For depth=2 networks: repeat 2.1 and 2.2 separately for layer_0 and layer_1. Are HP effects stronger or weaker at the second layer? Does the stability composite predict layer_1 reliability more than layer_0?

**Output:** paired version of the HP × RDM-property heatmap (layer_0 | layer_1), for tasks with enough depth=2 networks.

### 2.6 UMAP of networks by RDM similarity
Compute pairwise RDM-to-RDM distance (1 − Spearman correlation between flattened upper triangles) for a subsample of ~200 successful networks per task. Embed in 2D with UMAP.

- Colour by each HP: do HP variables explain the layout?
- Colour by RDM summary statistics: is the embedding structured by reliability, dimensionality?
- Are there discrete clusters, or a continuous manifold?

**Output:** UMAP scatter plots, one per HP/statistic, per task.

---

## Finding #3 — Representations over the course of learning

The analyses in this section ask how representational geometry changes during training — when it forms, how fast it changes, whether it survives past the performance peak, and whether different networks take similar routes through representational space. Together they characterise the coupling between learning dynamics and representational structure, which determines how much RSA conclusions depend on exactly when during training you capture the network.

The checkpoint infrastructure provides three complementary time axes: step checkpoints (log₄-spaced gradient updates, dense early in training), performance checkpoints (saved when a running performance threshold is first crossed), and a best/final pair (peak-performance and end-of-training weights). Different analyses use different axes depending on what question they answer.

The production analyses (scripts 31–37) were informed by a self-contained exploratory script (`30_explore_trajectories.py`) that applied RDM galleries, similarity curves, and 2D/3D trajectory plots to three representative tasks (mnist_10way, fourrooms, adding).

### 3.1 When does the representation crystallize?
`31_spearman_stability.py`, `32_crystallisation_time.py`

**Question:** Does representational geometry stabilise well before training ends, or does it keep changing all the way through? If it crystallises early, then RDMs are a stable property of the network's solution and capturing them at any late-training point is valid. If it keeps drifting, then the specific checkpoint chosen matters.

**What we did:** Two complementary approaches, both measuring similarity between the last-hidden-layer geometry at each late-training step checkpoint and the geometry at the very end of training. Script 31 summarises this per network as the *minimum* Spearman r across checkpoints in the last N% of training (four window sizes: 1%, 5%, 10%, 25%), giving a single stability scalar whose distribution reveals how much residual drift remains. Script 32 asks a sharper question: at what point in training does similarity to the final checkpoint first reach ≥ 0.99 and *stay there*, giving a crystallisation time as a fraction of total training, shown as an ECDF split by successful vs. failed networks.

- Scripts: `31_spearman_stability.py`, `32_crystallisation_time.py`
- Output: `figures/spearman_stability_{window}.pdf`, `tables/spearman_stability_{window}.csv`; `figures/crystallisation_time.pdf`, `tables/crystallisation_time.csv`

### 3.2 Critical period — rate of representational change
`33_critical_period.py`, `34_change_vs_performance.py`

**Question:** Is representational change concentrated in an early critical period of high plasticity, or is it distributed throughout training? A critical period would imply the network lays down its geometric structure in the first fraction of training updates and then largely refines the readout; distributed change would imply geometry and performance are continuously coupled.

**What we did:** For every consecutive pair of checkpoints, we compute how much the geometry changed relative to the length of the interval — representational change per unit training time (script 33, time axis) or per unit performance gain (script 34, performance axis). Dividing by interval length is necessary to make change rates comparable across the uneven spacing of the checkpoint schedule: without normalisation, a large change over a long interval looks the same as a small change over a short one, obscuring where change is actually concentrated. The time-axis analysis uses all checkpoint types for which a step number is recoverable; the performance-axis analysis uses only performance checkpoints, which are available for all tasks including RL and are directly aligned to task difficulty. Results are grouped by paradigm (supervised / RNN / RL) since learning dynamics differ qualitatively across them.

- Scripts: `33_critical_period.py`, `34_change_vs_performance.py`
- Output: `figures/critical_period.pdf`, `tables/critical_period.csv`; `figures/change_vs_performance.pdf`, `tables/change_vs_performance.csv`

### 3.3 Overfitting — does the representation degrade?
`35_overfitting.py`

**Question:** Many networks continue training past their performance peak and overfit. When performance degrades after best, does the representational geometry degrade with it, or is the geometric structure more robust than the readout weights? The answer bears on how much the specific stopping point matters for RSA: if geometry survives overfitting, best-checkpoint and final-checkpoint RDMs are interchangeable; if it degrades, they are not.

**What we did:** For each successful network where best and final are distinct training points, we measure the performance drop (normalised best minus final performance) and the representational change (1 − Spearman r between best and final activation matrices), then scatter the two against each other per task. A near-zero slope would indicate geometric robustness; a positive slope would indicate the representation tracks performance loss.

- Script: `35_overfitting.py`
- Output: `figures/overfitting.pdf`, `tables/overfitting.csv`

### 3.4 Trajectory mapping in representational space
`36_trajectory_mds.py`

**Question:** Do different networks follow similar paths through representational space as they learn, or do they take qualitatively different routes to similar endpoints? Do successful networks converge toward a shared representational region, or remain dispersed? Convergence and parallel trajectories would suggest the task structure constrains the representational solution; divergence would point to a larger variety of valid solutions.

**What we did:** For each task, we jointly embed all step-checkpoint RDM vectors from up to ~300–500 successful networks in 2D via MDS on Spearman dissimilarity. Each network's checkpoints are connected in order, forming a trajectory, with a log-spaced colour gradient from light green (start) to black (end of training) so early and late training are visually distinct even when trajectories overlap. Using step checkpoints rather than performance checkpoints gives a denser and more evenly distributed time axis, and avoids the ambiguity of not knowing exactly when performance thresholds were crossed relative to each other. The embedding is computed jointly across all networks so the global geometry of trajectories — convergence, divergence, clustering — is directly visible.

- Script: `36_trajectory_mds.py`
- Output: `figures/trajectory_mds_{task}.pdf` (one per task, 9 total)

### 3.5 RDM gallery through learning
`37_rdm_gallery_thru_learning.py`

**Question:** What does representational geometry actually look like at each stage of training, and how does it differ between successful and failed networks? The quantitative analyses in 3.1–3.4 summarise change in single numbers; the gallery provides the visual ground-truth that anchors those summaries and lets us check that the measures are capturing something meaningful.

**What we did:** For each task, we display the RDM at every step checkpoint for 20 networks: 15 successful networks sampled to span the full performance range, and 5 failed networks chosen as the best-performing failures (closest to the success threshold, most informative for comparison). Best and final checkpoint RDMs are appended as additional columns where available, with the actual training step from `metadata.json` labelled on each cell. Rows run best-to-worst performance.

- Script: `37_rdm_gallery_thru_learning.py`
- Output: `figures/rdm_gallery_{task}.pdf` (one per task, 9 total)

---

## Finding #4 — Representations and performance prediction

**Core question:** Can early representations predict final performance, and do they add information over the loss curve alone?

### 4.1 Early RDM → predict final success
At each performance checkpoint, compute per-network RDM properties (category structure, noise ceiling contribution, dimensionality). Use these as predictors of whether the network eventually reaches the success threshold (binary outcome). Train a simple logistic regression or compute AUC at each checkpoint level.

The most informative version: at perf=0.1 (barely off the ground), which RDM property best predicts eventual success at perf=0.9? This has a direct practical application — identifying training runs that will fail before they waste compute.

**Output:** AUC vs. checkpoint level, per RDM property, per task. Table: best predictor per task and the checkpoint at which prediction becomes reliable (AUC > 0.8).

### 4.2 Representation vs. loss as early predictor
Using epoch checkpoints (0.25, 1, 4, 16, 64 epochs), we have both RDM properties *and* train/val loss from `history.json` at each timepoint. Compare prediction quality: does RDM category structure at epoch 4 predict final performance better or worse than val loss at epoch 4?

The hypothesis: for tasks with tricky loss landscapes (parity, adding), val loss can be low without the representations being well-organised, making it a misleading early signal. RDM properties might give earlier, cleaner separation. If that's true, it's a practically meaningful finding beyond this specific project.

**Output:** scatter plots (early predictor vs. final performance), with Spearman correlation, for each epoch checkpoint and each predictor type (loss vs. RDM properties). Side-by-side comparison per task.

### 4.3 Representations of failed networks — what went wrong? [Very optional]
Near-chance networks have sparse performance checkpoints (maybe only perf=0.025 or perf=0.05). Two sub-questions:

**a. Did they ever have good representations?** Compare early-checkpoint RDMs of near-chance networks vs. partial networks at the same performance level. If they look the same early but diverge later, failure is a late-stage problem (e.g., premature convergence, representation collapse). If they look different from the start, failure mode is encoded in the initial representational trajectory.

**b. Do "almost worked" networks (partial) look like successful networks?** Partial networks reached some performance but not the success threshold. Are their representations structurally similar to successful ones (suggesting they are "close" in representational space and might have succeeded with more training or slightly different HPs), or qualitatively different?

Distinguishing failure modes requires looking at loss curves from `history.json`: did the network learn slowly throughout, or learn normally and then plateau or regress?

**Output:** trajectory maps (from 3.4) with failed/partial/successful networks colour-coded. Supplementary, not main paper.

---

## Optional — Deeper and larger networks

**Rationale:** The main dataset uses shallow, small networks (1–2 hidden layers, hidden_size 16–256). The findings about HP effects and representational reliability might be specific to this regime, or they might reflect general principles of gradient-based learning. A small-scale extension with qualitatively different architectures tests generalisability.

For each extension, run a targeted subset of Finding #1 and #2 analyses (noise ceiling, category structure, stability composite → reliability) and explicitly compare results to the corresponding shallow-network task.

### O.1 CNNs on CIFAR-10 (~100 networks)
A 4-layer CNN (2 conv + 2 dense), trained on CIFAR-10 (10 classes, 32×32 RGB). The HP space covers the same axes as supervised tasks: optimizer, LR, L2, init_scale, plus depth (number of conv layers) and optionally dropout.

The stimulus set for RDMs: a fixed sample of test images (10 classes × 10 exemplars = 100 stimuli), analogous to mnist_10way. No GP needed at this scale — quasi-random sampling of HP space is sufficient.

Key question: does the stability composite (low LR, adam, small init) predict RDM reliability in CNNs the way it does in MLPs? Do CNN representations show a higher noise ceiling (more constrained by architecture) or lower (more sensitive to initialisation)?

### O.2 Q*bert (~20 networks)
The user has an existing Q*bert repository with working agents. Rather than running a new HP search, run 20 networks from known-working configurations while capturing RDMs.

**Open challenges to resolve before implementation:**
- **Stimuli for the RDM:** depends on the state representation used by the Q*bert implementation. If pixel-based (CNN input): use a fixed set of canonical game frames (one per meaningful board state, levels 1–4 only). If feature-based: use a grid over the key state dimensions analogous to CartPole's angle × angular velocity grid. This needs to be determined from the existing implementation.
- **Scope:** focus on the first 3–4 levels; few networks will go beyond that, so alignment of the stimulus set to those levels is important.
- **No GP:** HP space is small, and the number of networks (20) is insufficient to fit a GP. Manually select configurations covering a few values of LR, optimizer, and architecture.

Key question: do the representational geometry patterns from CartPole and FourRooms (RL paradigm) replicate at larger scale and harder task?

### O.3 Open-source LLMs [Speculative — future work]
Pre-trained open-source LLMs (e.g., Llama, Phi, Mistral) have accessible weights, and activations can be extracted via PyTorch forward hooks.

**Framing:** different model families represent different architectural HP choices — depth, width, attention structure, activation functions, training recipe. Comparing across families is therefore directly analogous to our HP variation analysis, just at a much larger scale and with a different kind of HP space. The central question becomes: **as tasks and models scale up, does RSA become more reliable?** If larger, more capable models produce higher noise ceilings and more consistent representational geometry across runs/variants, that would be a strong argument for the validity of RSA in the large-model regime — and a direct extension of the small-network findings in this paper.

**Key challenge:** defining a stimulus set. For LLMs, stimuli are text inputs rather than images or states. Options: (a) semantically structured word lists or short prompts with controlled categorical structure (e.g., animal names, tools, abstract concepts — same categories used in human neuroscience RSA studies, enabling potential brain-model comparison); (b) sentence pairs with systematically varied meaning. The choice is not obvious and determines what the RDMs represent. This needs a clear conceptual decision before implementation.

**Note on prior literature:** RSA applied to LLMs and cross-model representational comparisons exist in the NLP literature. The specific angle here — HP variation across model families as a predictor of RSA reliability, and the scaling question — may be novel, but this needs a literature check before committing to the work.

**Recommendation:** conceptually directly relevant, but the stimulus design problem and the amount of work required (literature review, stimulus design, implementation of activation extraction across multiple model families) make this a post-paper extension unless it can be scoped very tightly.

---

## Outputs summary

| Output | File(s) |
|---|---|
| Noise ceiling distributions | `figures/f1_noise_ceiling.pdf` |
| Stochastic vs. HP variance | `figures/f1_variance_decomposition.pdf` |
| Category structure vs. performance | `figures/f1_category_structure.pdf` |
| Layer comparison (depth=2) | `figures/f1_layer_comparison.pdf` |
| Effective dimensionality | `figures/f1_dimensionality.pdf` |
| MNIST cross-task RSA matrix | `figures/f1_crosstask_rsa.pdf` |
| HP × RDM property heatmap | `figures/f2_hp_effects.pdf` |
| Latent variable scatter plots | `figures/f2_latent_vars.pdf` |
| PCA on RDMs | `figures/f2_rdm_pca.pdf` |
| UMAP of networks | `figures/f2_umap.pdf` |
| Layer comparison for HP effects | `figures/f2_layer_hp_effects.pdf` |
| Per-network RDM stats table | `tables/rdm_stats.csv` |
| Spearman stability (late-window) | `figures/spearman_stability_{window}.pdf`, `tables/spearman_stability_{window}.csv` (script 31) |
| Crystallisation time ECDF | `figures/crystallisation_time.pdf`, `tables/crystallisation_time.csv` (script 32) |
| Critical period (change rate vs. time) | `figures/critical_period.pdf`, `tables/critical_period.csv` (script 33) |
| Change rate vs. performance | `figures/change_vs_performance.pdf`, `tables/change_vs_performance.csv` (script 34) |
| Overfitting RDM stability | `figures/overfitting.pdf`, `tables/overfitting.csv` (script 35) |
| Trajectory MDS (per task) | `figures/trajectory_mds_{task}.pdf` — 9 files (script 36) |
| RDM gallery through learning | `figures/rdm_gallery_{task}.pdf` — 9 files (script 37) |
| Early prediction AUC curves | `figures/f4_early_prediction.pdf` |
| RDM vs. loss prediction comparison | `figures/f4_rdm_vs_loss.pdf` |
| Failed network trajectory analysis | `figures/f4_failed_networks.pdf` [very optional] |
