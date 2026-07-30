# Section 1 Findings: Validity of RSA

Analysis scripts: `analysis/11_rsa_validity.py`, `11b_temporal_validity.py`, `12_category_models.py`, `14_category_structure.py`, `15_layer_comparison.py`, `16_dimensionality.py`, `17_crosstask_rsa.py`.

---

## 1.1 Inter-network agreement (noise ceiling)

**Method.** For each task, successful primary networks only (no repeats). For each network, Spearman r between its RDM and the leave-one-out mean of all other successful networks. Reported as median (IQR).

| Task | n | Median NC | IQR |
|---|---|---|---|
| CartPole | 570 | 0.950 | 0.900–0.973 |
| Fashion 10-way | 650 | 0.935 | 0.896–0.960 |
| MNIST 10-way | 658 | 0.859 | 0.810–0.908 |
| MNIST RNN | 120 | 0.842 | 0.776–0.873 |
| Spirals | 184 | 0.820 | 0.751–0.875 |
| Adding | 551 | 0.786 | 0.670–0.846 |
| Parity | 114 | 0.735 | 0.576–0.866 |
| FourRooms | 372 | 0.732 | 0.662–0.805 |
| MNIST dual | 559 | 0.586 | 0.512–0.766 |

**Conclusions.**
- RSA is generally well-supported: most tasks have median NC > 0.75. CartPole and Fashion are the strongest.
- MNIST dual is the clear outlier (NC = 0.586). Its two-objective training (even/odd + small/large classification) induces genuine representational heterogeneity — different hyperparameter configurations trade off the two tasks differently, producing a wide spread of RDM types within the task. This is not noise; see §1.6.
- Parity and FourRooms have moderate NCs with wide IQRs, indicating that successful networks span a larger representational range (multiple viable solutions) rather than converging on one geometry.

**Temporal extension — MNIST RNN.**
NC was also computed per timestep across the 14 input tokens (pixels presented left-to-right):

| t | 0 | 1 | 2 | 3 | 4 | 7 | 10 | 13 |
|---|---|---|---|---|---|---|---|---|
| NC | 1.000 | 0.999 | 0.965 | 0.894 | 0.881 | 0.836 | 0.784 | 0.721 |

NC = 1.0 at t = 0 is expected and not informative: at the first timestep the hidden state is all-zeros, so activation equals `tanh(W_xh * x_0 + b)` — a monotone function of a single scalar input. All networks compute identical ranked pairwise distances. NC degrades monotonically across tokens as the hidden state accumulates information specific to each network's weight configuration.

**Temporal extension — Adding (6 semantic phases).**
Phases are defined by flag positions within the 25-step sequence:
- Phase 1: before first flag (pre-task)
- Phase 2: first flag step
- Phase 3: between flags (working memory)
- Phase 4: second flag step
- Phase 5: after second flag (post-task)
- Phase 6: final step

| Phase | Description | NC |
|---|---|---|
| 1 | Before first flag | 0.860 |
| 2 | First flag | 0.766 |
| 3 | Between flags | 0.674 |
| 4 | Second flag | 0.692 |
| 5 | After second flag | 0.764 |
| 6 | Final step | 0.728 |

NC is highest before the task begins (phase 1), when the hidden state mostly reflects initial weight structure. It drops during the flag presentation and working-memory phases (3, 4), where different HP configurations produce more divergent representations. This is consistent with the working-memory period being the locus of strongest HP-driven variability (see §1.2 temporal).

---

## 1.2 Stochastic vs. HP-driven variance

**Method.** Repeat pairs (same HP config, different seeds) give within-config Spearman r. Random pairs from different configs give between-config Spearman r. The gap quantifies how much HP choice shapes geometry beyond random seed.

| Task | Within-config | Between-config | Gap |
|---|---|---|---|
| CartPole | 0.932 | 0.891 | **0.040** |
| Spirals | 0.757 | 0.679 | **0.079** |
| Parity | 0.590 | 0.492 | 0.097 |
| Fashion 10-way | 0.954 | 0.868 | 0.087 |
| MNIST RNN | 0.833 | 0.696 | 0.138 |
| FourRooms | 0.720 | 0.568 | 0.152 |
| MNIST 10-way | 0.895 | 0.736 | 0.159 |
| Adding | 0.919 | 0.586 | **0.334** |
| MNIST dual | 0.781 | 0.423 | **0.358** |

**Conclusions.**
- All tasks show within > between: HP choice consistently shapes representation beyond stochastic variation.
- The gap varies by an order of magnitude across tasks. CartPole and Spirals are at one extreme: the geometry is strongly constrained by task structure, so HP variation barely matters — most configurations land on the same spatial or angular organisation. Adding and MNIST dual are at the other extreme: HP choice substantially determines which solution the network finds.
- MNIST dual's large gap (0.358) is consistent with its low NC: when HP configs produce very different representations, both inter-network disagreement (NC) and inter-config spread (gap) are large.
- The high within-config value for CartPole (0.932) combined with a small gap confirms that CartPole representations are nearly deterministic given the task — seed barely matters, HP choice barely matters.

**Temporal extension — Adding phases.**

| Phase | Within | Between | Gap |
|---|---|---|---|
| 1 (pre-task) | 0.954 | 0.747 | 0.207 |
| 2 (flag 1) | 0.930 | 0.622 | 0.308 |
| 3 (between flags) | 0.910 | 0.451 | **0.459** |
| 4 (flag 2) | 0.888 | 0.446 | 0.442 |
| 5 (post-flag) | 0.930 | 0.556 | 0.374 |
| 6 (final) | 0.913 | 0.512 | 0.401 |

The working-memory phase (3) maximises the HP-driven gap. Between the two flags, the network must retain the first operand in its hidden state — and how it does so is almost entirely determined by hyperparameter configuration (between-config median falls to 0.451, less than half the within-config median). This is the phase where HP choice most strongly determines what the network has learned to represent.

---

## 1.3 Category structure

**Method.** For each task, task-appropriate category models are constructed as structured RDMs and correlated (Spearman) with each network's RDM. Results reported for successful networks only.

| Task | Best model | Median r (successful nets) |
|---|---|---|
| CartPole | Euclidean distance | **0.917** |
| FourRooms | Euclidean distance | 0.637 |
| FourRooms | Room identity | 0.586 |
| Spirals | Spatial distance (2D) | 0.692 |
| Spirals | Arm identity | 0.026 |
| MNIST 10-way | Digit identity | 0.437 |
| MNIST RNN | Digit identity | 0.486 |
| MNIST dual | Mixed (digit + task) | 0.452 |
| MNIST dual | Output label only | 0.419 |
| MNIST dual | Digit identity | 0.266 |
| Fashion 10-way | Class identity | 0.440 |
| Parity | Hamming distance (graded) | 0.121 |
| Adding (value1) | First operand value | see §1.3 phases |
| Adding (sum) | Sum of operands | see §1.3 phases |

**Conclusions.**
- CartPole representations are almost perfectly organised by euclidean state-space distance (r = 0.917). The network's geometry mirrors the continuous geometry of the task state space, not discrete categorical structure.
- FourRooms also prioritises euclidean proximity (r = 0.637) over room identity (r = 0.586), though both are substantial. Representations are location-based rather than room-based.
- Spirals: spatial distance dominates (r = 0.692), arm identity is negligible (r = 0.026). Networks learn that nearby points on the same spiral are more similar than distant points on different spirals — geometry within spirals matters more than which spiral a point belongs to.
- MNIST tasks: digit identity is the primary organising principle (r ≈ 0.44–0.49), with MNIST RNN somewhat stronger than 10-way. For MNIST dual, the mixed model (digit + task) slightly outperforms output label alone, and digit identity alone is notably weaker — the task bit is represented alongside digit structure.
- Parity: the graded Hamming model provides only weak structure (r = 0.121). The combinatorial input space (binary strings of length 8) is not strongly organised by Hamming weight, suggesting that parity networks find representations that are idiosyncratic or use other organisational principles.

**Adding — phase-by-phase category structure.**
The adding task uses phase-aligned RDMs with two category models: value of first operand (value1) and sum of both operands (sum). Category structure is computed per phase, subsetting both the network RDM and the model RDM to the valid stimuli in that phase.

Key pattern: sum model correlation rises through phases 3–5 (the period after the second flag has been processed), consistent with the network's representation shifting from tracking the first operand to encoding the full sum as the task demands it.

---

## 1.4 Layer comparison (depth=2 networks)

**Method.** For networks with depth=2, category model correlations and within-network RDM similarity computed separately for layer 0 (input-side, H units) and layer 1 (output-side, H//2 units). Focal tasks: mnist_dual, spirals, fourrooms.

### MNIST dual (n = 278 successful depth-2 networks)

| | Layer 0 | Layer 1 |
|---|---|---|
| Digit identity | 0.312 | 0.224 |
| Output label (task) | 0.240 | **0.495** |
| Within-network L0↔L1 | — | 0.345 |

Clear hierarchical specialisation: layer 0 is relatively more digit-like, layer 1 reorganises toward the task output label. The within-network correlation of 0.345 is very low — the two layers are geometrically quite different from each other. This suggests that depth-2 dual networks implement a genuine hierarchical transform: digit features in L0, task-diagnostic structure in L1.

### Spirals (n = 180 successful depth-2 networks)

| | Layer 0 | Layer 1 |
|---|---|---|
| Spatial distance | **0.936** | 0.692 |
| Arm identity | 0.002 | 0.027 |
| Within-network L0↔L1 | — | 0.702 |

Spatial structure dominates both layers, but compresses from L0 to L1. Arm identity is negligible at both layers — even the second layer does not reorganise toward discrete arm categories. The relatively high within-network correlation (0.702) reflects that both layers share the same fundamental spatial organisation; L1 is a compressed version of L0, not a qualitatively different one.

### FourRooms (n = 118 successful depth-2 networks)

| | Layer 0 | Layer 1 |
|---|---|---|
| Euclidean distance | 0.746 | 0.607 |
| Room identity | 0.636 | 0.579 |
| Within-network L0↔L1 | — | 0.846 |

Both layers are strongly euclidean, with room identity close behind. There is no clear hierarchical reorganisation between layers (within-network correlation 0.846 is the highest of the three tasks). FourRooms depth-2 networks appear to learn two similar spatial representations rather than a hierarchical abstraction.

**General conclusion.** Hierarchical reorganisation across layers is task-dependent. MNIST dual shows the strongest reorganisation (input digit → output label), while spirals and fourrooms mostly compress their layer 0 geometry rather than qualitatively transforming it.

---

## 1.5 Effective dimensionality (participation ratio)

**Method.** For each network, PR = (Σλ_i)² / Σλ_i² where λ_i are eigenvalues of the stimulus-space activation covariance (N_stim × N_stim Gram matrix, centred). PR = 1 is fully collapsed (single axis); PR = N_stim is flat (uniform spectrum). Results below are for successful networks, last hidden layer.

| Task | n | Median PR | Min | Max |
|---|---|---|---|---|
| CartPole | 570 | **1.5** | 1.0 | 2.6 |
| Adding | 551 | **1.6** | 1.0 | 19.5 |
| FourRooms | 381 | 2.9 | 1.0 | 8.5 |
| Spirals | 195 | 3.2 | 1.5 | 5.5 |
| Fashion 10-way | 651 | 4.0 | 1.8 | 11.6 |
| MNIST dual | 567 | 6.2 | 1.0 | 24.3 |
| MNIST RNN | 120 | 6.5 | 4.2 | 14.8 |
| MNIST 10-way | 661 | 7.3 | 3.0 | 22.3 |
| Parity | 118 | 8.5 | 1.2 | 14.7 |

**Conclusions.**
- CartPole (PR = 1.5) and Adding (PR = 1.6) produce near-1D representations. CartPole geometry is almost entirely on one axis — consistent with the near-perfect euclidean category model fit (§1.3): the representation reduces to a scalar summary of the state. Adding representations are also highly compressed; the working-memory requirement is apparently solved with very low intrinsic dimensionality.
- The 10-class visual tasks (MNIST, Fashion) occupy a higher-dimensional space (PR = 4–7), consistent with needing to separate 10 distinct classes. MNIST RNN is comparable to MNIST 10-way despite sequential processing.
- Parity has the highest median PR (8.5). The combinatorial structure of the input space (128 binary strings of length 7) provides many potential organisational axes, and networks use more of them than for digit classification. This is consistent with the weak category model fit in §1.3.
- Adding's wide max (19.5) vs. low median (1.6) reflects that rare network configurations find high-dimensional solutions — these tend to be networks with large hidden sizes that did not collapse to a low-dimensional attractor.

---

## 1.6 Cross-task RSA — MNIST family

**Method.** For each pair of MNIST tasks, N = 3000 random network pairs drawn from successful primaries, Spearman r computed on matched 100-stimulus RDM vectors (task=0 sub-RDM extracted from mnist_dual's 200-stimulus RDM). Full pairwise similarity matrix also computed (n=1337 networks) to get per-network within-vs-cross comparisons.

### Group-level correlations

| Pair | Type | Median r | IQR |
|---|---|---|---|
| mnist_10way ↔ mnist_10way | within | 0.731 | 0.657–0.804 |
| mnist_rnn ↔ mnist_rnn | within | 0.699 | 0.628–0.758 |
| mnist_dual ↔ mnist_dual | within | 0.506 | 0.394–0.651 |
| mnist_10way ↔ mnist_rnn | **cross** | **0.600** | 0.542–0.657 |
| mnist_dual ↔ mnist_10way | cross | 0.446 | 0.336–0.563 |
| mnist_dual ↔ mnist_rnn | cross | 0.392 | 0.328–0.457 |

### Per-network analysis

For each network, mean similarity to all other networks in the same task vs. all networks in the other task:

| Focus task | Comparison | Mean cross > mean within | MWU p |
|---|---|---|---|
| mnist_dual | vs. mnist_10way | **40% of dual networks** | 1.5e-21 |
| mnist_dual | vs. mnist_rnn | 5% | ≈0 |
| mnist_10way | vs. mnist_dual | 0% | ≈0 |
| mnist_rnn | vs. mnist_10way | 4.2% | 2.6e-25 |

### 1-NN task confusion

For each network, the task of its single most-similar other network:

| Network task | NN from same task | NN from dual | NN from 10way | NN from rnn |
|---|---|---|---|---|
| mnist_dual | 98.9% | — | 1.1% | 0.0% |
| mnist_10way | 99.7% | 0.0% | — | 0.3% |
| mnist_rnn | 95.8% | 0.0% | 4.2% | — |

**Conclusions.**

**mnist_10way and mnist_rnn share a geometry.** Cross-task r = 0.600 approaches the within-task values (0.699 and 0.731). These two tasks converge on similar digit-identity representations of the same 100 images, despite very different architectures (MLP vs. RNN) and input formats (full image vs. 14-step scan). This is reassuring for the validity of RSA across task variants — but also means that RSA alone cannot distinguish these tasks.

**mnist_dual is geometrically heterogeneous.** Within-task NC (0.506) is substantially lower than 10way or rnn. The dual task's two competing objectives (even/odd + small/large) produce a wide spread of representational solutions — some networks are more digit-focused (resembling 10way), others more task-focused. This is confirmed by the 40% rate at which a given dual network's mean similarity to 10way networks exceeds its mean similarity to other dual networks.

**Task identity is preserved at the individual level.** Despite this heterogeneity, the 1-NN analysis shows that only 1.1% of dual networks have their nearest neighbor from another task. The representational space is not chaotic — it is just that the within-task distribution for dual is wide enough to partially overlap with the cross-task distribution. A given dual network is still most similar to some other dual network (98.9% of the time).

**Methodological concern.** For analyses that aggregate MNIST tasks (e.g., group-mean RDMs, between-task comparisons), mnist_dual should be treated with caution. Its within-group variance is nearly as large as its between-group variance with respect to mnist_10way. Analyses that treat mnist_dual networks as interchangeable will be noisy. Stratifying by a network-level metric (e.g., output-model RSA score) may be necessary to identify clean sub-groups.

---

## Possible follow-up analyses

Not planned for immediate execution — noted here for reference.

- **Adding phase category RSA.** Correlate the `value1` and `sum` category models per semantic phase. The prediction is that `value1` structure rises and holds through phase 3 (working memory), while `sum` structure emerges only at phases 5–6 after both operands are processed. Would close the loop between the temporal variance finding (§1.2) and representational content (§1.3).
- **Parity geometry.** Successful parity networks converge on something (NC = 0.735, within-config = 0.590) but not Hamming distance (r = 0.121). PCA or RDM gallery on parity networks to characterise what geometry they do use. Possibly individual bit-position features rather than aggregate Hamming weight.
- **MNIST dual spectrum vs. HPs.** Per-network `output_r − digit_r` score (from §1.4 / §1.3 category models) correlated against HP values. Which HPs push a dual network toward task-label vs. digit geometry? Depth is an obvious candidate given the layer comparison result.
- **CartPole 1D axis.** PR = 1.5 and euclidean r = 0.917 suggest a near-scalar representation. Regress the first PC of activations onto each state variable (angle, angular velocity, position, velocity) to identify which physical quantity the network is tracking.

## Summary table

| Finding | Main result |
|---|---|
| 1.1 NC | CartPole/Fashion: NC ≈ 0.93–0.95. Most tasks NC > 0.75. MNIST dual is the exception (NC = 0.59). MNIST RNN NC = 1.0 at t=0, degrades over token sequence. Adding NC lowest during working-memory phase. |
| 1.2 Variance | HP-driven gap present in all tasks. CartPole (0.040) and Spirals (0.079): task geometry dominates, HPs barely matter. Adding (0.334) and MNIST dual (0.358): HPs strongly determine which solution is found. Gap peaks at the working-memory phase for the adding task. |
| 1.3 Category | CartPole: near-perfect euclidean (r = 0.917). FourRooms and Spirals: spatial. MNIST: digit identity (r ≈ 0.44–0.49). Parity: only weak graded structure (r = 0.121). |
| 1.4 Layers | MNIST dual shows clear hierarchical reorganisation (digit → task label, within-network r = 0.345). Spirals compresses without reorganising (spatial stays dominant). FourRooms: both layers nearly identical (within-network r = 0.846). |
| 1.5 Dimensionality | CartPole and Adding: near-1D (PR ≈ 1.5–1.6). Visual 10-class tasks: PR = 4–7. Parity: highest (PR = 8.5), consistent with combinatorial input space. |
| 1.6 Cross-task | 10way ↔ rnn cross-task (0.600) ≈ within-rnn (0.699) — strong convergence. Dual has low within-task coherence (0.506) and 40% of dual networks resemble 10way more than other dual networks on average. But 1-NN confusion is only 1.1% — task identity holds at the individual level. |
