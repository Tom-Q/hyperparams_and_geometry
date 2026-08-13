# LLM Representational Geometry Study Plan

## Overview

Extension O.3 from the main analysis plan. The RSA framework applied to open-source LLMs using a
structured prompt set as stimuli. Goal: verify whether findings from small trained-from-scratch
networks generalise to the large pretrained model regime.

Three components:

- **Final-checkpoint cross-model analysis:** RSA validity, category structure, dimensionality,
  and base/instruct comparisons across a family-diverse set of models.
- **Pythia training dynamics (6 sizes × 16 checkpoints):** Direct analogue of Finding 3.
  Controls for family, data, and recipe while varying model size.
- **OLMo training dynamics (1 size × 16 checkpoints):** Replication of the training dynamics
  analyses in a second architecture, testing whether crystallisation and critical-period phenomena
  are architecture-general.

---

## Stimuli

**File:** `docs/stimuli_v5.1.json` — 256 stimuli, read-only. Do not modify.

**Design:** Two parallel halves, row order defines canonical RDM order.

- **s001–s128 (`mode=request`):** 8 themes × 4 tasks × 4 items — instruction/question stimuli.
  Meaningful for instruct-tuned models.
- **s129–s256 (`mode=passage`):** 8 genres × 4 subgenres × 4 items — declarative text passages
  written as pastiches of named real-world style anchors. Meaningful for base models.

Request half themes:

| Theme | Tasks |
|---|---|
| `stem` | math_algebra, physics, logic_syllogisms, programming_plain_english |
| `reading_comprehension` | fables, proverbs, protagonist, pronoun_reference |
| `translation` | french_to_german, spanish_to_italian, portuguese_to_dutch, danish_to_polish |
| `conflict_advice` | romantic_partner, neighbour, workplace, family |
| `poetry` | haiku, limerick, sonnet, free_verse |
| `health_advice` | nutrition, workout, sleep, stress |
| `political_argument` | immigration, taxation, criminal_justice, environment |
| `roleplay` | war_and_peace, pride_and_prejudice, star_wars, thousand_and_one_nights |

Passage half genres: fiction, journalism, stem_prose, interactive, correspondence, commercial,
legal_bureaucratic, instructional.

---

## Models

### Final-checkpoint set

**Pythia series — capacity axis (EleutherAI, base only)**

| Model | n_layers | d_model |
|---|---|---|
| pythia-70m | 6 | 512 |
| pythia-160m | 12 | 768 |
| pythia-410m | 24 | 1024 |
| pythia-1b | 16 | 2048 |
| pythia-1.4b | 24 | 2048 |
| pythia-2.8b | 32 | 2560 |

**Cross-family base + instruct pairs — architecture/recipe axis**

| Base | Instruct | n_layers | d_model |
|---|---|---|---|
| `meta-llama/Llama-3.2-1B` | `meta-llama/Llama-3.2-1B-Instruct` | 16 | 2048 |
| `Qwen/Qwen2.5-1.5B` | `Qwen/Qwen2.5-1.5B-Instruct` | 28 | 1536 |
| `HuggingFaceTB/SmolLM2-1.7B` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 24 | 2048 |
| `google/gemma-2-2b` | `google/gemma-2-2b-it` | 26 | 2304 |

**OLMo — second architecture with training checkpoints (AllenAI)**

| Model | Role | n_layers | d_model |
|---|---|---|---|
| `allenai/OLMo-1B-0724-hf` | Base, training dynamics | 16 | 2048 |
| `allenai/OLMo-2-0425-1B-SFT` | SFT (base → SFT on Tülu 3) | 16 | 2048 |
| `allenai/OLMo-2-0425-1B-Instruct` | Instruct (SFT + DPO + RLVR) | 16 | 2048 |

### Training dynamics checkpoints

**Pythia** — 16 log-spaced checkpoints across 143,000 steps (300B tokens):
`step1, step2, step4, step8, step16, step32, step64, step128, step256, step512,
step1000, step2000, step8000, step32000, step64000, step143000`

**OLMo** — 16 log-spaced checkpoints across 1,454,000 steps (~3T tokens):
`step1000-tokens2B, step2000-tokens4B, step3000-tokens6B, step4500-tokens9B,
step7000-tokens14B, step11000-tokens23B, step18000-tokens37B, step30000-tokens62B,
step49000-tokens102B, step79000-tokens165B, step128000-tokens268B, step209000-tokens438B,
step339000-tokens710B, step551000-tokens1155B, step895000-tokens1876B, step1454000-tokens3048B`

---

## Activation Extraction

**Status: complete.** All 122 model states extracted.

Output format per npz file:
- `last_{label}`, `mean_{label}` — (256, d_model) float32 — last-token and mean-pool at every
  layer (emb + L1 … L{n_layers})
- `token_counts` — (256,) int32 — tokens per stimulus for this tokenizer
- `nll` — (256,) float32 — mean per-token NLL
- `layer_labels`, `n_layers`, `d_model` — metadata

Row order matches the stimulus JSON: rows 0–127 = request half, rows 128–255 = passage half.

---

## RDM Construction (`build_llm_rdms.py`)

**Metric:** Pearson correlation distance (1 − r) as primary; cosine distance as secondary.
Equivalently: mean-centre activation matrix across stimuli, then compute cosine distances.

**Pooling:** Last-token as primary; mean-pool as secondary.

**Stimulus subsets:** For each model, compute three RDMs per layer:
- `full` — 256×256, all stimuli
- `request` — 128×128, rows 0–127
- `passage` — 128×128, rows 128–255

All three subsets are computed for every model regardless of base/instruct status. We do not pre-decide which half is "meaningful" for a given model type — that is an empirical question (see A.1, A.5). The difference between how a base model represents request stimuli vs. how an instruct model does is itself a finding.

**Layers:** All layers (emb + all transformer blocks).

**Visualisation:** For each model (at all checkpoints for dynamics models), save the `full` RDM
at the final layer as a PNG. Compile a PDF with one row per model:
- Pythia dynamics: 6 rows (one per size), 16 columns (checkpoints, early→late)
- OLMo dynamics: 1 row, 16 columns

**Output structure:**
```
output/production/llm/
  activations/          — npz files (complete)
  rdms/
    {slug}_last_full.npz        — all layers, full 256×256
    {slug}_last_request.npz     — all layers, request 128×128
    {slug}_last_passage.npz     — all layers, passage 128×128
  figures/
    rdm_gallery_pythia.pdf
    rdm_gallery_olmo.pdf
```

---

## Analysis

### Layer sets

Models have different depths (6–32 layers), so layers are referred to by percentile position
to allow cross-model comparison. For a model with n transformer blocks:

- **Percentile layer** p% → `round(p / 100 * n)`, clamped to [1, n]
- **Last 3 layers** → L{n}, L{n−1}, L{n−2}

Layer sets used per analysis:

| Analysis | Layers |
|---|---|
| A.1 cross-model agreement | 10th, 50th, 90th percentile + last 3 |
| A.2 category structure | 10th, 30th, 50th, 70th, 90th percentile + last |
| A.3 dimensionality | 10th, 30th, 50th, 70th, 90th percentile + last |
| A.4 depth profiles | All layers (emb + L1 … L{n}) |
| A.5 base/instruct comparison | Empirically chosen from A.1 + A.2 results |
| A.6 summary table | Empirically chosen from A.1 + A.2 results |
| A.7 cross-model matrix | Empirically chosen from A.1 + A.2 results |
| B/C dynamics | Empirically chosen from A.1 + A.4 results |

**Layer selection rule:** For analyses that require a single layer (A.5–A.7, B/C), we look at
both cross-model agreement (A.1) and category structure correlation (A.2/A.4) and choose the
layer that scores highest on both. If the two criteria disagree, we report results at each
candidate layer. This guards against systematic bias: cross-model agreement may favour early
layers if models share surface-form biases; category structure may favour layers that match our
assumed structure even if models are noisy there.

---

### A. Final-checkpoint cross-model

#### A.1 Cross-model agreement
Leave-one-out: for each model, correlate its RDM with the mean RDM of all other models.
Average over models. Computed at: 10th, 50th, 90th percentile layers and last 3 layers
(L{n}, L{n−1}, L{n−2}). Computed separately for the request half, passage half, and full set.

This is a theory-neutral measure of inter-model consistency — it does not assume anything about
what the representational structure should be. Called "cross-model agreement" (not noise ceiling)
because it measures model similarity, not measurement reliability. A high value at a layer means
models systematically agree on pairwise stimulus distances there; low means models diverge.

#### A.2 Category structure
Correlate each model's RDM with ideal block-diagonal RDMs encoding the stimulus hierarchy
(theme-level, task-level). Use Kendall τ. Layers: 10th, 30th, 50th, 70th, 90th percentile + last.
Compute for request half (theme/task structure) and passage half (genre/subgenre structure)
separately. This is an assumption-laden measure — it rewards models that organise stimuli the
way we expect; interpret alongside A.1 which is assumption-free.

For crossed themes (translation: content × language; poetry: topic × form; politics: issue ×
stance), correlate with competing ideal RDMs to assess whether the model organises by one
dimension or the other.

#### A.3 Effective dimensionality
Participation ratio of the stimulus-space covariance matrix. Layers: 10th, 30th, 50th, 70th,
90th percentile + last. Compute for request and passage halves separately.

#### A.4 Layer depth profiles
Plot category structure (τ with theme/task RDM) and dimensionality (PR) as a function of
normalised layer depth (0–1) for each model. Group by family. Compare request vs passage halves.

#### A.5 Base vs instruct comparison
Layer: empirically chosen from A.1 + A.2 results.

For each base/instruct pair (Llama, Qwen, SmolLM2, Gemma, OLMo), directly compare:
- RDM similarity between base and instruct (how much does instruction tuning change geometry?).
  Compute for both stimulus halves separately — the degree of change may differ by half.
- Category structure: compare theme/task organisation on request half and genre/subgenre
  organisation on passage half, for both model types. We make no prior assumptions about which
  model type performs better on which half; the cross (base on request, instruct on passage) is
  as interesting as the match (base on passage, instruct on request).

#### A.6 Summary statistics table
Per model: cross_model_agreement (A.1), category_corr_theme, category_corr_task (A.2),
dimensionality (A.3), mean_dissimilarity. Report for request and passage halves separately.
Layer: empirically chosen from A.1 + A.2. Group by family.

#### A.7 Cross-model RSA matrix
Pairwise RDM correlations between all 17 models. Layer: empirically chosen from A.1 + A.2.
Compute separately for request half, passage half, and full set — comparing the three matrices
shows how much stimulus type affects model similarity. MDS/clustering to visualise model
similarity. Colour by family and base/instruct status.

---

### B. Pythia training dynamics

All analyses use last-token activations at all 16 checkpoints × 6 sizes.
Layer: empirically chosen from A.1 (cross-model agreement) and A.4 (depth profiles) at the
final Pythia checkpoints, looking at both criteria before deciding to guard against systematic
bias. Passage half as primary stimulus set (base models); request half also computed.

#### B.1 Crystallisation
Spearman correlation between each checkpoint's RDM and the final checkpoint RDM, per size.
Plot on log-scaled step axis. Does geometry stabilise earlier (as fraction of training) for
larger models?

#### B.2 Critical period — rate of change
RDM dissimilarity between consecutive checkpoints divided by log interval length. Plot per size.
Is there an early burst of rapid change?

#### B.3 Trajectory MDS
MDS on pairwise RDM dissimilarities across checkpoints. Joint embedding across all 6 sizes,
coloured by training step. Do sizes follow parallel or diverging trajectories?

#### B.4 RDM gallery
Grid of RDM images: 6 rows (sizes) × 16 columns (checkpoints). Compiled into PDF.

---

### C. OLMo training dynamics

Same analyses as B.1–B.4 for OLMo-1B-0724. Single row in the RDM gallery PDF.

#### C.5 Cross-architecture comparison
Overlay OLMo and Pythia-1B crystallisation curves and change-rate curves on the same plot.
Both are 1B-class models; differences reflect architecture and data rather than capacity.

---

## Scripts

```
scripts/
  extract_llm_activations.py    — complete
  run_llm_extraction.sh         — complete
  build_llm_rdms.py             — next
  analyze_llm_final.py          — analyses A.1–A.7
  analyze_llm_dynamics.py       — analyses B.1–B.4 and C.1–C.5
```
