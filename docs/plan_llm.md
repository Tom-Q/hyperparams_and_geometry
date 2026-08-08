# LLM Representational Geometry Study Plan

## Overview

Extension O.3 from the main analysis plan. The same RSA framework applied to open-source LLMs,
using a structured prompt set as stimuli. The goal is to verify whether the findings from small
trained-from-scratch networks generalise to the large pretrained model regime — not to produce a
self-contained LLM paper.

The study has two components with different questions:

- **Final-checkpoint analysis (~11 models):** Basic RSA validity checks and descriptive summary
  statistics across a set of architecturally and family-diverse models. Low n limits this to
  descriptive work — no statistical tests of HP effects.
- **Pythia training dynamics (6 sizes × ~15 checkpoints each):** The main quantitative
  contribution. Pythia's fixed training setup and published checkpoints enable a direct analogue
  of Finding 3 (crystallisation, critical period, trajectory mapping), controlling for family,
  data, and training recipe while varying only model size.

---

## Stimuli

**File:** `data/stimuli_v1.1.json` — 128 stimuli, read-only. Do not modify.

**Design:** 8 themes × 4 tasks × 4 items = 128 stimuli, hierarchically ordered. Row order in the
JSON defines canonical RDM order.

| Theme | Tasks |
|---|---|
| `stem` | math_algebra, physics, logic_syllogisms, programming_plain_english |
| `reading_comprehension` | fables, proverbs, protagonist, pronoun_reference |
| `translation` | french_to_german, spanish_to_italian, portuguese_to_dutch, danish_to_polish |
| `conflict_advice` | romantic_partner, neighbour, workplace, family |
| `haiku` | nature, city_life, emotions, mundane_objects |
| `health_advice` | nutrition, workout, sleep, stress |
| `political_argument` | left_wing, right_wing, democracy_human_rights, environment |
| `roleplay` | lotr, war_and_peace, mean_girls, star_wars |

The 128 stimuli function as a single task (analogous to a single task in the main study). Each
model is one data point.

**Format:** `free_form` (majority) and `multiple_choice` (fables, proverbs, protagonist,
pronoun_reference). Multiple-choice stimuli include the answer options in the prompt text; no
special handling needed.

---

## Models

### Final-checkpoint set (~11 models)

One model at a time. Run sequentially, not in parallel.

**Pythia series — primary within-family size axis (EleutherAI)**

| Model | Parameters | d_model | n_layers |
|---|---|---|---|
| pythia-70m | 70M | 512 | 6 |
| pythia-160m | 160M | 768 | 12 |
| pythia-410m | 410M | 1024 | 24 |
| pythia-1b | 1B | 2048 | 16 |
| pythia-1.4b | 1.4B | 2048 | 24 |
| pythia-2.8b | 2.8B | 2560 | 32 |

For the final-checkpoint analysis, use the final Pythia checkpoint (step 143000).

**Cross-family models**

| Model | Parameters | Notes |
|---|---|---|
| `meta-llama/Llama-3.2-1B` | 1B | Llama architecture |
| `Qwen/Qwen2.5-1.5B` | 1.5B | Multilingual |
| `HuggingFaceTB/SmolLM2-1.7B` | 1.7B | HF compact model |
| `google/gemma-2-2b` | 2B | Requires HF license (one click) |

**Optional outgroup (non-transformer)**
- `state-spaces/mamba-130m` — include if extraction is straightforward; treat as outgroup in
  descriptive comparisons, interpret layer indices carefully.

### Pythia training checkpoints

Pythia was trained for ~143,000 gradient steps on 300B tokens. 154 checkpoints are available on
HuggingFace. Select ~15, approximately log-spaced:

`step1, step2, step4, step8, step16, step32, step64, step128, step256, step512, step1000,
step2000, step8000, step32000, step64000, step143000`

Use the same 15 checkpoints for all 6 Pythia sizes. This yields ~90 model states total for the
training dynamics analyses.

---

## Activation Extraction

### Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    output_hidden_states=True,
)
```

`output_hidden_states=True` returns a tuple of `(batch, seq_len, d_model)` tensors — one per
transformer block plus the embedding layer.

### Representation: last token

For all decoder-only models (all primary models), use the **last-token representation** at each
layer: `hidden_state[layer][:, -1, :]`. This is the natural representation for autoregressive
models and is directly analogous to using the last hidden layer in the main study.

### Distance metric

Use **Pearson correlation distance** (1 − Pearson r between activation vectors across units) as
the primary RDM metric, consistent with the ANN representational geometry literature. This is
equivalent to cosine distance on mean-centred vectors and avoids artefacts from scale differences
across model families. Concretely: mean-centre each (128, d_model) activation matrix across
stimuli, then compute cosine distances.

### Layer selection and output format

For the final-checkpoint cross-model analysis, extract **four relative-depth layers** per model:

| Label | Relative depth |
|---|---|
| emb | 0 (embedding, no contextual processing) |
| L50 | 0.50 (middle) |
| L75 | 0.75 (upper-middle) |
| Lout | 1.00 (final transformer block) |

For Pythia training dynamics, extract **Lout only** (final block) at each of the 15 checkpoints.
This is sufficient for the crystallisation and trajectory analyses and keeps storage manageable.

**Output format** — one `.npz` per (model, checkpoint):

```
activations_{model_slug}_step{step}.npz
  hidden_states  : (n_selected_layers, 128, d_model)  float32
  layer_indices  : (n_selected_layers,)  int32
  model_id       : str
  step           : int
```

`model_slug` replaces `/` with `__`.

**Memory:** load one model, extract all layers for all 128 stimuli, save, then
`del model; torch.cuda.empty_cache()` before loading the next.

---

## Analysis

The three analyses below mirror Findings 1, 2 (partially), and 3 of the main study.

---

### A. Final-checkpoint: RSA validity and summary statistics

Applies to all ~11 final-checkpoint models, using Lout activations. Small n means this is
descriptive — no HP regression or latent variable analysis.

#### A.1 Noise ceiling (≈ Finding 1.1)

For each model, correlate its Lout RDM with the mean RDM of all other models (leave-one-out).
Distribution of these correlations = the noise ceiling.

With only ~11 models this is low-powered. Report mean ± SD and the full distribution.
Compare against the noise ceilings from the main study tasks as qualitative reference: are
large LLMs more or less consistent with each other than small trained networks are with each other?

#### A.2 Category structure (≈ Finding 1.3)

The main analysis correlates each network's RDM with a block-diagonal "category model" RDM that
encodes the task structure. The LLM analogue uses the three levels of the stimulus hierarchy:

- **Theme RDM:** dissimilarity = 0 within theme (16 stimuli), 1 across — 8 blocks of 16
- **Task RDM:** dissimilarity = 0 within task (4 stimuli), 1 across — 32 blocks of 4
- **Item RDM:** all off-diagonal = 1 (items are definitionally distinct; not expected to be informative)

For each model, compute Kendall τ between the Lout RDM and each ideal RDM. This directly answers:
do LLMs organise stimuli according to the design hierarchy?

Compute this for all four extracted layers (emb, L50, L75, Lout) to see whether category structure
builds up through the network — a direct analogue of Finding 1.4.

#### A.3 Effective dimensionality (≈ Finding 1.5)

For each model, compute the participation ratio of the (128 × 128) stimulus-space covariance
matrix of Lout activations (same formula as the main study, computed in stimulus space):

```
PR = (Σ λ_i)² / Σ λ_i²
```

Does PR vary with model size? Does it correlate with noise ceiling?

#### A.4 Summary statistics table

Compute the four per-network RDM summary statistics from the main plan for each model:

- `reliability`: correlation with group mean RDM (from A.1)
- `category_corr_theme`, `category_corr_task`: τ with ideal RDMs (from A.2)
- `dimensionality`: participation ratio (from A.3)
- `mean_dissimilarity`: mean of RDM upper triangle

Report in a table grouped by family (Pythia sizes together, cross-family models together).
Do not fit a regression or run a significance test — n is too small.

---

### B. Pythia training dynamics (≈ Finding 3)

Pythia provides the controlled within-family variation needed for the training dynamics analyses.
All analyses in this section use Lout activations at the 15 selected checkpoints.

#### B.1 Crystallisation (≈ Finding 3.1)

For each Pythia size, compute the Spearman correlation between the RDM at each checkpoint and the
RDM at the final checkpoint (step 143000). Plot correlation as a function of training step
(log-scaled). When does geometry stabilise (reach ≥ 0.99 and stay there)?

This directly mirrors script 31/32. The question: do larger Pythia models crystallise earlier or
later in training (as a fraction of total steps)?

#### B.2 Critical period — rate of representational change (≈ Finding 3.2)

For each consecutive checkpoint pair, compute the RDM dissimilarity (1 − Spearman r) divided by
the log interval length (in steps). This gives the rate of representational change per unit log
training time, normalised to make early and late intervals comparable.

Plot change rate vs. training step for each Pythia size. Is there an early critical period of
rapid change followed by stabilisation? Does this pattern change with model size?

#### B.3 Trajectory mapping (≈ Finding 3.4)

Flatten each checkpoint RDM to its upper triangle. For each Pythia size, MDS on the dissimilarity
matrix between all 15 checkpoint RDM vectors, coloured by training step (light = early, dark =
late). Joint embedding across all 6 sizes shows whether different sizes follow parallel trajectories
or diverge early.

#### B.4 RDM gallery through training (≈ Finding 3.5)

For each Pythia size (6 panels), display the 128×128 RDM at each of the 15 training checkpoints,
with the training step labelled. Rows = Pythia sizes (small to large), columns = checkpoints
(early to late). Provides the visual ground truth that anchors the quantitative analyses in B.1–B.3.

---

## Implementation

### Scripts

```
scripts/
  extract_llm_activations.py   — tokenize stimuli, run forward pass, save hidden states
  build_llm_rdms.py            — mean-centre, compute Pearson distance RDMs, save .npy
  analyze_llm_final.py         — analyses A.1–A.4
  analyze_llm_pythia.py        — analyses B.1–B.4
```

`extract_llm_activations.py` takes `--model-id`, `--step` (for Pythia), and `--output-dir`.
Run one model at a time from the command line.

### Output structure

```
output/production/llm/
  activations/
    EleutherAI__pythia-70m_step143000.npz
    EleutherAI__pythia-70m_step64000.npz
    ...
    meta-llama__Llama-3.2-1B_step143000.npz
    ...
  rdms/
    rdm_EleutherAI__pythia-70m_step143000_Lout.npy
    ...
  figures/
  tables/
    rdm_stats_final.csv
```
