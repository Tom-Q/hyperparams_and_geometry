# Q*bert Extension Plan

## Overview

Optional extension O.2 from the main analysis plan. Train A2C networks on Q*bert and apply
representational geometry analyses, testing whether RL findings generalise to a larger-scale,
harder task.

Q*bert is a special case in this project: compute limits mean we can train far fewer networks
than for other tasks (~25 total vs. hundreds), so no Bayesian optimisation is used. Analyses
will be qualitative and descriptive rather than correlational — the main questions are whether
networks show similar representational geometry across HP settings, and how representations
evolve during training.

One network ("model network") is trained first for stimulus extraction and excluded from all
analyses. The remaining 24 are the analysis set. An additional 2–4 configs will be re-run
(repeat 1+) after training to estimate variability; exact configs chosen after inspecting which
reached level 4.

---

## Architecture

Adapted from `atari_acc/ai_acc/base_model.py` (`ImprovedA2CNetwork`). NatureCNN backbone:

```
Input: (4, 84, 84) — 4 stacked grayscale frames, scaled to [0, 1]
  → Conv(32, 8×8, stride 4) → [GroupNorm(8, 32)] → ReLU
  → Conv(64, 4×4, stride 2) → [GroupNorm(16, 64)] → ReLU
  → Conv(64, 3×3, stride 1) → [GroupNorm(16, 64)] → ReLU   → (B, 64, 7, 7)
  → [ResidualBlock(64)]                                      if use_residual
  → [SpatialAttentionModule]                                 if use_attention
  → Flatten → 3136-d
  → Linear(3136, 512) → [LayerNorm(512)] → Tanh             ← perception_fc (RDM layer)
  → policy_net: Linear(512, 64) → [LayerNorm] → Tanh → Linear(64, 6)
  → value_net:  Linear(512, 64) → [LayerNorm] → Tanh → Linear(64, 1)
```

GroupNorm and LayerNorm are only included when `use_batch_norm=True`.

**RDM layer:** `perception_fc` output (512-d). Shared bottleneck before the policy/value
split — directly analogous to the last hidden layer used for all other tasks. Saved as
`layer_0` in all checkpoint npz files. `metadata.json` records `"depth": 1` so existing
analysis scripts pick up `layer_0` automatically.

### Architectural flags

- **`use_batch_norm`** — GroupNorm on conv layers, LayerNorm on FC layers. Meaningful axis
  for representational geometry; normalization structure affects geometry significantly.

- **`use_residual`** — adds a ResidualBlock (two 3×3 convs with skip connection) after conv3,
  on the (B, 64, 7, 7) feature map. Adds processing depth before flattening.

- **`use_attention`** — adds a `SpatialAttentionModule` after conv3 (after the residual block
  if also active). This is **multi-head self-attention (MHSA)**, not the SE channel attention
  from the original `base_model.py`. The original used global-average-pool channel gating,
  which is blind to spatial position — inappropriate for Q*bert where the character's location
  on the pyramid is the key signal. The replacement:
  - Treats each of the 49 spatial positions (7×7) as a token (64-d feature vector)
  - Adds learnable positional embeddings (49 × 64) so tokens are position-aware
  - Applies pre-norm MHSA: LayerNorm → 4-head self-attention (16-d per head) → residual
  - Reshapes back to (B, 64, 7, 7); drop-in for the forward pass
  - Allows each spatial position to attend to all others, capturing spatial relationships
    (e.g. Q*bert's position relative to uncolored tiles)

---

## Training

- **Algorithm:** A2C with PPO-style clipped surrogate loss (`clip_ratio=0.2`)
- **Environments:** 16 parallel envs (`ALE/Qbert-v5`, `SyncVectorEnv`)
  - `AtariPreprocessing`: `frame_skip=1` (ALE/Qbert-v5 applies 4-frame skip internally),
    84×84 grayscale, scale to [0,1], `noop_max=30`, `terminal_on_life_loss=True` (training)
  - `FrameStackObservation(4)`
- **Total steps:** 60M env steps (~7,324 gradient updates at rollout_size=8192)
- **Rollout:** 512 steps × 16 envs = 8192 transitions per update
- **Evaluation:** every 200k env steps, 10 stochastic episodes, single env,
  `terminal_on_life_loss=False`
- **Performance metric:** mean raw episode score (not sign-clipped)
- **Reward clipping:** training uses sign-clipped rewards (+1/−1); raw scores tracked via
  EMA (α=0.01 per episode) for logging and checkpoint metadata

**Deviation from METHODS.md RL convention:** analysis RL tasks (CartPole, FourRooms) save only
`final.npz` because training stops at the success threshold, making final = peak. Q*bert trains
for the full 60M steps regardless of score, so `best.npz` and `best_weights.pt` are saved
separately whenever evaluation score improves.

---

## Hyperparameter Space

No Bayesian optimisation. Sobol sampling in a narrow, task-appropriate range.

**Design:** 8 categorical combinations × 3 Sobol continuous points = 24 analysis networks.
The same 3 continuous HP points are used for every categorical combo, so architectural
differences between combos are not confounded with continuous HP variation.

**Continuous ranges** (log scale for lr and entropy_coef):

| HP | Range | Scale |
|---|---|---|
| `learning_rate` | [1×10⁻⁴, 1×10⁻³] | log |
| `entropy_coef` | [5×10⁻³, 1×10⁻¹] | log |
| `gamma` | [0.98, 0.995] | linear |

Gamma floor raised to 0.98 (vs. 0.97 in the other RL tasks). Q*bert episodes run 500+ steps
to clear multiple levels; at gamma=0.97, a reward 100 steps away is discounted to ~0.05,
making long-horizon board-clearing nearly invisible to the agent.

**Categorical combinations** (all 2³ = 8):

| `use_batch_norm` | `use_attention` | `use_residual` |
|---|---|---|
| True | True | True |
| True | True | False |
| True | False | True |
| True | False | False |
| False | True | True |
| False | True | False |
| False | False | True |
| False | False | False |

**Sobol sampling** (seed=42, scrambled): 24 points drawn in 3D continuous space,
shuffled (rng seed=42), assigned 3 per categorical combo. Each combo gets distinct
continuous values; run `python scripts/run_qbert_network.py --list` for the full table.

All other parameters fixed: `value_coef=0.5`, `clip_ratio=0.2`, `clip_grad_norm=0.5`,
`n_steps=512`, `update_epochs=2`, `batch_size=256`.

**Model network config** (not in analysis set):
`lr=0.0003, entropy=0.01, gamma=0.99, use_batch_norm=True, use_attention=False, use_residual=False`

**Repeat runs:** after primary training, 2–4 configs that reached level 4 will be re-run
(`--repeat 1`, `--repeat 2`) to estimate intra-config variability. Specific configs chosen
after inspecting results.

---

## Checkpoint Scheme

Checkpoints follow the METHODS.md conventions adapted for RL (no epoch checkpoints).

### Step checkpoints (`step_XXXXXXX.npz`)

Log₄-spaced gradient update counts, with supplementary uniform coverage in the later range
where log₄ spacing becomes coarse:

**1, 4, 16, 64, 256, 1024, 2048, 3072, 4096, 5120, 6144, 7168**

(Log₄ sequence: 1, 4, 16, 64, 256, 1024, 4096. Additional every-1024 from 2048 onward.)
At 7324 total updates for a 60M step run, this gives 12 step checkpoints.
Filename uses the gradient update count (7 digits); the file's `step` field stores the
corresponding env step count.

### Performance checkpoints (`perf_X.npz`)

Saved when normalised performance `score / 15000` first crosses each threshold:
0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95.
Only fires for thresholds the network actually reaches.

### Other checkpoints

- **`best.npz` + `best_weights.pt`** — saved whenever evaluation score improves
- **`final.npz`** — end of training

### npz contents

Every checkpoint file contains:
- `layer_0` — shape (44, 512) float32; perception_fc activations over the 44 stimuli
- `step` — int64; env step count at time of save
- `avg_reward` — float32; EMA of raw episode reward at time of save (α=0.01/episode);
  `nan` if no episode has completed yet

---

## Stimuli

**44 stimuli total** (44×44 RDM), extracted from the model network's gameplay only.

**Composition:**
- 4 level-start frames (one per level, levels 1–4)
- 10 in-level frames per level (levels 1–4) = 40 frames

Frames are preprocessed input tensors, shape (4, 84, 84) float32. Saved to
`output/production/qbert/stimuli.npz` (keys: `inputs`, `levels`, `frame_types`).

Level detection uses the visual tile-color approach from `atari_acc/analysis.py`.
The Atari environment is not needed to compute RDMs for analysis networks — stimuli are
passed directly through `perception_fc` in a single forward pass.

**Scope:** levels 1–4 only, ensuring all functional analysis networks have encountered
these game states during training.

---

## Output Structure

```
output/production/qbert/
  stimuli.npz
  bo_state.json
  run_model_r0/
    metadata.json
    best_weights.pt
    best.npz
    final.npz
    step_0000001.npz  ...  step_7168000.npz  (up to 12)
    perf_0p025.npz    ...  (up to 10)
  run_0000_r0/   (analysis network #0, repeat 0)
    ...
  run_0009_r1/   (analysis network #9, repeat 1)
    ...
```

`bo_state.json` — one entry per primary trained network (manually appended, not BO-driven):
```json
[{"iteration": 0, "performance": <best_metric>, "config": { ... }}, ...]
```

---

## Success Criterion

A network is considered functional if it reaches **at least level 4** (raw score roughly
≥ 4,000–5,000). Non-functional networks are excluded from all analyses. Exact threshold
confirmed after training the model network. With 24 analysis networks and narrow HP ranges
anchored near a known-working config, most networks are expected to be functional.

---

## Analysis

Same pipeline as CartPole and FourRooms. Priority analyses given n≈24:

1. **Noise ceiling** — inter-network RDM agreement among functional networks
2. **Representational similarity structure** — do architectural groups (batch_norm on/off,
   attention on/off, residual on/off) cluster in RDM space?
3. **Training dynamics** — how do representations change over training (step checkpoints)?
   When do they stabilise (crystallisation)?
4. **Performance checkpoints** — does representational geometry differ between networks
   that reach high performance thresholds vs. those that plateau early?

HP correlation analyses (Finding #2 style) are underpowered at n=24 — treated as
exploratory at best.
