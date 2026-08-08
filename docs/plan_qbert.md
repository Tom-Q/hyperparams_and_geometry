# Q*bert Extension Plan

## Overview

Optional extension O.2 from the main analysis plan. Train A2C networks on Q*bert and apply
representational geometry analyses, testing whether RL findings generalise to a larger-scale,
harder task.

Q*bert is a special case in this project: compute limits mean we can train far fewer networks
than for other tasks (~32 total vs. hundreds), so no Bayesian optimisation is used. Analyses
will be qualitative and descriptive rather than correlational — the main questions are whether
networks show similar representational geometry across HP settings, and how representations
evolve during training.

Four networks ("source networks") were trained first for stimulus extraction and are excluded
from all analyses. The 32 analysis networks use a fresh 7D Sobol design with `use_batch_norm`
fixed to True (see Hyperparameter Space). An additional 2–4 configs will be re-run (repeat 1+)
after training to estimate variability; exact configs chosen after inspecting which reached the
success criterion.

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
  → Linear(3136, hidden_size) → [LayerNorm] → Tanh          ← layer_0 (always saved)
  → [Linear(hidden_size, hidden_size) → [LayerNorm] → Tanh] ← layer_1 (if depth=2)
  → policy_net: Linear(hidden_size, 64) → [LayerNorm] → Tanh → Linear(64, 6)
  → value_net:  Linear(hidden_size, 64) → [LayerNorm] → Tanh → Linear(64, 1)
```

GroupNorm and LayerNorm are included when `use_batch_norm=True` (fixed to True for all
analysis networks). The ResidualBlock uses GroupNorm throughout; an earlier version used
BatchNorm2d there, which is preserved via a `legacy_batch_norm` flag for loading old weights.

**RDM layer:** The last FC layer before the policy/value split — `layer_0` (always) or `layer_1`
(when `depth=2`). Directly analogous to the last hidden layer used for all other tasks. Both
layers are saved in checkpoint npz files when `depth=2`.

### Architectural hyperparameters

- **`hidden_size`** — FC layer width, sampled in [256, 768] log scale. Controls representational
  dimensionality before the policy/value split.

- **`depth`** — number of FC layers (1 or 2). When depth=2, a second `Linear(hidden_size,
  hidden_size)` layer is added after the first. Sampled as a Sobol boolean.

- **`use_batch_norm`** — GroupNorm on conv layers, LayerNorm on FC layers. Fixed to True for
  all analysis networks: a pilot run with the original 8D design (use_batch_norm sampled)
  found that every network with use_batch_norm=False failed to learn (patience stop or
  early_failure), while all use_batch_norm=True networks succeeded.

- **`use_residual`** — adds a ResidualBlock (two 3×3 convs with skip connection) after conv3,
  on the (B, 64, 7, 7) feature map.

- **`use_attention`** — adds a `SpatialAttentionModule` after conv3 (after the residual block
  if also active). This is **multi-head self-attention (MHSA)**, not the SE channel attention
  from the original `base_model.py`. The original used global-average-pool channel gating,
  which is blind to spatial position — inappropriate for Q*bert where the character's location
  on the pyramid is the key signal. The replacement:
  - Treats each of the 49 spatial positions (7×7) as a token (64-d feature vector)
  - Adds learnable positional embeddings (49 × 64) so tokens are position-aware
  - Applies pre-norm MHSA: LayerNorm → 4-head self-attention (16-d per head) → residual
  - Reshapes back to (B, 64, 7, 7); drop-in for the forward pass

---

## Training

- **Algorithm:** A2C with PPO-style clipped surrogate loss (`clip_ratio=0.2`)
- **Environments:** 16 parallel envs (`ALE/Qbert-v5`, `SyncVectorEnv`)
  - `AtariPreprocessing`: `frame_skip=1` (ALE/Qbert-v5 applies 4-frame skip internally),
    84×84 grayscale, scale to [0,1], `noop_max=30`, `terminal_on_life_loss=True` (training)
  - `FrameStackObservation(4)`
- **Max steps:** 60M env steps (~7,324 gradient updates at rollout_size=8192); may stop earlier
  via early stopping (see below)
- **Rollout:** 512 steps × 16 envs = 8192 transitions per update
- **Evaluation:** every 200k env steps, 20 stochastic episodes, single env,
  `terminal_on_life_loss=False`
- **Performance metric:** mean raw episode score (not sign-clipped)
- **Reward clipping:** training uses sign-clipped rewards (+1/−1); raw scores tracked via
  EMA (α=0.01 per episode) for logging and checkpoint metadata

### Early stopping

Three criteria, checked after every evaluation:

1. **Success:** if `frac_level5 ≥ 0.5` (level 5 reached in ≥50% of eval episodes), training
   stops. `metadata.json` records `"stop_reason": "success"`.

2. **Patience:** if `eval_mean` has not improved in 10M env steps, training stops.
   `metadata.json` records `"stop_reason": "patience"`.

3. **Early failure:** if level 2 has not been reached by 5M env steps, training stops.
   `metadata.json` records `"stop_reason": "early_failure"`.

Normal completion records `"stop_reason": "completed"`. In all cases, `final.npz` is saved
after the stopping condition is detected.

### Level logging

Every evaluation writes one row to `training_log.csv`:

```
step, eval_mean, eval_std, ema_reward,
frac_level2, steps_to_level2_mean, steps_to_level2_std,
...
frac_level20, steps_to_level20_mean, steps_to_level20_std
```

`frac_levelN` is the fraction of eval episodes that reached level N. `steps_to_levelN_*` is
the distribution of episode step counts at which level N was first reached (NaN if no episode
reached it). Level tracking uses ALE RAM address 99, which stores the cumulative
levels-completed count (0 = on level 1, 1 = on level 2, …). Address 57 tracks the same thing
for levels 1–4 but wraps back to 1 when level 5 is completed; addr 99 is monotonic.

---

## Hyperparameter Space

No Bayesian optimisation. 7D Sobol sampling, 32 draws. `use_batch_norm` fixed to True.

**Design:** 7 hyperparameters sampled jointly in a single Sobol sequence (seed=42, scrambled).
Continuous dimensions are log- or linearly mapped; boolean dimensions are thresholded at 0.5.

| Dim | HP | Range | Scale | Boolean threshold |
|---|---|---|---|---|
| 0 | `learning_rate` | [1×10⁻⁴, 1×10⁻³] | log | — |
| 1 | `entropy_coef` | [5×10⁻³, 1×10⁻¹] | log | — |
| 2 | `gamma` | [0.98, 0.995] | linear | — |
| 3 | `hidden_size` | [256, 768] | log | — |
| 4 | `use_attention` | — | — | ≥ 0.5 → True |
| 5 | `use_residual` | — | — | ≥ 0.5 → True |
| 6 | `depth` | — | — | ≥ 0.5 → 2, else 1 |

Gamma floor raised to 0.98 (vs. 0.97 in the other RL tasks). Q*bert episodes run 500+ steps
to clear multiple levels; at gamma=0.97, a reward 100 steps away is discounted to ~0.05,
making long-horizon board-clearing nearly invisible to the agent.

All other parameters fixed: `value_coef=0.5`, `clip_ratio=0.2`, `clip_grad_norm=0.5`,
`n_steps=512`, `update_epochs=2`, `batch_size=256`.

Run `python scripts/run_qbert_network.py --list` for the full config table.

**Repeat runs:** after primary training, 2–4 configs that reached the success criterion will
be re-run (`--repeat 1`, `--repeat 2`) to estimate intra-config variability.

---

## Checkpoint Scheme

Checkpoints follow the METHODS.md conventions adapted for RL (no epoch checkpoints).

### Step checkpoints (`step_XXXXXXX.npz`)

Log₄-spaced gradient update counts, with supplementary uniform coverage in the later range
where log₄ spacing becomes coarse:

**1, 4, 16, 64, 256, 1024, 2048, 3072, 4096, 5120, 6144, 7168**

(Log₄ sequence: 1, 4, 16, 64, 256, 1024, 4096. Additional every-1024 from 2048 onward.)
Filename uses the gradient update count (7 digits); the file's `step` field stores the
corresponding env step count. Early-stopped runs will have fewer step checkpoints.

### Performance checkpoints (`perf_X.npz`)

Saved when normalised performance `score / 15000` first crosses each threshold:
0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95.
Only fires for thresholds the network actually reaches.

### Other checkpoints

- **`best.npz` + `best_weights.pt`** — saved whenever evaluation score improves
- **`final.npz`** — end of training (after any early stop)

### npz contents

Every checkpoint file contains:
- `layer_0` — shape (53, hidden_size) float32; first FC activations over the 53 stimuli
- `layer_1` — shape (53, hidden_size) float32; second FC activations (only when `depth=2`)
- `step` — int64; env step count at time of save
- `avg_reward` — float32; EMA of raw episode reward at time of save; `nan` before first episode

---

## Stimuli

**53 stimuli total** (53×53 RDM), extracted from four source networks.

**Composition:**
- 5 level-start frames (one per level, levels 1–5)
- 12 intermediate frames per level × 4 levels = 48 frames (3 per source network per level)
- Total: 5 + 48 = 53

**Every stimulus comes from a different episode that reached level 5.** Frames from episodes
that did not clear level 4 are never used. Intermediate frames are drawn uniformly at random
from the interior of the level segment (excluding first and last frames).

**Stimuli are sorted in pseudo-chronological order** (by level then within-level progress) so
that RDMs are visually interpretable without reordering.

**Source networks** (not in analysis set):

| Network | lr | entropy | gamma | hidden | attn | residual | depth |
|---|---|---|---|---|---|---|---|
| run_model_r1 | 3×10⁻⁴ | 0.010 | 0.990 | 512 | ✓ | ✓ | 1 |
| run_0000_r0 | 9.9×10⁻⁴ | 0.0068 | 0.992 | 406 | ✗ | ✗ | 2 |
| run_0003_r0 | 5.5×10⁻⁴ | 0.036 | 0.981 | 742 | ✗ | ✓ | 2 |
| run_0004_r0 | 3.5×10⁻⁴ | 0.013 | 0.986 | 634 | ✗ | ✗ | 1 |

`run_model_r1` was trained with an older codebase (BatchNorm2d in ResidualBlock rather than
GroupNorm) and is loaded with the `legacy_batch_norm=True` flag, which restores BatchNorm2d
for that block so weights load correctly with `strict=True`.

**Level-start assignments** (which source network provides each level's start frame):

| Level | Network |
|-------|---------|
| 1 | run_model_r1 |
| 2 | run_0000_r0 |
| 3 | run_0003_r0 |
| 4 | run_0004_r0 |
| 5 | run_model_r1 |

Intermediate frames are distributed evenly (3 per network per level) across all four source
networks.

Frames are preprocessed input tensors, shape (4, 84, 84) float32. Saved to
`output/production/qbert/stimuli.npz` (keys: `inputs`, `levels`, `frame_types`, `source_nets`,
`progress`).

---

## Output Structure

```
output/production/qbert/
  stimuli.npz
  stimuli_images/
  bo_state.json
  run_model_r1/          (source network, excluded from analysis)
  run_0000_r0/           (source network, excluded from analysis)
  run_0003_r0/           (source network, excluded from analysis)
  run_0004_r0/           (source network, excluded from analysis)
  archived_runs/         (pilot 8D Sobol run, not used in analysis)
  run_0000_r0/           (analysis network #0, repeat 0)
  ...
  run_0031_r0/           (analysis network #31, repeat 0)
  run_0000_r1/           (analysis network #0, repeat 1)
  ...
```

`bo_state.json` — one entry per primary trained network:
```json
[{"iteration": 0, "performance": <best_metric>, "config": { ... }}, ...]
```

---

## Success Criterion

A network is considered **functional** if it achieves `frac_level5 ≥ 0.5` at any evaluation
point — i.e., it reaches level 5 in at least half of eval episodes. This is also the early
stopping trigger. Non-functional networks (stopped by patience, early_failure, or completing
60M steps without reaching the criterion) are excluded from all analyses.

---

## Analysis

Same pipeline as CartPole and FourRooms. Priority analyses given n≈32:

1. **Noise ceiling** — inter-network RDM agreement among functional networks
2. **Representational similarity structure** — do architectural groups (attention on/off,
   residual on/off) cluster in RDM space?
3. **Training dynamics** — how do representations change over training (step checkpoints)?
   When do they stabilise (crystallisation)?
4. **Performance checkpoints** — does representational geometry differ between networks
   that reach high performance thresholds vs. those that plateau early?

HP correlation analyses (Finding #2 style) are underpowered at n=32 — treated as
exploratory at best.
