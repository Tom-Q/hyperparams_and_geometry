# Q*bert Extension Plan

## Overview

Optional extension O.2 from the main analysis plan. Train ~20 A2C networks on Q*bert and apply
the same representational geometry analyses as CartPole and FourRooms, testing whether RL findings
generalise to a larger-scale, harder task.

One network ("model network") is trained first, used only for stimulus extraction, and excluded
from all analyses. The remaining ~19 networks are the analysis set.

---

## Architecture

Adapted from the `atari_acc` repo (`base_model.py`). NatureCNN-inspired:

```
Input: 84×84×4 (grayscale, 4 stacked frames)
  → Conv(32, 8×8, stride 4) → GroupNorm → ReLU
  → Conv(64, 4×4, stride 2) → GroupNorm → ReLU
  → Conv(64, 3×3, stride 1) → GroupNorm → ReLU
  → Flatten → 3136-d
  → Linear(3136, 512) → [LayerNorm] → Tanh   ← perception_fc (RDM layer)
  → policy_net: Linear(512, 64) → Tanh → Linear(64, N_actions)
  → value_net:  Linear(512, 64) → Tanh → Linear(64, 1)
```

**RDM layer:** `perception_fc` output (512-d). This is the shared bottleneck before the
policy/value split — directly analogous to the last hidden layer used for all other tasks.

Optional architectural flags (categorical HPs): `use_attention`, `use_residual`, `use_batch_norm`.

---

## Training

- **Algorithm:** A2C (custom implementation from `base_model.py`)
- **Environments:** 16 parallel Atari envs (`ALE/Qbert-v5`)
- **Total steps:** up to 60M, stopping early once the success threshold is reached
- **Success criterion:** mean episode score ≥ 15,000 (approximately clearing levels 5–6),
  used as the stopping signal and as the upper bound for defining the analysis scope
- **Performance metric:** mean raw episode score (continuous, higher = better)
- **Evaluation:** every 200k steps, 10 episodes, deterministic policy
- **Best model:** saved when evaluation score improves (analogous to `model_best.pt`)

---

## Hyperparameter Space

No Bayesian optimisation at this scale. Manual grid/random sampling across:

| HP | Type | Range |
|---|---|---|
| `learning_rate` | continuous | 0.0001 – 0.001 |
| `entropy_coef` | continuous | 0.005 – 0.05 |
| `gamma` | continuous | 0.97 – 0.995 |
| `use_batch_norm` | categorical | True / False |
| `use_attention` | categorical | True / False |
| `use_residual` | categorical | True / False |

`clip_ratio` and LR decay schedule are fixed (clip=0.2, linear decay to 10% of initial LR).
All other architecture parameters are fixed (same CNN backbone, same head sizes).

---

## Stimuli

**Source:** all stimuli are extracted from the **model network's** gameplay only — saved as
preprocessed input tensors (84×84×4 float32). For analysis networks, RDMs are computed by
passing these fixed tensors through the trained `perception_fc` via a single forward pass;
the Atari environment is not needed.

**Composition (44 stimuli total → 44×44 RDM):**
- 4 level-start frames (one per level, levels 1–4): the first frame of each level, identical
  across all runs of the model network
- 10 in-between frames per level (levels 1–4) = 40 frames: selected using the event detection
  infrastructure from `atari_acc` (level completion %, enemy appearances, Q*bert position
  diversity), ensuring coverage of meaningfully different board states within each level

**Scope:** levels 1–4 only, even though training targets level 5–6. This keeps the stimulus
set interpretable and ensures all analysis networks have visited the relevant game states.

---

## Infrastructure (to be built in this repo)

Relevant code from `atari_acc` will be copied into `src/qbert/` and adapted:
- `base_model.py` → `src/qbert/base_model.py` (network + training loop)
- `model_configurations.py` → `src/qbert/model_configurations.py`
- Stimulus extraction utilities from `ai_acc/event_analysis.py` and `ai_acc/analysis.py`

**What needs to be added:**

1. **Step-checkpoint saving** — at each checkpoint, save `perception_fc` activations over the
   44 stimuli as `step_XXXXXXX.npz`. Checkpoint spacing TBD (likely fixed intervals of ~2–5M
   steps rather than log₄, given the much longer training runs).

2. **`metadata.json`** — written at end of training:
   ```json
   {
     "task": "qbert",
     "config": { ... full HP config ... },
     "best_step": ...,
     "best_metric": <best mean episode score>,
     "final_step": ...,
     "final_metric": <final mean episode score>
   }
   ```

3. **`bo_state.json`** — one entry per trained network (manually constructed, not BO-driven):
   ```json
   [{"iteration": 0, "performance": <best_metric>, "config": { ... }}, ...]
   ```
   Performance stored as raw score (higher = better, consistent with other tasks).

4. **`final.npz` / `best.npz`** — activations over the 44 stimuli at the final step and at
   the best-scoring checkpoint respectively (following the RL convention of `final.npz` only,
   since the best checkpoint corresponds to peak performance).

5. **Stimulus file** — `output/production/qbert/stimuli.npz` containing the 44 input tensors
   extracted from the model network, plus metadata (level, event type, frame index).

**Output directory structure** follows the existing convention:
```
output/production/qbert/
  stimuli.npz
  bo_state.json
  run_0000_r0/
    metadata.json
    best.npz
    final.npz
    step_2000000.npz
    step_4000000.npz
    ...
```

---

## Success Threshold

With ~19 analysis networks and no BO, the p90-based threshold from `02_performance_lorenz.py`
is not meaningful at this scale. Instead, use a fixed threshold based on game structure:

- **Successful:** mean episode score ≥ 5,000 (reliably clearing level 2+)
- **Failed:** mean episode score < 5,000

Exact value TBD after training the model network and inspecting the score distribution.

---

## Analysis

Same pipeline as CartPole and FourRooms. Priority analyses:

1. **Noise ceiling** (Finding #1.1) — inter-network RDM agreement among successful networks
2. **Category structure** (Finding #1.3) — geometric model TBD (board state, level, position)
3. **HP effects** (Finding #2.1) — Spearman/ANOVA of HP vs. RDM summary statistics
4. **Critical period** (Finding #3.2) — rate of representational change over training
5. **Crystallisation** (Finding #3.1) — when does the RDM stabilise?

Finding #4 (early prediction) is likely underpowered at n≈19 — skip unless results are striking.

---

## Open Questions

- Exact step-checkpoint interval (depends on training speed on available hardware)
- Geometric category model for Q*bert (level completion %, enemy proximity, Q*bert position?)
- Whether 19 networks is sufficient for HP effects analysis given the small n
