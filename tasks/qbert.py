"""Task: Q*bert via A2C on ALE/Qbert-v5.

No Bayesian optimisation — networks are trained from a manual grid defined in
scripts/run_qbert_network.py. This task class provides the shared interface
(HP space, stimuli, env factory) for analysis scripts.

Stimuli must be pre-built before training analysis networks:
  python scripts/extract_qbert_stimuli.py --model-run-dir output/production/qbert/run_model_r0

Paradigm "rl" means analysis scripts group Q*bert with CartPole/FourRooms.
Training uses src/qbert/train.py, not src/train_rl.py.
"""
from pathlib import Path

import numpy as np

from .base import Task

REPO_ROOT = Path(__file__).parent.parent

# Analysis networks have use_batch_norm fixed to True — not included here.
# depth is a Sobol-sampled boolean (1 or 2).
QBERT_HP_CATEGORICAL = {
    "use_attention":  [True, False],
    "use_residual":   [True, False],
    "depth":          [1, 2],
}


class QbertTask(Task):
    name      = "qbert"
    paradigm  = "rl"              # grouped with RL tasks in analysis

    input_size  = (4, 84, 84)     # stacked grayscale frames, post-AtariPreprocessing
    output_size = 6                # Q*bert discrete action space
    n_steps     = None

    # Thresholds and metrics.
    # Note: success is determined by frac_level5 >= 0.5, not by this score threshold.
    # Analysis scripts use the is_functional HDF5 attribute instead.
    success_threshold = 15_000.0
    chance_perf       = 0.0
    max_metric        = 15_000.0
    metric_name       = "mean_episode_score"
    max_steps         = 60_000_000

    def get_data(self, data_dir="data", seed=42):
        """Return an env factory for a single (non-vectorised) Q*bert env."""
        def env_factory():
            import ale_py
            import gymnasium as gym
            from gymnasium.wrappers import FrameStackObservation
            from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
            gym.register_envs(ale_py)
            env = gym.make("ALE/Qbert-v5", render_mode=None)
            env = AtariPreprocessing(env, noop_max=30, frame_skip=1, screen_size=84,
                                     terminal_on_life_loss=True, grayscale_obs=True,
                                     grayscale_newaxis=False, scale_obs=True)
            return FrameStackObservation(env, 4)
        return env_factory

    def get_rdm_stimuli(self, data_dir="data", seed=42):
        """Load pre-built stimuli from output/production/qbert/stimuli.npz.

        Returns (inputs, metadata) where inputs has shape (53, 4, 84, 84) float32.
        Run scripts/extract_qbert_stimuli.py to generate stimuli.npz first.
        """
        path = REPO_ROOT / "output" / "production" / "qbert" / "stimuli.npz"
        assert path.exists(), (
            f"Stimuli file not found: {path}\n"
            "Run: python scripts/extract_qbert_stimuli.py --model-run-dir <model_run_dir>"
        )
        data = np.load(path)
        inputs   = data["inputs"]
        metadata = {k: data[k] for k in data.files if k != "inputs"}
        return inputs, metadata

    def categorical_space(self):
        return QBERT_HP_CATEGORICAL

    def cont_param_ranges(self):
        return [
            ("learning_rate", 0.0001, 0.001),
            ("entropy_coef",  0.005,  0.1),
            ("gamma",         0.98,   0.995),
            ("hidden_size",   256,    768),    # log scale
        ]
