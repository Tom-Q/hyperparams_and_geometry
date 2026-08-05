"""Task: Q*bert via A2C on ALE/Qbert-v5.

No Bayesian optimisation — networks are trained from a manual grid defined in
scripts/run_qbert_network.py. This task class provides the shared interface
(HP space, stimuli, env factory) for analysis scripts.

Stimuli must be pre-built before training analysis networks:
  python scripts/extract_qbert_stimuli.py --model-run-dir output/production/qbert/run_model_r0

Paradigm "qbert" is handled by src/qbert/train.py, not src/train_rl.py.
"""
from pathlib import Path

import numpy as np

from .base import Task

REPO_ROOT = Path(__file__).parent.parent

QBERT_HP_CATEGORICAL = {
    "use_batch_norm": [True, False],
    "use_attention":  [True, False],
    "use_residual":   [True, False],
}


class QbertTask(Task):
    name      = "qbert"
    paradigm  = "qbert"           # custom; not dispatched through run_task.py

    input_size  = (4, 84, 84)     # stacked grayscale frames, post-AtariPreprocessing
    output_size = 6                # Q*bert discrete action space
    n_steps     = None

    # Thresholds and metrics
    success_threshold = 15_000.0  # mean eval episode score (early-stop criterion)
    chance_perf       = 0.0       # random-policy baseline
    max_metric        = 15_000.0  # ceiling for normalisation (same as success threshold)
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

        Returns (inputs, metadata) where inputs has shape (44, 4, 84, 84) float32.
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
        ]
