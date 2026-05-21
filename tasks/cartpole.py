"""Task 7: CartPole-v1 via online Q-learning."""
import numpy as np
from .base import Task
from ._shared import RL_CATS

# Grid over (pole_angle, pole_angular_velocity); other dims held at 0.
N_SIDE           = 14   # 14×14 = 196 stimuli
POLE_ANGLE_RANGE = (-0.2, 0.2)
POLE_VEL_RANGE   = (-2.0, 2.0)


def _make_cartpole_stimuli():
    angles = np.linspace(*POLE_ANGLE_RANGE, N_SIDE)
    vels   = np.linspace(*POLE_VEL_RANGE,   N_SIDE)
    grid_a, grid_v = np.meshgrid(angles, vels)
    # state: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
    states = np.zeros((N_SIDE * N_SIDE, 4), dtype=np.float32)
    states[:, 2] = grid_a.ravel()
    states[:, 3] = grid_v.ravel()
    metadata = {
        "pole_angles": grid_a.ravel().astype(np.float32),
        "pole_vels":   grid_v.ravel().astype(np.float32),
    }
    return states, metadata


class CartPoleTask(Task):
    name              = "cartpole"
    paradigm          = "rl"
    input_size        = 4   # [cart_pos, cart_vel, pole_angle, pole_ang_vel]
    output_size       = 2   # Q-values for left/right
    n_steps           = None
    hidden_size_range = (4, 256)
    success_threshold = 195.0   # CartPole-v1 standard: avg return ≥ 195 over 100 eps
    metric_name       = "mean_return"
    max_steps         = 500_000

    def get_data(self, data_dir="data", seed=42):
        import gymnasium as gym
        def env_factory():
            return gym.make("CartPole-v1")
        return env_factory

    def get_rdm_stimuli(self, data_dir="data", seed=42):
        return _make_cartpole_stimuli()

    def categorical_space(self):
        return RL_CATS
