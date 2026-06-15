"""Spec checks for cartpole: env factory shape/action contract and RDM stimuli
grid over (pole_angle, pole_angular_velocity)."""
import numpy as np

from tasks import TASKS
from tasks.cartpole import N_SIDE, POLE_ANGLE_RANGE, POLE_VEL_RANGE


def test_env_factory_returns_working_cartpole():
    task = TASKS["cartpole"]()
    env_factory = task.get_data()
    env = env_factory()

    obs, _ = env.reset()
    assert obs.shape == (task.input_size,) == (4,)

    obs2, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    assert obs2.shape == (4,)
    assert isinstance(reward, (int, float, np.floating))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert env.action_space.n == task.output_size == 2


def test_rdm_stimuli_grid_over_pole_angle_and_velocity():
    task = TASKS["cartpole"]()
    states, meta = task.get_rdm_stimuli()

    assert states.shape == (N_SIDE * N_SIDE, 4)
    # only pole_angle (col 2) and pole_angular_vel (col 3) vary; rest held at 0
    assert (states[:, 0] == 0).all()
    assert (states[:, 1] == 0).all()

    angles = meta["pole_angles"]
    vels   = meta["pole_vels"]
    assert angles.shape == (N_SIDE * N_SIDE,)
    assert vels.shape == (N_SIDE * N_SIDE,)
    np.testing.assert_allclose(states[:, 2], angles)
    np.testing.assert_allclose(states[:, 3], vels)

    assert np.isclose(angles.min(), POLE_ANGLE_RANGE[0])
    assert np.isclose(angles.max(), POLE_ANGLE_RANGE[1])
    assert np.isclose(vels.min(), POLE_VEL_RANGE[0])
    assert np.isclose(vels.max(), POLE_VEL_RANGE[1])

    # exactly N_SIDE distinct values for each axis
    assert len(np.unique(angles)) == N_SIDE
    assert len(np.unique(vels)) == N_SIDE


def test_chance_and_perf_bounds():
    task = TASKS["cartpole"]()
    assert task.chance_perf == 0.0
    assert task.max_metric == 500.0
    assert task.success_threshold == 195.0
