"""Spec checks for fourrooms: env factory shape/action contract, RBF encoding,
and RDM stimuli covering all free cells."""
import numpy as np

from tasks import TASKS
from tasks.fourrooms import FREE_CELLS, GRID, GOAL_POS, N_RBF, _rbf_encode


def test_env_factory_returns_working_fourrooms():
    task = TASKS["fourrooms"]()
    env_factory = task.get_data()
    env = env_factory()

    obs, _ = env.reset()
    assert obs.shape == (task.input_size,) == (N_RBF,)
    assert np.isfinite(obs).all()

    obs2, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    assert obs2.shape == (N_RBF,)
    assert reward == -0.01
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert env.action_space.n == task.output_size == 4


def test_reset_never_starts_on_wall_or_goal():
    task = TASKS["fourrooms"]()
    env_factory = task.get_data()
    env = env_factory()
    for _ in range(200):
        env._pos = None
        env.reset()
        r, c = env._pos
        assert GRID[r, c] == 0
        assert (r, c) != GOAL_POS


def test_rdm_stimuli_cover_all_free_cells():
    task = TASKS["fourrooms"]()
    inputs, meta = task.get_rdm_stimuli()

    assert inputs.shape == (N_RBF, N_RBF) == (len(FREE_CELLS), len(FREE_CELLS))
    rows, cols = meta["rows"], meta["cols"]
    assert rows.shape == (N_RBF,)
    assert cols.shape == (N_RBF,)

    cells = list(zip(rows.tolist(), cols.tolist()))
    assert cells == FREE_CELLS

    # every cell is free (not a wall) per the grid
    for r, c in cells:
        assert GRID[r, c] == 0

    # each stimulus matches the RBF encoding of its own cell
    for i, (r, c) in enumerate(cells):
        np.testing.assert_allclose(inputs[i], _rbf_encode((r, c)))


def test_chance_and_perf_bounds():
    task = TASKS["fourrooms"]()
    assert task.chance_perf == -1.0
    assert task.max_metric == 0.0
    assert task.success_threshold == -0.2
