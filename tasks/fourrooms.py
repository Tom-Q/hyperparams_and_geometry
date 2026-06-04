"""Task 8: FourRooms gridworld with RBF state encoding + online Q-learning."""
import numpy as np
from .base import Task
from ._shared import RL_HYPERPARAMS

# FourRooms: 11×11 grid with internal walls creating 4 rooms.
GRID = np.array([
    [1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,1,0,0,0,0,1],
    [1,0,0,0,0,1,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,1,0,0,0,0,1],
    [1,1,0,1,1,1,1,1,0,1,1],
    [1,0,0,0,0,1,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,1,0,0,0,0,1],
    [1,0,0,0,0,1,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1],
], dtype=np.int8)

GOAL_POS    = (9, 9)
ACTIONS     = [(-1,0),(1,0),(0,-1),(0,1)]   # up, down, left, right
N_ACTIONS   = len(ACTIONS)
FREE_CELLS  = [(r, c) for r in range(11) for c in range(11) if GRID[r, c] == 0]
RBF_SIGMA   = 1.5
CENTRES_ARR = np.array(FREE_CELLS, dtype=np.float32)   # precomputed, shape (N, 2)
N_RBF       = len(FREE_CELLS)


def _rbf_encode(pos):
    """Encode a (row, col) grid position as an RBF feature vector."""
    p = np.array(pos, dtype=np.float32)
    dists_sq = ((CENTRES_ARR - p) ** 2).sum(axis=1)
    return np.exp(-dists_sq / (2 * RBF_SIGMA ** 2)).astype(np.float32)


class FourRoomsEnv:
    """Minimal FourRooms implementation (no Gymnasium dependency)."""

    def __init__(self, max_steps=500):
        self.max_steps   = max_steps
        self._step_count = 0
        self._pos        = None

    class _ActionSpace:
        def __init__(self, n):
            self.n = n
        def sample(self):
            return np.random.randint(self.n)

    @property
    def action_space(self):
        return self._ActionSpace(N_ACTIONS)

    def reset(self):
        self._step_count = 0
        while True:
            r = np.random.randint(1, 10)
            c = np.random.randint(1, 10)
            if GRID[r, c] == 0 and (r, c) != GOAL_POS:
                self._pos = (r, c)
                break
        return _rbf_encode(self._pos), {}

    def step(self, action):
        self._step_count += 1
        dr, dc = ACTIONS[action]
        nr, nc = self._pos[0] + dr, self._pos[1] + dc
        if GRID[nr, nc] == 0:
            self._pos = (nr, nc)
        reached_goal = self._pos == GOAL_POS
        reward       = 1.0 if reached_goal else -0.01
        done         = reached_goal or self._step_count >= self.max_steps
        return _rbf_encode(self._pos), reward, done, False, {}

    def close(self):
        pass


class FourRoomsTask(Task):
    name              = "fourrooms"
    paradigm          = "rl"
    input_size        = N_RBF
    output_size       = N_ACTIONS
    n_steps           = None
    success_threshold = 0.8    # set empirically after pre-testing
    chance_perf       = -5.0  # random policy return (always times out)
    max_metric        = 1.0   # approximate maximum achievable return
    metric_name       = "mean_return"
    max_steps         = 100_000

    def get_data(self, data_dir="data", seed=42):
        def env_factory():
            return FourRoomsEnv()
        return env_factory

    def get_rdm_stimuli(self, data_dir="data", seed=42):
        """All non-wall cells, each RBF-encoded. Ground truth = 2D grid position."""
        inputs   = np.stack([_rbf_encode(cell) for cell in FREE_CELLS]).astype(np.float32)
        metadata = {
            "rows": np.array([c[0] for c in FREE_CELLS], dtype=np.int32),
            "cols": np.array([c[1] for c in FREE_CELLS], dtype=np.int32),
        }
        return inputs, metadata

    def categorical_space(self):
        return RL_HYPERPARAMS
