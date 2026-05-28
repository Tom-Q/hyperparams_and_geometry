"""Task 10: Adding problem (T=50, 2-dim per step, scalar output, MSE)."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from .base import Task
from ._shared import RNN_CATS

T           = 50
N_TRAIN     = 10_000
N_VAL       = 1_000
N_STIMULI   = 100
STIM_SEED   = 200   # fixed seed distinct from get_data seeds
# Save hidden state at these time steps only (to limit storage for T=50)
RDM_TIME_INDICES = [0, 4, 9, 19, 34, 49]   # 0-indexed: steps 1, 5, 10, 20, 35, 50


def _generate_adding(n, seed):
    """Generate n adding-problem sequences.
    Each step: (value in [0,1], flag ∈ {0,1}).
    Exactly 2 flags are set; target = sum of flagged values.
    Returns x: (n, T, 2), y: (n,) float32.
    """
    rng = np.random.default_rng(seed)
    values = rng.uniform(0, 1, (n, T)).astype(np.float32)
    flags  = np.zeros((n, T), dtype=np.float32)
    for i in range(n):
        pos = rng.choice(T, size=2, replace=False)
        flags[i, pos] = 1.0
    x = np.stack([values, flags], axis=2)   # (n, T, 2)
    y = (values * flags).sum(axis=1)        # (n,) in [0, 2]
    return x, y


class AddingTask(Task):
    name              = "adding"
    paradigm          = "rnn"
    input_size        = 2
    output_size       = 1
    n_steps           = T
    success_threshold = 0.02   # MSE < 0.02 counts as a successful network
    chance_accuracy   = 0.0    # regression; no meaningful chance baseline
    metric_name       = "val_mse"
    rdm_time_indices  = RDM_TIME_INDICES

    def get_data(self, data_dir="data", seed=42):
        x_train, y_train = _generate_adding(N_TRAIN, seed=seed)
        x_val,   y_val   = _generate_adding(N_VAL,   seed=seed + 1)
        ds_train = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
        ds_val   = TensorDataset(torch.tensor(x_val),   torch.tensor(y_val))
        return ds_train, ds_val

    def get_rdm_stimuli(self, data_dir="data", seed=42):
        """100 fixed sequences; hidden state saved at a subset of time steps."""
        x, y = _generate_adding(N_STIMULI, seed=STIM_SEED)
        return x, {"targets": y}   # shape: (100, 50, 2)

    def categorical_space(self):
        return RNN_CATS

    def make_loss(self):
        return nn.MSELoss()
