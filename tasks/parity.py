"""Task 5: 12-bit parity (binary classification)."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from .base import Task
from ._shared import SUPERVISED_CATS

N_BITS     = 12
PER_LEVEL  = 20   # levels 0,12 → 1; levels 1,11 → 12; levels 2-10 → 20; total = 206


def _all_parity_patterns():
    """All 2^12 = 4096 patterns with parity label."""
    n = 2 ** N_BITS
    patterns = ((np.arange(n)[:, None] >> np.arange(N_BITS)) & 1).astype(np.float32)
    labels   = (patterns.sum(axis=1) % 2).astype(np.float32)
    return patterns, labels


def _parity_rdm_stimuli(seed=0):
    """Stratified stimuli: up to PER_LEVEL per number-of-ones level, no random top-up."""
    patterns, labels = _all_parity_patterns()
    n_ones = patterns.sum(axis=1).astype(int)
    rng    = np.random.default_rng(seed)
    chosen = []
    for k in range(N_BITS + 1):
        idx = np.where(n_ones == k)[0]
        n   = min(PER_LEVEL, len(idx))
        chosen.extend(rng.choice(idx, size=n, replace=False).tolist())
    chosen   = np.array(chosen)
    inputs   = patterns[chosen]
    metadata = {
        "n_ones": n_ones[chosen].astype(np.int32),
        "labels": labels[chosen].astype(np.int32),
    }
    return inputs, metadata


class ParityTask(Task):
    name              = "parity"
    paradigm          = "supervised"
    input_size        = N_BITS
    output_size       = 1
    n_steps           = None
    hidden_size_range = (4, 256)
    success_threshold = 0.95
    metric_name       = "val_acc"

    def get_data(self, data_dir="data", seed=42):
        patterns, labels = _all_parity_patterns()
        # All 4096 patterns used for both train and val: this is a deterministic
        # function so generalisation = memorisation; early stopping watches train loss.
        ds = TensorDataset(torch.tensor(patterns), torch.tensor(labels))
        return ds, ds

    def get_rdm_stimuli(self, data_dir="data", seed=42):
        return _parity_rdm_stimuli(seed=seed)

    def categorical_space(self):
        return SUPERVISED_CATS

    def make_loss(self):
        return nn.BCEWithLogitsLoss()
