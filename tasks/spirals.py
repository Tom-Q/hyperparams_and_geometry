"""Task 4: 2D spirals (3-class)."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from .base import Task
from ._shared import SUPERVISED_CATS

N_TRAIN        = 3000   # 1000 per arm
N_VAL          = 600
N_PER_ARM      = 66     # 3 × 66 = 198 stimuli (evenly balanced across arms)


def _generate_spirals(n_per_class, noise=0.1, seed=0):
    """3-arm Archimedean spirals in 2D. Returns x: (N,2), y: (N,) int64."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(3):
        t = np.linspace(0.1, 1.0, n_per_class)
        angle = t * 4 * np.pi + (2 * np.pi * c / 3)
        x1 = t * np.cos(angle) + rng.normal(0, noise * t, n_per_class)
        x2 = t * np.sin(angle) + rng.normal(0, noise * t, n_per_class)
        xs.append(np.stack([x1, x2], axis=1))
        ys.append(np.full(n_per_class, c, dtype=np.int64))
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def _spiral_rdm_stimuli():
    """198 stimuli: 66 evenly spaced points along each of 3 spiral arms (no noise)."""
    inputs_list, class_list = [], []
    for c in range(3):
        t = np.linspace(0.1, 1.0, N_PER_ARM)
        angle = t * 4 * np.pi + (2 * np.pi * c / 3)
        x1 = t * np.cos(angle)
        x2 = t * np.sin(angle)
        inputs_list.append(np.stack([x1, x2], axis=1))
        class_list.extend([c] * N_PER_ARM)
    inputs  = np.concatenate(inputs_list, axis=0).astype(np.float32)
    classes = np.array(class_list, dtype=np.int32)
    return inputs, classes


class SpiralsTask(Task):
    name              = "spirals"
    paradigm          = "supervised"
    input_size        = 2
    output_size       = 3
    n_steps           = None
    success_threshold = 0.85
    metric_name       = "val_acc"
    max_epochs        = 300

    def get_data(self, data_dir="data", seed=42):
        x_train, y_train = _generate_spirals(N_TRAIN // 3, seed=seed)
        x_val,   y_val   = _generate_spirals(N_VAL   // 3, seed=seed + 1)
        ds_train = TensorDataset(
            torch.tensor(x_train), torch.tensor(y_train, dtype=torch.long)
        )
        ds_val = TensorDataset(
            torch.tensor(x_val), torch.tensor(y_val, dtype=torch.long)
        )
        return ds_train, ds_val

    def get_rdm_stimuli(self, data_dir="data", seed=42):
        inputs, classes = _spiral_rdm_stimuli()
        return inputs, {"classes": classes}

    def categorical_space(self):
        return SUPERVISED_CATS

    def make_loss(self):
        return nn.CrossEntropyLoss()
