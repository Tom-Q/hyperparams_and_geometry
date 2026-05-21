"""Task 1: MNIST dual-task (even/odd + <5 via task bit)."""
import numpy as np
import torch.nn as nn

from .base import Task
from ._shared import SUPERVISED_CATS


class MNISTDualTask(Task):
    name              = "mnist_dual"
    paradigm          = "supervised"
    input_size        = 785   # 784 pixels + 1 task bit
    output_size       = 1
    n_steps           = None
    hidden_size_range = (4, 1024)
    success_threshold = 0.90
    metric_name       = "val_acc"

    def get_data(self, data_dir="data", seed=42):
        from src.dataset import load_mnist_splits
        ds_train, ds_val, _, _ = load_mnist_splits(data_dir=data_dir, seed=seed)
        return ds_train, ds_val

    def get_rdm_stimuli(self, data_dir="data", seed=42):
        """200 stimuli: 10 exemplars × 10 digits × 2 tasks."""
        from src.dataset import load_mnist_splits
        _, _, _, ds_test = load_mnist_splits(data_dir=data_dir, seed=seed)

        rng = np.random.default_rng(seed)
        images    = ds_test.images.numpy()        # (N*2, 784)
        task_bits = ds_test.task_bits.numpy()     # (N*2,)
        digits    = ds_test.digits.numpy()        # (N*2,)

        # Only use task_bit=0 exemplars; both task bits are appended per exemplar below
        mask0 = task_bits == 0
        imgs0 = images[mask0]
        digs0 = digits[mask0]

        inputs_list, digit_list, task_list = [], [], []
        for d in range(10):
            idx = np.where(digs0 == d)[0]
            chosen = rng.choice(idx, size=10, replace=False)
            for i in chosen:
                for tb in [0, 1]:
                    x = np.concatenate([imgs0[i], [float(tb)]])
                    inputs_list.append(x)
                    digit_list.append(d)
                    task_list.append(tb)

        inputs = np.stack(inputs_list, axis=0).astype(np.float32)
        metadata = {
            "digits": np.array(digit_list, dtype=np.int32),
            "tasks":  np.array(task_list,  dtype=np.int32),
        }
        return inputs, metadata

    def categorical_space(self):
        return SUPERVISED_CATS

    def make_loss(self):
        return nn.BCEWithLogitsLoss()
