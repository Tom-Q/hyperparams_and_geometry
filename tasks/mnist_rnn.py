"""Task 9: MNIST row-by-row (14 steps × 56-dim = 2 rows per step)."""
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchvision import datasets
from sklearn.model_selection import train_test_split

from .base import Task
from ._shared import RNN_HYPERPARAMS

N_STEPS    = 14   # two rows per step
INPUT_SIZE = 56   # two rows × 28 pixels


def _mnist_to_sequences(data_dir):
    ds_train = datasets.MNIST(data_dir, train=True,  download=True)
    ds_test  = datasets.MNIST(data_dir, train=False, download=True)
    train_x = ds_train.data.float().view(-1, N_STEPS, INPUT_SIZE) / 255.0
    train_y = ds_train.targets.long()
    test_x  = ds_test.data.float().view(-1, N_STEPS, INPUT_SIZE) / 255.0
    test_y  = ds_test.targets.long()
    return train_x, train_y, test_x, test_y


class MNISTRNNTask(Task):
    name              = "mnist_rnn"
    paradigm          = "rnn"
    input_size        = INPUT_SIZE
    output_size       = 10
    n_steps           = N_STEPS
    success_threshold = 0.90
    chance_perf       = 0.1      # 10-way classification
    metric_name       = "val_acc"

    def get_data(self, data_dir="data", seed=42):
        train_x, train_y, _, _ = _mnist_to_sequences(data_dir)
        idx = np.arange(len(train_y))
        idx_train, idx_val = train_test_split(
            idx, test_size=0.125, stratify=train_y.numpy(), random_state=seed
        )
        ds_train = TensorDataset(train_x[idx_train], train_y[idx_train])
        ds_val   = TensorDataset(train_x[idx_val],   train_y[idx_val])
        return ds_train, ds_val

    def get_rdm_stimuli(self, data_dir="data", seed=42):
        """100 stimuli: 10 exemplars × 10 digits, each as 28-step sequence."""
        _, _, test_x, test_y = _mnist_to_sequences(data_dir)
        rng = np.random.default_rng(seed)
        inputs_list, digit_list = [], []
        for d in range(10):
            idx = np.where(test_y.numpy() == d)[0]
            chosen = rng.choice(idx, size=10, replace=False)
            for i in chosen:
                inputs_list.append(test_x[i].numpy())
                digit_list.append(d)
        inputs   = np.stack(inputs_list).astype(np.float32)   # (100, 14, 56)
        metadata = {"digits": np.array(digit_list, dtype=np.int32)}
        return inputs, metadata

    def categorical_space(self):
        return RNN_HYPERPARAMS

    def make_loss(self):
        return nn.CrossEntropyLoss()
