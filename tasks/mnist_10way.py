"""Task 2: MNIST 10-way classification."""
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchvision import datasets
from sklearn.model_selection import train_test_split

from .base import Task
from ._shared import SUPERVISED_CATS


def _load_mnist_flat(data_dir):
    ds_train = datasets.MNIST(data_dir, train=True,  download=True)
    ds_test  = datasets.MNIST(data_dir, train=False, download=True)
    train_x = ds_train.data.float().view(-1, 784) / 255.0
    train_y = ds_train.targets.long()
    test_x  = ds_test.data.float().view(-1, 784) / 255.0
    test_y  = ds_test.targets.long()
    return train_x, train_y, test_x, test_y


class MNIST10WayTask(Task):
    name              = "mnist_10way"
    paradigm          = "supervised"
    input_size        = 784
    output_size       = 10
    n_steps           = None
    success_threshold = 0.90
    metric_name       = "val_acc"

    def get_data(self, data_dir="data", seed=42):
        train_x, train_y, _, _ = _load_mnist_flat(data_dir)
        idx = np.arange(len(train_y))
        idx_train, idx_val = train_test_split(
            idx, test_size=0.125, stratify=train_y.numpy(), random_state=seed
        )
        ds_train = TensorDataset(train_x[idx_train], train_y[idx_train])
        ds_val   = TensorDataset(train_x[idx_val],   train_y[idx_val])
        return ds_train, ds_val

    def get_rdm_stimuli(self, data_dir="data", seed=42):
        """100 stimuli: 10 exemplars × 10 digits."""
        _, _, test_x, test_y = _load_mnist_flat(data_dir)
        rng = np.random.default_rng(seed)
        inputs_list, digit_list = [], []
        for d in range(10):
            idx = np.where(test_y.numpy() == d)[0]
            chosen = rng.choice(idx, size=10, replace=False)
            for i in chosen:
                inputs_list.append(test_x[i].numpy())
                digit_list.append(d)
        inputs   = np.stack(inputs_list).astype(np.float32)
        metadata = {"digits": np.array(digit_list, dtype=np.int32)}
        return inputs, metadata

    def categorical_space(self):
        return SUPERVISED_CATS

    def make_loss(self):
        return nn.CrossEntropyLoss()
