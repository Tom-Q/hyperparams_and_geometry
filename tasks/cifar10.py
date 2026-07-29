"""Task: CIFAR-10 10-way classification with CNN+FC architecture."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms as T
from sklearn.model_selection import train_test_split

from .base import Task


_TRAIN_TRANSFORM = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
])


class _AugDataset(torch.utils.data.Dataset):
    """Wraps tensor (x, y) pairs and applies a transform to x on each __getitem__."""
    def __init__(self, x, y, transform):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.transform(self.x[idx]), self.y[idx]


def _load_cifar10(data_dir):
    """Load CIFAR-10, returning float32 (N, 3, 32, 32) tensors in [0, 1]."""
    ds_train = datasets.CIFAR10(data_dir, train=True,  download=True)
    ds_test  = datasets.CIFAR10(data_dir, train=False, download=True)

    # ds.data is (N, 32, 32, 3) uint8 numpy; targets is a Python list
    train_x = torch.from_numpy(
        ds_train.data.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    )
    train_y = torch.tensor(ds_train.targets, dtype=torch.long)
    test_x  = torch.from_numpy(
        ds_test.data.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    )
    test_y  = torch.tensor(ds_test.targets, dtype=torch.long)
    return train_x, train_y, test_x, test_y


class Cifar10Task(Task):
    name              = "cifar10"
    paradigm          = "supervised"
    input_size        = (3, 32, 32)   # informational; train_supervised uses build_model
    output_size       = 10
    n_steps           = None
    success_threshold = 0.60          # tentative; revisit after first results
    chance_perf       = 0.1           # 10-way classification
    metric_name       = "val_acc"

    def get_data(self, data_dir="data", seed=42):
        train_x, train_y, _, _ = _load_cifar10(data_dir)
        idx = np.arange(len(train_y))
        idx_train, idx_val = train_test_split(
            idx, test_size=0.125, stratify=train_y.numpy(), random_state=seed
        )
        ds_train = _AugDataset(train_x[idx_train], train_y[idx_train], _TRAIN_TRANSFORM)
        ds_val   = TensorDataset(train_x[idx_val],   train_y[idx_val])
        return ds_train, ds_val

    def get_rdm_stimuli(self, data_dir="data", seed=42):
        """100 stimuli: 10 exemplars × 10 CIFAR-10 classes from the test set."""
        _, _, test_x, test_y = _load_cifar10(data_dir)
        rng = np.random.default_rng(seed)
        inputs_list, class_list = [], []
        for c in range(10):
            idx = np.where(test_y.numpy() == c)[0]
            chosen = rng.choice(idx, size=10, replace=False)
            for i in chosen:
                inputs_list.append(test_x[i].numpy())
                class_list.append(c)
        inputs   = np.stack(inputs_list).astype(np.float32)   # (100, 3, 32, 32)
        metadata = {"classes": np.array(class_list, dtype=np.int32)}
        return inputs, metadata

    def categorical_space(self):
        return {
            "n_fc_layers":    [1, 2, 3],
            "hidden_size":    [64, 512],    # CONT_FROM_CAT → continuous [64, 512]
            "batch_size":     [16, 128],    # CONT_FROM_CAT → continuous [16, 128]
            "activation":     ["relu", "tanh"],
            "use_batchnorm":  [False, True],
        }

    def cont_param_ranges(self):
        return [
            ("learning_rate", 1e-4, 1e-2),
            ("l2_reg",        1e-6, 1e-3),
        ]

    def build_model(self, config):
        from src.model_cnn_mlp import CNNMLP
        return CNNMLP(
            output_size   = self.output_size,
            hidden_size   = int(config["hidden_size"]),
            n_fc_layers   = int(config["n_fc_layers"]),
            activation    = config["activation"],
            use_batchnorm = bool(config["use_batchnorm"]),
        )

    def make_loss(self):
        return nn.CrossEntropyLoss()
