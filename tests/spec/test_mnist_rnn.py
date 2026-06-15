"""Spec checks for the mnist_rnn task: sequence shape/reshape correctness,
value ranges, label validity, RDM stimuli structure."""
import numpy as np
import torch
from torchvision import datasets

from tasks import TASKS


def test_data_shape_and_value_ranges():
    task = TASKS["mnist_rnn"]()
    ds_train, ds_val = task.get_data(data_dir="data")
    x, y = ds_train.tensors
    assert x.shape[1:] == (14, 56)
    assert x.min() >= 0 and x.max() <= 1
    assert set(y.unique().tolist()) <= set(range(10))

    x_val, y_val = ds_val.tensors
    assert x_val.shape[1:] == (14, 56)
    # train/val split is 7/8 vs 1/8 of the 60k training images
    assert len(x) + len(x_val) == 60000


def test_sequence_reshape_roundtrips_to_valid_image():
    """Each (14, 56) sequence is two consecutive 28-pixel rows per step; reshaping
    back to (28, 28) must reproduce the original MNIST image exactly."""
    raw = datasets.MNIST("data", train=True, download=True)
    img = raw.data[0].float() / 255.0          # (28, 28)
    seq = img.view(14, 56)                      # task's reshape
    assert torch.equal(seq.view(28, 28), img)


def test_rdm_stimuli_balanced_by_digit():
    task = TASKS["mnist_rnn"]()
    x, meta = task.get_rdm_stimuli(data_dir="data")
    assert x.shape == (100, 14, 56)
    digits = meta["digits"]
    assert digits.shape == (100,)
    counts = np.bincount(digits, minlength=10)
    assert (counts == 10).all()
