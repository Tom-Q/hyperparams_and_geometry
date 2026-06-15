"""Spec checks for mnist_dual: input format (pixels + task bit), label
correctness for both task bits, dataset doubling, RDM stimuli structure."""
import numpy as np
import torch

from tasks import TASKS


def test_data_shape_and_value_ranges():
    task = TASKS["mnist_dual"]()
    ds_train, ds_val = task.get_data(data_dir="data")

    x, y = ds_train[0]
    assert x.shape == (785,)
    assert x[:784].min() >= 0 and x[:784].max() <= 1
    assert x[784].item() in (0.0, 1.0)
    assert y.item() in (0.0, 1.0)


def test_dataset_is_doubled_for_both_task_bits():
    task = TASKS["mnist_dual"]()
    ds_train, ds_val = task.get_data(data_dir="data")
    # 80% of 60000 MNIST training images, doubled (task_bit=0 and task_bit=1 copies)
    assert len(ds_train) == 2 * 48000
    assert (ds_train.task_bits == 0).sum() == 48000
    assert (ds_train.task_bits == 1).sum() == 48000


def test_label_matches_independent_recomputation():
    task = TASKS["mnist_dual"]()
    ds_train, ds_val = task.get_data(data_dir="data")
    digits, task_bits, labels = ds_train.digits, ds_train.task_bits, ds_train.labels

    expected = torch.where(
        task_bits == 0,
        (digits % 2 == 0).float(),   # task 0: even=1, odd=0
        (digits < 5).float(),        # task 1: <5=1, >=5=0
    )
    assert torch.equal(labels, expected)


def test_rdm_stimuli_balanced():
    task = TASKS["mnist_dual"]()
    inputs, meta = task.get_rdm_stimuli(data_dir="data")
    assert inputs.shape == (200, 785)
    assert set(np.unique(inputs[:, 784])) <= {0.0, 1.0}

    digits, tasks = meta["digits"], meta["tasks"]
    assert digits.shape == (200,) and tasks.shape == (200,)
    # 10 exemplars x 10 digits x 2 task bits
    assert (np.bincount(digits, minlength=10) == 20).all()
    assert (np.bincount(tasks, minlength=2) == 100).all()
