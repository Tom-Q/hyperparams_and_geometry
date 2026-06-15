"""Spec checks for fashion_10way: flat pixel shape/value ranges, label validity,
train/val split sizes, RDM stimuli structure."""
import numpy as np

from tasks import TASKS


def test_data_shape_and_value_ranges():
    task = TASKS["fashion_10way"]()
    ds_train, ds_val = task.get_data(data_dir="data")
    x, y = ds_train.tensors
    assert x.shape[1:] == (784,)
    assert x.min() >= 0 and x.max() <= 1
    assert set(y.unique().tolist()) == set(range(10))


def test_train_val_split_sizes():
    task = TASKS["fashion_10way"]()
    ds_train, ds_val = task.get_data(data_dir="data")
    # 87.5/12.5 stratified split of the 60k FashionMNIST training images
    assert len(ds_train) == 52500
    assert len(ds_val) == 7500


def test_rdm_stimuli_balanced_by_class():
    task = TASKS["fashion_10way"]()
    x, meta = task.get_rdm_stimuli(data_dir="data")
    assert x.shape == (100, 784)
    classes = meta["classes"]
    assert classes.shape == (100,)
    assert (np.bincount(classes, minlength=10) == 10).all()
