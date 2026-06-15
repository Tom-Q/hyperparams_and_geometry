"""Spec checks for spirals: data shape/class balance, radius range, and
noise-free RDM stimuli lying exactly on the 3 spiral arms with the expected
2*pi/3 angular offset between classes."""
import numpy as np
import torch

from tasks import TASKS


def test_data_shapes_and_balance():
    task = TASKS["spirals"]()
    ds_train, ds_val = task.get_data(data_dir="data")

    x, y = ds_train.tensors
    assert x.shape == (3000, 2)
    assert torch.equal(torch.bincount(y, minlength=3), torch.tensor([1000, 1000, 1000]))

    xv, yv = ds_val.tensors
    assert xv.shape == (600, 2)
    assert torch.equal(torch.bincount(yv, minlength=3), torch.tensor([200, 200, 200]))


def test_radius_in_expected_range():
    task = TASKS["spirals"]()
    ds_train, _ = task.get_data(data_dir="data")
    x, _ = ds_train.tensors
    radius = x.norm(dim=1)
    # noise-free radius (t) is in [0.1, 1.0]; with noise std = 0.1*t, values stay well below 1.5
    assert radius.min() >= 0
    assert radius.max() <= 1.5


def test_rdm_stimuli_lie_on_noise_free_spiral_arms():
    task = TASKS["spirals"]()
    x, meta = task.get_rdm_stimuli(data_dir="data")
    classes = meta["classes"]
    assert x.shape == (198, 2)
    assert (np.bincount(classes, minlength=3) == 66).all()

    radius = np.linalg.norm(x, axis=1)
    angle  = np.arctan2(x[:, 1], x[:, 0])

    # noise-free radius == t, spanning [0.1, 1.0] within each arm
    for c in range(3):
        r_c = np.sort(radius[classes == c])
        np.testing.assert_allclose(r_c, np.linspace(0.1, 1.0, 66), atol=1e-5)

    # angular offset between arm c and arm 0 at matching radius is 2*pi*c/3
    order0 = np.argsort(radius[classes == 0])
    for c in (1, 2):
        orderc = np.argsort(radius[classes == c])
        diff = (angle[classes == c][orderc] - angle[classes == 0][order0]) % (2 * np.pi)
        expected = (2 * np.pi * c / 3) % (2 * np.pi)
        np.testing.assert_allclose(diff, expected, atol=1e-5)


def test_chance_perf_is_uniform_3way():
    assert TASKS["spirals"]().chance_perf == 1 / 3
