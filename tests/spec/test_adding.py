"""Spec checks for the adding task: sequence shape, value ranges, label correctness."""
import numpy as np

from tasks import TASKS


def test_data_shape_and_value_ranges():
    task = TASKS["adding"]()
    ds_train, ds_val = task.get_data(data_dir="data")
    x, y = ds_train.tensors
    assert x.shape[1:] == (25, 2)

    values, flags = x[:, :, 0].numpy(), x[:, :, 1].numpy()
    assert values.min() >= 0 and values.max() <= 1
    assert set(np.unique(flags)) <= {0.0, 1.0}
    assert (flags.sum(axis=1) == 2).all()


def test_label_matches_independent_recomputation():
    task = TASKS["adding"]()
    ds_train, _ = task.get_data(data_dir="data")
    x, y = ds_train.tensors
    expected = (x[:, :, 0] * x[:, :, 1]).sum(axis=1)
    np.testing.assert_allclose(y.numpy(), expected.numpy(), atol=1e-6)


def test_chance_perf_matches_theory():
    # Var[v1+v2] for v1,v2 ~ U(0,1) i.i.d. = 1/6; chance_perf is the negated MSE of
    # the naive "always predict 1.0" predictor, i.e. -Var[v1+v2].
    rng = np.random.default_rng(0)
    v = rng.uniform(0, 1, (200_000, 2)).sum(axis=1)
    assert abs(-v.var() - TASKS["adding"]().chance_perf) < 0.005


def test_rdm_stimuli_shape():
    task = TASKS["adding"]()
    x, meta = task.get_rdm_stimuli(data_dir="data")
    assert x.shape == (100, 25, 2)
    assert meta["targets"].shape == (100,)
