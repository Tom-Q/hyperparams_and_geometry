"""Spec checks for parity: full pattern enumeration, label correctness
(popcount parity), and stratified RDM stimuli."""
import numpy as np

from tasks import TASKS
from tasks.parity import N_BITS, PER_LEVEL


def test_data_is_all_patterns_with_correct_labels():
    task = TASKS["parity"]()
    ds_train, ds_val = task.get_data(data_dir="data")
    assert ds_train is ds_val  # deterministic task: train == val
    x, y = ds_train.tensors
    assert x.shape == (2 ** N_BITS, N_BITS)
    assert set(np.unique(x.numpy())) <= {0.0, 1.0}

    # every pattern appears exactly once
    rows = [tuple(row.tolist()) for row in x]
    assert len(set(rows)) == 2 ** N_BITS

    expected = (x.sum(axis=1) % 2)
    np.testing.assert_allclose(y.numpy(), expected.numpy(), atol=1e-6)


def test_rdm_stimuli_stratified_by_n_ones():
    task = TASKS["parity"]()
    x, meta = task.get_rdm_stimuli(data_dir="data")
    n_ones, labels = meta["n_ones"], meta["labels"]

    assert x.shape[0] == n_ones.shape[0] == labels.shape[0]
    np.testing.assert_array_equal(x.sum(axis=1), n_ones)
    np.testing.assert_array_equal(labels, n_ones % 2)

    counts = np.bincount(n_ones, minlength=N_BITS + 1)
    from math import comb
    expected_counts = [min(PER_LEVEL, comb(N_BITS, k)) for k in range(N_BITS + 1)]
    assert counts.tolist() == expected_counts
