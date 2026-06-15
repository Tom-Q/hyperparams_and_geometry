"""Shared helpers for smoke tests: verify RDM activation files saved during training."""
import numpy as np


def check_rnn_activations(npz_path, n_layers, n_steps, n_stimuli, hidden_size):
    """Verify an RNN activation .npz: keys layer_{l}_t_{t} for l in range(n_layers),
    t in range(n_steps), each array (n_stimuli, hidden_size), finite, non-constant.
    """
    data = np.load(npz_path)
    expected_keys = {f"layer_{l}_t_{t}" for l in range(n_layers) for t in range(n_steps)}
    actual_keys = set(data.files)
    assert actual_keys == expected_keys, (
        f"{npz_path}: key mismatch.\n"
        f"  missing={expected_keys - actual_keys}\n"
        f"  extra={actual_keys - expected_keys}"
    )
    for k in expected_keys:
        arr = data[k]
        assert arr.shape == (n_stimuli, hidden_size), f"{npz_path}:{k} shape={arr.shape}"
        assert np.isfinite(arr).all(), f"{npz_path}:{k} contains NaN/Inf"
        assert arr.std() > 0, f"{npz_path}:{k} has zero spread (all values identical)"


def mlp_layer_sizes(hidden_size, depth):
    """Hidden layer sizes for the MLP: H, H//2, ... (depth entries), matching model_mlp.MLP."""
    sizes, h = [], hidden_size
    for _ in range(depth):
        sizes.append(max(1, h))
        h //= 2
    return sizes


def check_mlp_activations(npz_path, layer_sizes, n_stimuli):
    """Verify an MLP activation .npz: keys layer_0..layer_{len(layer_sizes)-1},
    each array (n_stimuli, layer_sizes[i]), finite, non-constant.
    """
    data = np.load(npz_path)
    expected_keys = {f"layer_{i}" for i in range(len(layer_sizes))}
    actual_keys = set(data.files)
    assert actual_keys == expected_keys, (
        f"{npz_path}: key mismatch.\n"
        f"  missing={expected_keys - actual_keys}\n"
        f"  extra={actual_keys - expected_keys}"
    )
    for i, size in enumerate(layer_sizes):
        arr = data[f"layer_{i}"]
        assert arr.shape == (n_stimuli, size), f"{npz_path}:layer_{i} shape={arr.shape}"
        assert np.isfinite(arr).all(), f"{npz_path}:layer_{i} contains NaN/Inf"
        assert arr.std() > 0, f"{npz_path}:layer_{i} has zero spread (all values identical)"
