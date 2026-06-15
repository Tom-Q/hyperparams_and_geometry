"""Generalised RDM activation saving for MLP and RNN models."""
import numpy as np
import torch


def stimuli_to_tensor(inputs_array):
    """Convert numpy array to float32 tensor."""
    return torch.tensor(inputs_array, dtype=torch.float32)


def save_activations_mlp(model, stimuli_tensor, path, device):
    """Save post-activation hidden-layer outputs to path (npz compressed).

    path : full file path (without .npz suffix — numpy adds it automatically)
    Keys: layer_0, layer_1, … each (N_stimuli, hidden_size)
    """
    model.eval()
    with torch.no_grad():
        acts = model.get_layer_activations(stimuli_tensor.to(device))
    np.savez_compressed(
        path,
        **{k: v.numpy().astype(np.float32) for k, v in acts.items()},
    )
    model.train()


def save_activations_rnn(model, sequences_tensor, path, device):
    """Save per-layer, per-time-step hidden states to path (npz compressed).

    sequences_tensor : (N, T, input_size) float32
    Keys: layer_0_t_0, layer_0_t_1, …, layer_1_t_0, … each (N, H)
    """
    model.eval()
    with torch.no_grad():
        layer_acts = model.get_step_activations(sequences_tensor.to(device))  # list of (N, T, H)
    payload = {
        f"layer_{l}_t_{t}": acts[:, t, :].numpy().astype(np.float32)
        for l, acts in enumerate(layer_acts)
        for t in range(acts.shape[1])
    }
    np.savez_compressed(path, **payload)
    model.train()
