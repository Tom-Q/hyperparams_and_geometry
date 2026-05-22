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


def save_activations_rnn(model, sequences_tensor, path, device, time_indices=None):
    """Save per-time-step hidden states to path (npz compressed).

    sequences_tensor : (N, T, input_size) float32
    time_indices     : list of time steps to save; None = save all
    Keys: t_0, t_5, … each (N, H)
    """
    model.eval()
    with torch.no_grad():
        step_acts = model.get_step_activations(sequences_tensor.to(device))
    if time_indices is None:
        time_indices = list(range(len(step_acts)))
    payload = {
        f"t_{t}": step_acts[t].cpu().numpy().astype(np.float32)
        for t in time_indices if t < len(step_acts)
    }
    np.savez_compressed(path, **payload)
    model.train()
