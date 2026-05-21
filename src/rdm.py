"""Generalised RDM activation saving for MLP and RNN models."""
from pathlib import Path

import numpy as np
import torch


def stimuli_to_tensor(inputs_array):
    """Convert numpy array to float32 tensor."""
    return torch.tensor(inputs_array, dtype=torch.float32)


def save_activations_mlp(model, stimuli_tensor, step, run_dir, device):
    """Save post-activation hidden-layer outputs for a fixed MLP stimulus set."""
    model.eval()
    act_dir = Path(run_dir) / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        acts = model.get_layer_activations(stimuli_tensor.to(device))
    np.savez(
        act_dir / f"step_{step:07d}.npz",
        **{k: v.numpy().astype(np.float32) for k, v in acts.items()},
    )
    model.train()


def save_activations_rnn(model, sequences_tensor, step, run_dir, device,
                         time_indices=None):
    """Save per-time-step hidden states for a fixed RNN stimulus set.

    sequences_tensor : (N, T, input_size) float32
    time_indices     : list of time steps to save; None = save all
    Saves act_dir/step_{step:07d}.npz with keys "t_{t}" each (N, H).
    """
    model.eval()
    act_dir = Path(run_dir) / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        step_acts = model.get_step_activations(sequences_tensor.to(device))
    # step_acts: list of (N, H) tensors, one per time step
    if time_indices is None:
        time_indices = list(range(len(step_acts)))
    payload = {
        f"t_{t}": step_acts[t].cpu().numpy().astype(np.float32)
        for t in time_indices if t < len(step_acts)
    }
    np.savez(act_dir / f"step_{step:07d}.npz", **payload)
    model.train()
