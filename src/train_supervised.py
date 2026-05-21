import json
import math
from pathlib import Path

import numpy as np
import torch

from .dataset import make_loader
from .model_mlp import MLP
from .rdm import save_activations_mlp, stimuli_to_tensor
from .utils import MIN_EPOCHS, MAX_EPOCHS, EARLY_STOP_PATIENCE, log4_checkpoints, make_optimizer, l1_penalty


def _evaluate(model, loader, criterion, device, multiclass=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if multiclass:
                loss = criterion(logits, y)
                correct += (logits.argmax(dim=1) == y).sum().item()
            else:
                logits = logits.squeeze(1)
                loss = criterion(logits, y)
                correct += ((logits > 0).float() == y).sum().item()
            total_loss += loss.item() * len(y)
            total      += len(y)
    model.train()
    return total_loss / total, correct / total


def train_network(task, config, run_dir, rdm_inputs, ds_train=None, ds_val=None,
                  device=None, max_epochs_override=None):
    """Train one MLP network for the given task. Returns metric value (float)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ds_train is None or ds_val is None:
        ds_train, ds_val = task.get_data(data_dir="data")

    max_epochs = max_epochs_override if max_epochs_override is not None else MAX_EPOCHS

    batch_size   = config["batch_size"]
    train_loader = make_loader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader   = make_loader(ds_val,   batch_size=512,        shuffle=False)

    multiclass = task.output_size > 1
    model = MLP(
        input_size  = task.input_size,
        output_size = task.output_size,
        hidden_size = int(config["hidden_size"]),
        depth       = int(config["depth"]),
        activation  = config["activation"],
        init_scale  = float(config["init_scale"]),
    ).to(device)

    optimizer  = make_optimizer(model, config)
    criterion  = task.make_loss()
    l1_coef    = config["l1_reg"]
    stimuli_t  = stimuli_to_tensor(rdm_inputs)

    steps_per_epoch  = math.ceil(len(ds_train) / batch_size)
    total_steps      = max_epochs * steps_per_epoch
    checkpoint_steps = set(log4_checkpoints(total_steps))

    curve_steps, curve_train_loss, curve_val_loss, curve_val_acc = [], [], [], []

    global_step   = 0
    best_val_loss = float("inf")
    no_improve    = 0
    final_metric  = 0.0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0

        for x, y in train_loader:
            global_step += 1
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            if multiclass:
                loss = criterion(logits, y) + l1_penalty(model, l1_coef)
            else:
                loss = criterion(logits.squeeze(1), y) + l1_penalty(model, l1_coef)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y)

            if global_step in checkpoint_steps:
                save_activations_mlp(model, stimuli_t, global_step, run_dir, device)

        epoch_loss   /= len(ds_train)
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device, multiclass)
        final_metric = val_acc

        curve_steps.append(global_step)
        curve_train_loss.append(epoch_loss)
        curve_val_loss.append(val_loss)
        curve_val_acc.append(val_acc)

        if epoch >= MIN_EPOCHS - 1:
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP_PATIENCE:
                    break
        elif val_loss < best_val_loss:
            best_val_loss = val_loss

    np.savez(
        run_dir / "training_curves.npz",
        steps      = np.array(curve_steps),
        train_loss = np.array(curve_train_loss),
        val_loss   = np.array(curve_val_loss),
        val_acc    = np.array(curve_val_acc),
    )

    saved_config = dict(config)
    saved_config["effective_depth"] = model.effective_depth
    with open(run_dir / "config.json", "w") as f:
        json.dump(saved_config, f, indent=2)

    return float(final_metric)
