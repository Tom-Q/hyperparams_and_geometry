import json
import math
from pathlib import Path

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
                  ds_test=None, device=None, max_epochs_override=None, verbose=False,
                  save_activations=True):
    """Train one MLP network for the given task. Returns best val_acc (float)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ds_train is None or ds_val is None:
        ds_train, ds_val = task.get_data(data_dir="data")

    max_epochs = max_epochs_override if max_epochs_override is not None else (task.max_epochs or MAX_EPOCHS)

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

    history = []

    global_step        = 0
    best_val_loss      = float("inf")   # for early stopping
    no_improve         = 0
    best_model_metric  = -1.0           # val_acc; -1 ensures first epoch always saves
    best_epoch         = 0
    best_step          = 0
    final_epoch        = 0
    final_metric       = 0.0

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

            if save_activations and global_step in checkpoint_steps:
                save_activations_mlp(model, stimuli_t,
                                     run_dir / f"step_{global_step:07d}", device)

        epoch_loss /= len(ds_train)
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device, multiclass)

        if verbose:
            print(f"  epoch {epoch+1:3d}  val_acc={val_acc:.4f}  val_loss={val_loss:.4f}", flush=True)

        history.append({
            "epoch":      epoch + 1,
            "step":       global_step,
            "train_loss": round(float(epoch_loss), 6),
            "val_loss":   round(float(val_loss),   6),
            "val_acc":    round(float(val_acc),     6),
        })

        # Best model tracking (by val_acc)
        if val_acc > best_model_metric:
            best_model_metric = val_acc
            best_epoch        = epoch + 1
            best_step         = global_step
            torch.save(model.state_dict(), run_dir / "model_best.pt")

        # Early stopping (by val_loss)
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

    final_epoch  = epoch + 1
    final_metric = val_acc

    if save_activations:
        save_activations_mlp(model, stimuli_t, run_dir / "final", device)
        model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=device))
        save_activations_mlp(model, stimuli_t, run_dir / "best", device)

    # Test-set evaluation using best weights (optional)
    test_metric = None
    if ds_test is not None:
        test_loader = make_loader(ds_test, batch_size=512, shuffle=False)
        _, test_acc = _evaluate(model, test_loader, criterion, device, multiclass)
        test_metric = round(float(test_acc), 6)

    # Persist metadata and history
    metadata = {
        "task":         task.name,
        "paradigm":     task.paradigm,
        "config":       dict(config),
        "best_epoch":   best_epoch,
        "best_step":    best_step,
        "best_metric":  round(float(best_model_metric), 6),
        "final_epoch":  final_epoch,
        "final_step":   global_step,
        "final_metric": round(float(final_metric), 6),
    }
    if test_metric is not None:
        metadata["test_metric"] = test_metric

    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return float(best_model_metric)
