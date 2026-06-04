import json
import math
import time
from pathlib import Path

import torch

from .dataset import make_loader
from .model_rnn import RNNModel
from .rdm import save_activations_rnn
from .utils import MIN_EPOCHS, MAX_EPOCHS, EARLY_STOP_PATIENCE, EARLY_STOP_THRESHOLD, log4_checkpoints, epoch_checkpoints, format_epoch_label, perf_checkpoint_thresholds, make_optimizer, l1_penalty


def _evaluate(model, loader, criterion, device, multiclass=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            if multiclass:
                loss    = criterion(logits, y)
                correct += (logits.argmax(dim=1) == y).sum().item()
            else:
                logits  = logits.squeeze(1)
                loss    = criterion(logits, y)
                correct += ((logits > 0).float() == y).sum().item()
            total_loss += loss.item() * len(y)
            total      += len(y)
    model.train()
    return total_loss / total, correct / total


def train_network(task, config, run_dir, rdm_inputs, ds_train=None, ds_val=None,
                  ds_test=None, device=None, max_epochs_override=None, verbose=False,
                  save_activations=True):
    """Train one RNN. Returns best metric value (float)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ds_train is None or ds_val is None:
        ds_train, ds_val = task.get_data(data_dir="data")

    max_epochs = max_epochs_override if max_epochs_override is not None else (task.max_epochs or MAX_EPOCHS)

    batch_size   = config["batch_size"]
    train_loader = make_loader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader   = make_loader(ds_val,   batch_size=256,        shuffle=False)

    multiclass = task.output_size > 1
    use_mse    = task.metric_name == "val_mse"

    model = RNNModel(
        input_size   = task.input_size,
        output_size  = task.output_size,
        hidden_size  = int(config["hidden_size"]),
        cell_type    = config["cell_type"],
        n_rnn_layers = int(config["n_rnn_layers"]),
        init_scale   = float(config["init_scale"]),
    ).to(device)

    optimizer    = make_optimizer(model, config)
    criterion    = task.make_loss()
    l1_coef      = config["l1_reg"]
    rdm_tensor   = torch.tensor(rdm_inputs, dtype=torch.float32)
    time_indices = task.rdm_time_indices

    steps_per_epoch   = math.ceil(len(ds_train) / batch_size)
    total_steps       = max_epochs * steps_per_epoch
    checkpoint_steps  = set(log4_checkpoints(total_steps))
    epoch_ckpt_steps  = epoch_checkpoints(steps_per_epoch, max_epochs)
    perf_ckpts        = perf_checkpoint_thresholds(task.chance_perf, task.max_metric)
    perf_crossed      = set()

    history = []

    global_step       = 0
    best_val_loss     = float("inf")    # for early stopping
    no_improve        = 0
    # Best model tracking: val_acc (higher better) or val_mse (lower better)
    best_model_metric = float("inf") if use_mse else -1.0
    best_epoch        = 0
    best_step         = 0
    final_epoch       = 0
    final_metric      = 0.0
    t0                = time.time()

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        t_epoch = time.time()

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
                save_activations_rnn(model, rdm_tensor,
                                     run_dir / f"step_{global_step:07d}",
                                     device, time_indices=time_indices)
            if save_activations and global_step in epoch_ckpt_steps:
                label = format_epoch_label(epoch_ckpt_steps[global_step])
                save_activations_rnn(model, rdm_tensor,
                                     run_dir / f"epoch_{label}",
                                     device, time_indices=time_indices)

        epoch_loss /= len(ds_train)
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device, multiclass)

        if verbose:
            elapsed     = time.time() - t0
            epoch_time  = time.time() - t_epoch
            metric_str  = f"val_mse={val_loss:.4f}" if use_mse else f"val_acc={val_acc:.4f}"
            print(f"  epoch {epoch+1:3d}  {metric_str}  [{epoch_time:.1f}s/epoch  {elapsed:.0f}s total]",
                  flush=True)

        record = {
            "epoch":      epoch + 1,
            "step":       global_step,
            "train_loss": round(float(epoch_loss), 6),
            "val_loss":   round(float(val_loss),   6),
        }
        if not use_mse:
            record["val_acc"] = round(float(val_acc), 6)
        history.append(record)

        # Best model tracking
        improved = (val_loss < best_model_metric) if use_mse else (val_acc > best_model_metric)
        if improved:
            best_model_metric = val_loss if use_mse else val_acc
            best_epoch        = epoch + 1
            best_step         = global_step
            torch.save(model.state_dict(), run_dir / "model_best.pt")
            if save_activations:
                perf_val = -best_model_metric if use_mse else best_model_metric
                for raw_t, label in perf_ckpts:
                    if label not in perf_crossed and perf_val >= raw_t:
                        save_activations_rnn(model, rdm_tensor, run_dir / f"perf_{label}",
                                             device, time_indices=time_indices)
                        perf_crossed.add(label)

        # Early stopping (by val_loss): track from epoch 1, stop only after MIN_EPOCHS
        if val_loss < best_val_loss * (1 - EARLY_STOP_THRESHOLD):
            best_val_loss = val_loss
            no_improve    = 0
        else:
            no_improve += 1
        if epoch + 1 >= MIN_EPOCHS and no_improve >= EARLY_STOP_PATIENCE:
            break

    final_epoch  = epoch + 1
    final_metric = val_loss if use_mse else val_acc

    if save_activations:
        save_activations_rnn(model, rdm_tensor, run_dir / "final",
                             device, time_indices=time_indices)
        model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=device))
        save_activations_rnn(model, rdm_tensor, run_dir / "best",
                             device, time_indices=time_indices)

    # Test-set evaluation using best weights (optional)
    test_metric = None
    if ds_test is not None:
        test_loader = make_loader(ds_test, batch_size=256, shuffle=False)
        test_loss, test_acc = _evaluate(model, test_loader, criterion, device, multiclass)
        test_metric = round(float(test_loss if use_mse else test_acc), 6)

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

    # MSE is negated so the BO can always maximise: higher = better across all tasks.
    # The adding task's chance_perf and max_metric are defined accordingly (both negative/zero).
    return -float(best_model_metric) if use_mse else float(best_model_metric)
