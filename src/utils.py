"""Shared training utilities used by all training loops."""
import torch

MIN_EPOCHS              = 10
MAX_EPOCHS              = 100
EARLY_STOP_PATIENCE     = 5
EARLY_STOP_THRESHOLD    = 0.0001  # relative: val_loss must improve by ≥0.01% to reset patience

EPOCH_CKPT_VALUES = [0.25, 1, 4, 16, 64]
PERF_THRESHOLDS   = [0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95]


def log4_checkpoints(total_steps):
    """Powers of 4: 1, 4, 16, 64, … up to total_steps, plus total_steps."""
    steps, s = {1}, 4
    while s <= total_steps:
        steps.add(s)
        s *= 4
    steps.add(total_steps)
    return sorted(steps)


def epoch_checkpoints(steps_per_epoch, max_epochs):
    """Return dict mapping global_step → epoch_value for epoch checkpoints.
    Only includes EPOCH_CKPT_VALUES that fit within max_epochs.
    """
    result = {}
    for e in EPOCH_CKPT_VALUES:
        if e > max_epochs:
            break
        s = round(e * steps_per_epoch)
        if s >= 1:
            result[s] = e
    return result


def format_epoch_label(e):
    """Epoch value to filesystem-safe string: 0.25 → '0p25', 4 → '4'."""
    return f"{e:g}".replace(".", "p")


def perf_checkpoint_thresholds(chance_perf, max_metric):
    """Return list of (raw_threshold, label) for performance checkpoints.
    raw_threshold is in the task's native metric space (higher = better).
    label is a filesystem-safe string of the normalised threshold value.
    """
    span = max_metric - chance_perf
    return [
        (chance_perf + t * span, f"{t:g}".replace(".", "p"))
        for t in PERF_THRESHOLDS
    ]


def make_optimizer(model, config):
    lr  = config["learning_rate"]
    l2  = config["l2_reg"]
    opt = config["optimizer"].lower()
    if opt == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=l2)
    elif opt == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    raise ValueError(f"Unknown optimizer: {opt}")


def l1_penalty(model, coef):
    if coef == 0:
        return 0.0
    return coef * sum(p.abs().sum() for p in model.parameters() if p.ndim > 1)
