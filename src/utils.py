"""Shared training utilities used by all training loops."""
import torch

MIN_EPOCHS          = 15
MAX_EPOCHS          = 100
EARLY_STOP_PATIENCE = 10


def log4_checkpoints(total_steps):
    """Powers of 4: 1, 4, 16, 64, … up to total_steps, plus total_steps."""
    steps, s = {1}, 4
    while s <= total_steps:
        steps.add(s)
        s *= 4
    steps.add(total_steps)
    return sorted(steps)


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
