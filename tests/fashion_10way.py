"""Smoke test: Fashion-MNIST 10-way. Run from repo root:
    python tests/fashion_10way.py
    python tests/fashion_10way.py --hidden-size 512 --depth 3
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks import TASKS
from src.train_supervised import train_network

DEFAULTS = {
    "hidden_size":   256,
    "depth":         2,
    "activation":    "relu",
    "batch_size":    64,
    "optimizer":     "adam",
    "learning_rate": 1e-3,
    "l1_reg":        0.0,
    "l2_reg":        0.0,
    "init_scale":    1.0,
}

p = argparse.ArgumentParser()
p.add_argument("--hidden-size",   type=int,   default=DEFAULTS["hidden_size"])
p.add_argument("--depth",         type=int,   default=DEFAULTS["depth"])
p.add_argument("--activation",    type=str,   default=DEFAULTS["activation"])
p.add_argument("--batch-size",    type=int,   default=DEFAULTS["batch_size"])
p.add_argument("--optimizer",     type=str,   default=DEFAULTS["optimizer"])
p.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
p.add_argument("--l1-reg",        type=float, default=DEFAULTS["l1_reg"])
p.add_argument("--l2-reg",        type=float, default=DEFAULTS["l2_reg"])
p.add_argument("--init-scale",    type=float, default=DEFAULTS["init_scale"])
args = p.parse_args()

config = {
    "hidden_size":   args.hidden_size,
    "depth":         args.depth,
    "activation":    args.activation,
    "batch_size":    args.batch_size,
    "optimizer":     args.optimizer,
    "learning_rate": args.learning_rate,
    "l1_reg":        args.l1_reg,
    "l2_reg":        args.l2_reg,
    "init_scale":    args.init_scale,
}

task = TASKS["fashion_10way"]()
ds_train, ds_val = task.get_data(data_dir="data")
rdm_inputs, _    = task.get_rdm_stimuli(data_dir="data")

print(f"train={len(ds_train)}  val={len(ds_val)}  |  Success: val_acc >= {task.success_threshold}")
print(f"Config: {config}\n")

final = train_network(
    task       = task,
    config     = config,
    run_dir    = Path("experiments_test") / "fashion_10way",
    rdm_inputs = rdm_inputs,
    ds_train   = ds_train,
    ds_val     = ds_val,
    verbose    = True,
)
print(f"\nBest val_acc: {final:.4f}  ({'SOLVED' if final >= task.success_threshold else 'FAILED'})")
