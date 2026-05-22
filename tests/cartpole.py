"""Smoke test: CartPole. Run from repo root:
    python tests/cartpole.py
    python tests/cartpole.py --hidden-size 128 --gamma 0.9
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks import TASKS
from src.train_rl import train_network

DEFAULTS = {
    "hidden_size":   64,
    "depth":         2,
    "activation":    "relu",
    "optimizer":     "adam",
    "learning_rate": 1e-3,
    "l1_reg":        0.0,
    "l2_reg":        0.0,
    "init_scale":    1.0,
    "gamma":         0.99,
}

p = argparse.ArgumentParser()
p.add_argument("--hidden-size",   type=int,   default=DEFAULTS["hidden_size"])
p.add_argument("--depth",         type=int,   default=DEFAULTS["depth"])
p.add_argument("--activation",    type=str,   default=DEFAULTS["activation"])
p.add_argument("--optimizer",     type=str,   default=DEFAULTS["optimizer"])
p.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
p.add_argument("--l1-reg",        type=float, default=DEFAULTS["l1_reg"])
p.add_argument("--l2-reg",        type=float, default=DEFAULTS["l2_reg"])
p.add_argument("--init-scale",    type=float, default=DEFAULTS["init_scale"])
p.add_argument("--gamma",         type=float, default=DEFAULTS["gamma"])
args = p.parse_args()

config = {
    "hidden_size":   args.hidden_size,
    "depth":         args.depth,
    "activation":    args.activation,
    "optimizer":     args.optimizer,
    "learning_rate": args.learning_rate,
    "l1_reg":        args.l1_reg,
    "l2_reg":        args.l2_reg,
    "init_scale":    args.init_scale,
    "gamma":         args.gamma,
}

task          = TASKS["cartpole"]()
env_factory   = task.get_data()
rdm_inputs, _ = task.get_rdm_stimuli()

print(f"Max steps: {task.max_steps:,}  |  Success: mean_return >= {task.success_threshold}")
print(f"Config: {config}\n")

final = train_network(
    task        = task,
    config      = config,
    run_dir     = Path("experiments_test") / "cartpole",
    rdm_inputs  = rdm_inputs,
    env_factory = env_factory,
    verbose     = True,
)
print(f"\nFinal mean_return: {final:.2f}  ({'SOLVED' if final >= task.success_threshold else 'FAILED'})")
