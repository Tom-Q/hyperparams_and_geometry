"""Smoke test for supervised MLP tasks. Run from repo root:
    python test_supervised.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tasks import TASKS
from src.train_supervised import train_network

# Good configs for each task
CONFIGS = {
    "mnist_dual": {
        "hidden_size":   256,
        "depth":         2,
        "activation":    "relu",
        "batch_size":    64,
        "optimizer":     "adam",
        "learning_rate": 1e-3,
        "l1_reg":        0.0,
        "l2_reg":        0.0,
        "init_scale":    1.0,
    },
    "mnist_10way": {
        "hidden_size":   256,
        "depth":         2,
        "activation":    "relu",
        "batch_size":    64,
        "optimizer":     "adam",
        "learning_rate": 1e-3,
        "l1_reg":        0.0,
        "l2_reg":        0.0,
        "init_scale":    1.0,
    },
    "fashion_10way": {
        "hidden_size":   256,
        "depth":         2,
        "activation":    "relu",
        "batch_size":    64,
        "optimizer":     "adam",
        "learning_rate": 1e-3,
        "l1_reg":        0.0,
        "l2_reg":        0.0,
        "init_scale":    1.0,
    },
    "spirals": {
        "hidden_size":   64,
        "depth":         3,
        "activation":    "relu",
        "batch_size":    64,
        "optimizer":     "adam",
        "learning_rate": 1e-3,
        "l1_reg":        0.0,
        "l2_reg":        0.0,
        "init_scale":    1.0,
    },
    "parity": {
        "hidden_size":   128,
        "depth":         3,
        "activation":    "relu",
        "batch_size":    64,
        "optimizer":     "adam",
        "learning_rate": 1e-3,
        "l1_reg":        0.0,
        "l2_reg":        0.0,
        "init_scale":    1.0,
    },
}


def run(task_name):
    config = CONFIGS[task_name]
    task   = TASKS[task_name]()
    ds_train, ds_val = task.get_data(data_dir="data")
    rdm_inputs, _    = task.get_rdm_stimuli(data_dir="data")
    run_dir          = Path("experiments_test") / task_name

    print(f"\n{'='*60}")
    print(f"Task:    {task_name}")
    print(f"Success: val_acc >= {task.success_threshold}")
    print(f"Config:  {config}")
    print(f"{'='*60}")

    final = train_network(
        task       = task,
        config     = config,
        run_dir    = run_dir,
        rdm_inputs = rdm_inputs,
        ds_train   = ds_train,
        ds_val     = ds_val,
        verbose    = True,
    )
    solved = final >= task.success_threshold
    print(f"\nFinal val_acc: {final:.4f}  ({'SOLVED' if solved else 'FAILED'})")


for task_name in CONFIGS:
    run(task_name)
