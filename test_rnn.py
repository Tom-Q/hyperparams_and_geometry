"""Smoke test for RNN tasks. Run from repo root:
    python test_rnn.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tasks import TASKS
from src.train_rnn import train_network

MNIST_RNN_CONFIG = {
    "hidden_size":   128,
    "cell_type":     "gru",
    "n_rnn_layers":  1,
    "batch_size":    64,
    "optimizer":     "adam",
    "learning_rate": 1e-3,
    "l1_reg":        0.0,
    "l2_reg":        0.0,
    "init_scale":    1.0,
}

ADDING_CONFIG = {
    "hidden_size":   64,
    "cell_type":     "gru",
    "n_rnn_layers":  1,
    "batch_size":    64,
    "optimizer":     "adam",
    "learning_rate": 1e-3,
    "l1_reg":        0.0,
    "l2_reg":        0.0,
    "init_scale":    1.0,
}


def run(task_name, config):
    task = TASKS[task_name]()
    ds_train, ds_val = task.get_data(data_dir="data")
    rdm_inputs, _    = task.get_rdm_stimuli(data_dir="data")
    run_dir          = Path("experiments_test") / task_name

    print(f"\n{'='*60}")
    print(f"Task:     {task_name}")
    print(f"Metric:   {task.metric_name}  (success threshold: {task.success_threshold})")
    print(f"Config:   {config}")
    print(f"{'='*60}")

    final = train_network(
        task     = task,
        config   = config,
        run_dir  = run_dir,
        rdm_inputs = rdm_inputs,
        ds_train = ds_train,
        ds_val   = ds_val,
        verbose  = True,
    )

    if task.metric_name == "val_mse":
        solved = final <= task.success_threshold
        print(f"\nFinal val_mse: {final:.4f}  ({'SOLVED' if solved else 'FAILED'})")
    else:
        solved = final >= task.success_threshold
        print(f"\nFinal val_acc: {final:.4f}  ({'SOLVED' if solved else 'FAILED'})")


run("mnist_rnn", MNIST_RNN_CONFIG)
run("adding",    ADDING_CONFIG)
