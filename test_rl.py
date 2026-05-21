"""Smoke test for RL tasks. Run from repo root:
    python test_rl.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tasks import TASKS
from src.train_rl import train_network

CARTPOLE_CONFIG = {
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

FOURROOMS_CONFIG = {
    "hidden_size":   64,
    "depth":         2,
    "activation":    "relu",
    "optimizer":     "adam",
    "learning_rate": 5e-4,
    "l1_reg":        0.0,
    "l2_reg":        0.0,
    "init_scale":    1.0,
    "gamma":         0.99,
}


def run(task_name, config):
    task        = TASKS[task_name]()
    env_factory = task.get_data()
    rdm_inputs, _ = task.get_rdm_stimuli()
    run_dir     = Path("experiments_test") / task_name

    print(f"\n{'='*60}")
    print(f"Task:      {task_name}")
    print(f"Max steps: {task.max_steps:,}")
    print(f"Success:   mean_return >= {task.success_threshold}")
    print(f"Config:    {config}")
    print(f"{'='*60}")

    final = train_network(
        task        = task,
        config      = config,
        run_dir     = run_dir,
        rdm_inputs  = rdm_inputs,
        env_factory = env_factory,
        verbose     = True,
    )
    print(f"\nFinal mean_return: {final:.2f}  "
          f"({'SOLVED' if final >= task.success_threshold else 'FAILED'})")


run("cartpole",  CARTPOLE_CONFIG)
run("fourrooms", FOURROOMS_CONFIG)
