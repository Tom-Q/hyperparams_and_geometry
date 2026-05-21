"""Spiral task diagnostic. Run from repo root:
    python test_spirals.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tasks import TASKS
from src.train_supervised import train_network

CONFIG = {
    "hidden_size":   128,
    "depth":         3,
    "activation":    "relu",
    "batch_size":    64,
    "optimizer":     "adam",
    "learning_rate": 1e-3,
    "l1_reg":        0.0,
    "l2_reg":        0.0,
    "init_scale":    1.0,
}

task = TASKS["spirals"]()
ds_train, ds_val = task.get_data(data_dir="data")
rdm_inputs, _    = task.get_rdm_stimuli(data_dir="data")

print(f"train={len(ds_train)}  val={len(ds_val)}")
print(f"Success: val_acc >= {task.success_threshold}")
print(f"Config: {CONFIG}\n")

final = train_network(
    task               = task,
    config             = CONFIG,
    run_dir            = Path("experiments_test") / "spirals",
    rdm_inputs         = rdm_inputs,
    ds_train           = ds_train,
    ds_val             = ds_val,
    max_epochs_override= 500,
    verbose            = True,
)
solved = final >= task.success_threshold
print(f"\nBest val_acc: {final:.4f}  ({'SOLVED' if solved else 'FAILED'})")
