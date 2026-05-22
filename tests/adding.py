"""Smoke test: Adding problem (T=50). Run from repo root:
    python tests/adding.py
    python tests/adding.py --cell-type rnn --hidden-size 128
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks import TASKS
from src.train_rnn import train_network

DEFAULTS = {
    "hidden_size":   64,
    "n_rnn_layers":  1,
    "batch_size":    64,
    "optimizer":     "adam",
    "learning_rate": 1e-3,
    "l1_reg":        0.0,
    "l2_reg":        0.0,
    "init_scale":    1.0,
}

p = argparse.ArgumentParser()
p.add_argument("--cell-type",     type=str,   default=None,
               help="rnn or gru; if omitted, tests both")
p.add_argument("--hidden-size",   type=int,   default=DEFAULTS["hidden_size"])
p.add_argument("--n-rnn-layers",  type=int,   default=DEFAULTS["n_rnn_layers"])
p.add_argument("--batch-size",    type=int,   default=DEFAULTS["batch_size"])
p.add_argument("--optimizer",     type=str,   default=DEFAULTS["optimizer"])
p.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
p.add_argument("--l1-reg",        type=float, default=DEFAULTS["l1_reg"])
p.add_argument("--l2-reg",        type=float, default=DEFAULTS["l2_reg"])
p.add_argument("--init-scale",    type=float, default=DEFAULTS["init_scale"])
args = p.parse_args()

base = {
    "hidden_size":   args.hidden_size,
    "n_rnn_layers":  args.n_rnn_layers,
    "batch_size":    args.batch_size,
    "optimizer":     args.optimizer,
    "learning_rate": args.learning_rate,
    "l1_reg":        args.l1_reg,
    "l2_reg":        args.l2_reg,
    "init_scale":    args.init_scale,
}
cell_types = [args.cell_type] if args.cell_type else ["gru", "rnn"]

task = TASKS["adding"]()
ds_train, ds_val = task.get_data(data_dir="data")
rdm_inputs, _    = task.get_rdm_stimuli(data_dir="data")

print(f"Sequence: {task.n_steps} steps × {task.input_size} features  (scalar MSE output)")
print(f"Success: val_mse <= {task.success_threshold}  (random baseline ≈ 0.167)\n")

for cell_type in cell_types:
    config = {**base, "cell_type": cell_type}
    print(f"\n{'='*60}  cell_type={cell_type}")
    final = train_network(
        task       = task,
        config     = config,
        run_dir    = Path("experiments_test") / f"adding_{cell_type}",
        rdm_inputs = rdm_inputs,
        ds_train   = ds_train,
        ds_val     = ds_val,
        verbose    = True,
    )
    print(f"Best val_mse: {final:.4f}  ({'SOLVED' if final <= task.success_threshold else 'FAILED'})")
