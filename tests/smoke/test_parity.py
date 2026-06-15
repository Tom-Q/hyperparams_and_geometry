"""Smoke test for parity: trains one network and checks the recorded
metadata/history and RDM activation files (structure, checkpoint naming, and
content sanity)."""
import json
import math

import pytest

from tasks import TASKS
from src.train_supervised import train_network
from src.utils import log4_checkpoints, epoch_checkpoints, format_epoch_label, PERF_THRESHOLDS
from tests._helpers import check_mlp_activations, mlp_layer_sizes

# Parity is hard for shallow MLPs; needs a wider net, higher LR, and more
# epochs than the other supervised tasks to actually solve (val_acc >= 0.95).
CONFIG = {
    "hidden_size":   256,
    "depth":         2,
    "batch_size":    64,
    "optimizer":     "adam",
    "learning_rate": 5e-3,
    "l1_reg":        0.0,
    "l2_reg":        0.0,
    "init_scale":    1.0,
    "activation":    "relu",
}
MAX_EPOCHS = 150


@pytest.mark.slow
def test_train_and_activations(tmp_path):
    task = TASKS["parity"]()
    ds_train, ds_val = task.get_data(data_dir="data")
    rdm_inputs, _ = task.get_rdm_stimuli(data_dir="data")
    n_stimuli = rdm_inputs.shape[0]
    layer_sizes = mlp_layer_sizes(CONFIG["hidden_size"], CONFIG["depth"])

    run_dir = tmp_path / "parity"
    train_network(
        task=task, config=CONFIG, run_dir=run_dir,
        rdm_inputs=rdm_inputs, ds_train=ds_train, ds_val=ds_val,
        max_epochs_override=MAX_EPOCHS, verbose=False, save_activations=True,
    )

    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "history.json").exists()
    meta = json.loads((run_dir / "metadata.json").read_text())
    final_step = meta["final_step"]

    # --- checkpoint directory structure ---
    steps_per_epoch = math.ceil(len(ds_train) / CONFIG["batch_size"])
    total_steps = MAX_EPOCHS * steps_per_epoch

    expected_step = {f"step_{s:07d}" for s in log4_checkpoints(total_steps) if s <= final_step}
    actual_step = {p.stem for p in run_dir.glob("step_*.npz")}
    assert actual_step == expected_step

    expected_epoch = {
        f"epoch_{format_epoch_label(e)}"
        for s, e in epoch_checkpoints(steps_per_epoch, MAX_EPOCHS).items()
        if s <= final_step
    }
    actual_epoch = {p.stem for p in run_dir.glob("epoch_*.npz")}
    assert actual_epoch == expected_epoch

    valid_perf_labels = {f"{t:g}".replace(".", "p") for t in PERF_THRESHOLDS}
    perf_files = list(run_dir.glob("perf_*.npz"))
    for p in perf_files:
        assert p.stem[len("perf_"):] in valid_perf_labels

    assert (run_dir / "final.npz").exists()
    assert (run_dir / "best.npz").exists()

    # A config that reaches val_acc == 1.0 should cross every perf threshold.
    assert len(perf_files) == len(PERF_THRESHOLDS)

    # --- activation content: no NaN/Inf, right shape, non-degenerate ---
    for f in run_dir.glob("*.npz"):
        check_mlp_activations(f, layer_sizes=layer_sizes, n_stimuli=n_stimuli)
