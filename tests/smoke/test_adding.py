"""Smoke test for the adding task: trains one network and checks the recorded
metadata/history and RDM activation files (structure, checkpoint naming, and
content sanity)."""
import json
import math

import pytest

from tasks import TASKS
from src.train_rnn import train_network
from src.utils import log4_checkpoints, epoch_checkpoints, format_epoch_label, PERF_THRESHOLDS
from tests._helpers import check_rnn_activations

# A config that should solve the task comfortably within MAX_EPOCHS.
CONFIG = {
    "hidden_size":   64,
    "n_rnn_layers":  1,
    "batch_size":    64,
    "optimizer":     "adam",
    "learning_rate": 1e-3,
    "l1_reg":        0.0,
    "l2_reg":        0.0,
    "init_scale":    1.0,
    "cell_type":     "gru",
}
MAX_EPOCHS = 30


@pytest.mark.slow
def test_train_and_activations(tmp_path):
    task = TASKS["adding"]()
    ds_train, ds_val = task.get_data(data_dir="data")
    rdm_inputs, _ = task.get_rdm_stimuli(data_dir="data")
    n_stimuli = rdm_inputs.shape[0]

    run_dir = tmp_path / "adding_gru"
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

    # A reasonable config should cross at least some perf checkpoints.
    assert len(perf_files) > 0

    # --- activation content: no NaN/Inf, right shape, non-degenerate ---
    for f in run_dir.glob("*.npz"):
        check_rnn_activations(
            f, n_layers=CONFIG["n_rnn_layers"], n_steps=task.n_steps,
            n_stimuli=n_stimuli, hidden_size=CONFIG["hidden_size"],
        )
