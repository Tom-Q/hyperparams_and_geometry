"""Smoke test for fourrooms: trains one network with online Q-learning and
checks the recorded metadata/history and RDM activation files (structure,
checkpoint naming, and content sanity)."""
import json

import pytest

from tasks import TASKS
from src.train_rl import train_network
from src.utils import log4_checkpoints, PERF_THRESHOLDS
from tests._helpers import check_mlp_activations, mlp_layer_sizes

CONFIG = {
    "hidden_size":   64,
    "depth":         1,
    "optimizer":     "adam",
    "learning_rate": 1e-3,
    "l1_reg":        0.0,
    "l2_reg":        0.0,
    "init_scale":    1.0,
    "activation":    "relu",
}
MAX_STEPS = 20_000


@pytest.mark.slow
def test_train_and_activations(tmp_path):
    task = TASKS["fourrooms"]()
    env_factory = task.get_data()
    rdm_inputs, _ = task.get_rdm_stimuli()
    n_stimuli = rdm_inputs.shape[0]
    layer_sizes = mlp_layer_sizes(CONFIG["hidden_size"], CONFIG["depth"])

    run_dir = tmp_path / "fourrooms"
    train_network(
        task=task, config=CONFIG, run_dir=run_dir,
        rdm_inputs=rdm_inputs, env_factory=env_factory,
        max_steps_override=MAX_STEPS, verbose=False, save_activations=True,
    )

    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "history.json").exists()
    meta = json.loads((run_dir / "metadata.json").read_text())
    final_step = meta["final_step"]

    # --- checkpoint directory structure ---
    expected_step = {f"step_{s:07d}" for s in log4_checkpoints(MAX_STEPS) if s <= final_step}
    actual_step = {p.stem for p in run_dir.glob("step_*.npz")}
    assert actual_step == expected_step

    # RL training has no epoch checkpoints or "best" snapshot.
    assert not list(run_dir.glob("epoch_*.npz"))
    assert not (run_dir / "best.npz").exists()

    valid_perf_labels = {f"{t:g}".replace(".", "p") for t in PERF_THRESHOLDS}
    perf_files = list(run_dir.glob("perf_*.npz"))
    for p in perf_files:
        assert p.stem[len("perf_"):] in valid_perf_labels

    assert (run_dir / "final.npz").exists()

    # A modest amount of online Q-learning should cross several perf thresholds.
    assert len(perf_files) >= 3

    # --- activation content: no NaN/Inf, right shape, non-degenerate ---
    for f in run_dir.glob("*.npz"):
        check_mlp_activations(f, layer_sizes=layer_sizes, n_stimuli=n_stimuli)
