#!/usr/bin/env python3
"""
Reconstruct missing bo_state.json entries from orphaned run directories
(those whose iteration number is not in the existing bo_state.json).

Reads metadata.json from each orphaned run_NNNN_r0/ directory and reconstructs
the observation dict, replaying the _pending_repeat logic in order to correctly
assign is_repeat / repeat_of.

Backs up bo_state.json to bo_state.json.bak before writing.

Usage:
    cd /home/thomas/hyperparams_and_geometry
    .venv/bin/python scripts/reconstruct_bo_state.py <task_name>
    .venv/bin/python scripts/reconstruct_bo_state.py mnist_10way
    .venv/bin/python scripts/reconstruct_bo_state.py fashion_10way
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks import TASKS
from src.bo import (
    _cont_params_for_task,
    _cont_to_unit,
    get_primary_observations,
    load_state,
    save_state,
)


def _pending_repeat(observations):
    """Mirror of run_bo._pending_repeat — must stay in sync with that function."""
    primary_obs = get_primary_observations(observations)
    n_primary = len(primary_obs)
    if n_primary == 0 or n_primary % 4 != 0:
        return None, None
    last_primary_idx = None
    for i in range(len(observations) - 1, -1, -1):
        if not observations[i].get("is_repeat", False):
            last_primary_idx = i
            break
    has_repeat = any(
        o.get("is_repeat") and o.get("repeat_of") == last_primary_idx
        for o in observations
    )
    if has_repeat:
        return None, None
    return observations[last_primary_idx]["config"], last_primary_idx


def configs_match(a, b):
    """Compare two config dicts, tolerating float rounding differences."""
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, float) and isinstance(vb, float):
            if abs(va - vb) / max(abs(va), abs(vb), 1e-12) > 1e-6:
                return False
        elif va != vb:
            return False
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: reconstruct_bo_state.py <task_name>")
        sys.exit(1)
    task_name = sys.argv[1]
    output_dir = Path("output/production") / task_name
    state_path = output_dir / "bo_state.json"
    backup_path = output_dir / "bo_state.json.bak"

    task = TASKS[task_name]()
    cont_params = _cont_params_for_task(task)

    observations = load_state(state_path)
    print(f"Loaded {len(observations)} existing observations "
          f"(iterations 0–{observations[-1]['iteration']})")

    existing_iters = {obs["iteration"] for obs in observations}

    # Collect orphaned dirs in numerical order
    all_run_dirs = sorted(
        output_dir.glob("run_????_r0"),
        key=lambda d: int(d.name.split("_")[1])
    )
    orphaned = [d for d in all_run_dirs
                if int(d.name.split("_")[1]) not in existing_iters]
    print(f"Found {len(orphaned)} orphaned run directories "
          f"(iterations {int(orphaned[0].name.split('_')[1])}–"
          f"{int(orphaned[-1].name.split('_')[1])})")

    reconstructed = 0
    warnings = []

    for run_dir in orphaned:
        iter_num = int(run_dir.name.split("_")[1])
        meta_path = run_dir / "metadata.json"

        if not meta_path.exists():
            msg = f"iter {iter_num}: no metadata.json in {run_dir.name} — skipping"
            print(f"  WARNING: {msg}")
            warnings.append(msg)
            continue

        meta = json.load(open(meta_path))
        config = meta["config"]
        best_metric = meta["best_metric"]

        repeat_config, repeat_of_idx = _pending_repeat(observations)

        if repeat_config is not None:
            is_repeat = True
            cont_unit_vals = observations[repeat_of_idx]["cont_unit_vals"]
            if not configs_match(config, repeat_config):
                msg = (f"iter {iter_num}: repeat config mismatch\n"
                       f"    expected: {repeat_config}\n"
                       f"    got:      {config}")
                print(f"  WARNING: {msg}")
                warnings.append(msg)
        else:
            is_repeat = False
            cont_unit_vals = _cont_to_unit(config, cont_params)
            repeat_of_idx = None

        obs = {
            "iteration":      iter_num,
            "config":         config,
            "cont_unit_vals": cont_unit_vals,
            "val_accs":       [best_metric],
            "performance":    best_metric,
            "is_repeat":      is_repeat,
            "repeat_of":      repeat_of_idx,
        }

        observations.append(obs)
        reconstructed += 1

    print(f"\nReconstructed {reconstructed} entries  ({len(warnings)} warnings)")

    # Verify order
    iters = [o["iteration"] for o in observations]
    if iters != sorted(iters):
        print("Iterations out of order — sorting...")
        observations.sort(key=lambda o: o["iteration"])

    n_primary = sum(1 for o in observations if not o.get("is_repeat"))
    n_repeat  = len(observations) - n_primary
    print(f"Total: {len(observations)} obs  ({n_primary} primary, {n_repeat} repeats, "
          f"{100*n_repeat/len(observations):.1f}% repeat rate)")

    # Back up, then save
    shutil.copy(state_path, backup_path)
    print(f"Backed up existing state to {backup_path}")
    save_state(state_path, observations)
    print(f"Saved reconstructed state to {state_path}")

    if warnings:
        print(f"\n{len(warnings)} warnings — review above before continuing the run.")


if __name__ == "__main__":
    main()
