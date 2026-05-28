"""
Spirals culling test: 144 Sobol primaries -> 144 GP primaries with UCB-based combo culling.

Stored in experiments_culling_test/spirals/ for comparison with experiments/spirals/.
Every DIAG_EVERY GP primaries, prints a table showing UCB max per combo and which
categorical values are being excluded, so we can see whether and where culling breaks down.

Usage:
    .venv/bin/python run_spirals_culling_test.py [--data-dir data] [--beta 8.0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# Override N_SOBOL before bo functions capture it — must happen before bo imports
import src.bo as _bo_module
_bo_module.N_SOBOL = 144

from tasks import TASKS
from src.train_supervised import train_network
from src.bo import (
    get_all_combos, cat_params_for_task, _cont_params_for_task,
    suggest_next, save_state, load_state,
    build_run_counts, get_primary_observations,
    get_active_combos, build_XY, fit_gp, _combo_ucb_max,
)

TASK_NAME  = "spirals"
OUTPUT_DIR = "experiments_culling_test"
STATE_FILE = "bo_state.json"
N_SOBOL    = 144    # primary observations in Sobol phase (1 per combo)
N_GP       = 144    # primary observations in GP phase
N_PRIMARIES = N_SOBOL + N_GP                  # 288
N_ITER      = N_PRIMARIES * 5 // 4            # 360 total (20% repeats)
DIAG_EVERY  = 10    # GP primaries between diagnostic table prints


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--beta",     type=float, default=8.0)
    return p.parse_args()


def _pending_repeat(observations):
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


def run_config(task, config, run_id_base, output_dir, rdm_inputs, ds_train, ds_val):
    run_dir = Path(output_dir) / f"{run_id_base}_r0"
    print(f"    ->  {run_dir.name}")
    val_acc = train_network(
        task=task, config=config, run_dir=run_dir,
        rdm_inputs=rdm_inputs, ds_train=ds_train, ds_val=ds_val,
        verbose=True,
    )
    flag = "OK" if val_acc >= task.success_threshold else "FAILED"
    print(f"        val_acc={val_acc:.4f}  [{flag}]")
    return val_acc


def print_culling_diagnostic(observations, task, beta):
    """Fit GP on current data, print UCB max per combo and per-category exclusion counts."""
    cont_params = _cont_params_for_task(task)
    cat_params  = cat_params_for_task(task)
    all_combos  = get_all_combos(task)

    X, Y = build_XY(observations, cont_params, cat_params)
    gp   = fit_gp(X, Y, len(cont_params))

    active_keys = {
        json.dumps(c, sort_keys=True)
        for c in get_active_combos(gp, all_combos, cont_params, cat_params,
                                   task.success_threshold, beta)
    }

    ucb_rows = []
    for combo in all_combos:
        ucb = _combo_ucb_max(gp, combo, cont_params, cat_params, beta)
        is_active = json.dumps(combo, sort_keys=True) in active_keys
        ucb_rows.append((ucb, combo, is_active))

    n_active   = sum(1 for _, _, a in ucb_rows if a)
    n_excluded = len(all_combos) - n_active
    print(f"\n  --- Culling diagnostic: {n_active}/{len(all_combos)} active, "
          f"{n_excluded} excluded (threshold={task.success_threshold}) ---")

    # Per-categorical-value: how many combos containing that value are excluded
    for name, choices in cat_params:
        parts = []
        for val in choices:
            subset = [(u, a) for u, c, a in ucb_rows if c[name] == val]
            n_excl = sum(1 for _, a in subset if not a)
            parts.append(f"{val}: {n_excl}/{len(subset)} excl")
        print(f"    {name:14s}  " + "   ".join(parts))

    # Bottom 10 combos by UCB
    ucb_rows.sort(key=lambda x: x[0])
    print(f"  Bottom 10 by UCB max:")
    for ucb, combo, is_active in ucb_rows[:10]:
        tag = "active  " if is_active else "EXCLUDED"
        vals = "  ".join(f"{k}={v}" for k, v in combo.items())
        print(f"    [{tag}]  ucb={ucb:.4f}  {vals}")
    print()


def main():
    args = parse_args()

    task       = TASKS[TASK_NAME]()
    output_dir = Path(OUTPUT_DIR) / TASK_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / STATE_FILE

    all_combos = get_all_combos(task)
    cat_params = cat_params_for_task(task)

    print(f"Task: {task.name}  ({len(all_combos)} categorical combos)")
    print(f"Plan: {N_SOBOL} Sobol primaries -> {N_GP} GP primaries "
          f"(~{N_ITER} total iterations with 20% repeats)")
    print(f"Output: {output_dir}")

    print("\nLoading data...")
    ds_train, ds_val = task.get_data(data_dir=args.data_dir)
    print(f"  train={len(ds_train)}  val={len(ds_val)}")
    rdm_inputs, _ = task.get_rdm_stimuli(data_dir=args.data_dir)
    print(f"  RDM stimuli: {rdm_inputs.shape}")

    observations  = load_state(state_path)
    n_done        = len(observations)
    primary_obs   = get_primary_observations(observations)
    n_primary     = len(primary_obs)
    n_repeat      = n_done - n_primary
    run_counts    = build_run_counts(primary_obs, all_combos, cat_params)
    coverage      = sum(1 for c in run_counts if c > 0)
    gp_primaries  = max(0, n_primary - N_SOBOL)
    print(f"\nResuming: {n_done} total ({n_primary} primary, {n_repeat} repeats), "
          f"{coverage}/{len(all_combos)} combos visited, "
          f"{gp_primaries} GP primaries done.")

    for iteration in range(n_done, N_ITER):
        print(f"\n{'='*60}")

        repeat_config, repeat_of_idx = _pending_repeat(observations)

        if repeat_config is not None:
            config    = repeat_config
            is_repeat = True
            print(f"[{iteration+1}/{N_ITER}]  REPEAT of run_{repeat_of_idx:04d}")
        else:
            config, combo_idx, mode = suggest_next(observations, task, beta=args.beta)
            is_repeat = False
            primary_now = get_primary_observations(observations)
            counts_now  = build_run_counts(primary_now, all_combos, cat_params)
            n_prev      = counts_now[combo_idx]
            print(f"[{iteration+1}/{N_ITER}]  combo #{combo_idx}  "
                  f"({mode}, {n_prev} prior primary obs)")

            if "gp" in mode:
                gp_primaries += 1
                if gp_primaries % DIAG_EVERY == 0:
                    print_culling_diagnostic(observations, task, args.beta)

        pretty = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in config.items()}
        print(f"  config: {json.dumps(pretty, separators=(',', ':'))}")

        val_acc = run_config(task, config, f"run_{iteration:04d}", output_dir,
                             rdm_inputs, ds_train, ds_val)
        print(f"  mean_metric = {val_acc:.4f}")

        observations.append({
            "iteration":   iteration,
            "config":      config,
            "val_accs":    [val_acc],
            "mean_metric": val_acc,
            "is_repeat":   is_repeat,
            "repeat_of":   repeat_of_idx if is_repeat else None,
        })
        save_state(state_path, observations)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    primary_final   = get_primary_observations(observations)
    n_success       = sum(1 for o in primary_final if o["mean_metric"] >= task.success_threshold)
    counts_final    = build_run_counts(primary_final, all_combos, cat_params)
    coverage_final  = sum(1 for c in counts_final if c > 0)
    n_repeats_final = len(observations) - len(primary_final)

    sobol_primary  = [o for o in primary_final if o["iteration"] < N_SOBOL * 5 // 4]
    gp_primary     = [o for o in primary_final if o not in sobol_primary]

    print(f"Total runs:  {len(observations)}  ({len(primary_final)} primary, {n_repeats_final} repeats, "
          f"{100*n_repeats_final/len(observations):.1f}%)")
    print(f"Successes:   {n_success}/{len(primary_final)} ({100*n_success/len(primary_final):.1f}%)")
    print(f"Coverage:    {coverage_final}/{len(all_combos)} combos visited")
    print(f"Best:        {max(primary_final, key=lambda o: o['mean_metric'])['mean_metric']:.4f}")

    print("\nFinal culling state:")
    print_culling_diagnostic(observations, task, args.beta)


if __name__ == "__main__":
    main()
