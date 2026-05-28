"""
Spirals culling test v2: reuse Sobol nets, GP phase with exponential beta decay.

Copies the 180 Sobol-phase observations from experiments_culling_test/spirals/
(144 primaries + 36 repeats) into experiments_culling_test_v2/spirals/ and runs
144 new GP primaries with beta decaying exponentially from BETA_START to BETA_END.

Beta schedule: beta_t = BETA_START * (BETA_END/BETA_START)^(t / (N_GP - 1))
Both acquisition and exclusion use sqrt(beta) × σ (BoTorch convention).

Usage:
    .venv/bin/python run_spirals_culling_test_v2.py [--data-dir data]
"""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

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

TASK_NAME   = "spirals"
SOURCE_DIR  = "experiments_culling_test"   # where Sobol nets live
OUTPUT_DIR  = "experiments_culling_test_v2"
STATE_FILE  = "bo_state.json"
N_SOBOL     = 144    # primary observations to copy from source
N_GP        = 144    # new GP primaries to run
N_SOBOL_TOTAL = 180  # total iterations in Sobol phase (144 primary + 36 repeats)
N_ITER      = N_SOBOL_TOTAL + N_GP * 5 // 4   # 180 + 180 = 360 total
BETA_START  = 8.0
BETA_END    = 2.0
DIAG_EVERY  = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    return p.parse_args()


def beta_at(gp_primary_idx, n_gp_total, beta_start, beta_end):
    """Exponential decay: beta_start at t=0, beta_end at t=n_gp_total-1."""
    t = gp_primary_idx / max(1, n_gp_total - 1)
    return beta_start * (beta_end / beta_start) ** t


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
    ucb_rows = [
        (_combo_ucb_max(gp, c, cont_params, cat_params, beta), c,
         json.dumps(c, sort_keys=True) in active_keys)
        for c in all_combos
    ]
    n_active = sum(1 for _, _, a in ucb_rows if a)
    print(f"\n  --- Culling diagnostic (beta={beta:.2f}, sqrt={math.sqrt(beta):.2f}): "
          f"{n_active}/{len(all_combos)} active ---")

    for name, choices in cat_params:
        parts = []
        for val in choices:
            subset = [(u, a) for u, c, a in ucb_rows if c[name] == val]
            n_excl = sum(1 for _, a in subset if not a)
            parts.append(f"{val}: {n_excl}/{len(subset)} excl")
        print(f"    {name:14s}  " + "   ".join(parts))

    ucb_rows.sort(key=lambda x: x[0])
    print(f"  Bottom 10 by UCB max (threshold={task.success_threshold}):")
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

    # ------------------------------------------------------------------
    # Bootstrap from source Sobol state if not already done
    # ------------------------------------------------------------------
    if not state_path.exists():
        source_path = Path(SOURCE_DIR) / TASK_NAME / STATE_FILE
        if not source_path.exists():
            print(f"ERROR: source state not found at {source_path}")
            sys.exit(1)
        with open(source_path) as f:
            all_source = json.load(f)
        sobol_obs = [o for o in all_source if o["iteration"] < N_SOBOL_TOTAL]
        with open(state_path, "w") as f:
            json.dump(sobol_obs, f, indent=2)
        print(f"Copied {len(sobol_obs)} Sobol-phase observations from {source_path}")
    else:
        print(f"State file already exists at {state_path}, resuming.")

    print(f"\nTask: {task.name}  ({len(all_combos)} categorical combos)")
    print(f"Beta schedule: {BETA_START} -> {BETA_END} (exponential) over {N_GP} GP primaries")
    print(f"Output: {output_dir}")

    print("\nLoading data...")
    ds_train, ds_val = task.get_data(data_dir=args.data_dir)
    rdm_inputs, _    = task.get_rdm_stimuli(data_dir=args.data_dir)
    print(f"  train={len(ds_train)}  val={len(ds_val)}  "
          f"RDM stimuli: {rdm_inputs.shape}")

    observations = load_state(state_path)
    n_done       = len(observations)
    primary_obs  = get_primary_observations(observations)
    n_primary    = len(primary_obs)
    gp_primaries = max(0, n_primary - N_SOBOL)
    print(f"\nResuming: {n_done} total obs, {n_primary} primaries, "
          f"{gp_primaries} GP primaries already done.")

    for iteration in range(n_done, N_ITER):
        print(f"\n{'='*60}")

        repeat_config, repeat_of_idx = _pending_repeat(observations)

        if repeat_config is not None:
            config    = repeat_config
            is_repeat = True
            print(f"[{iteration+1}/{N_ITER}]  REPEAT of obs_{repeat_of_idx:04d}")
        else:
            current_beta = beta_at(gp_primaries, N_GP, BETA_START, BETA_END)
            config, combo_idx, mode = suggest_next(observations, task, beta=current_beta)
            is_repeat = False
            primary_now = get_primary_observations(observations)
            counts_now  = build_run_counts(primary_now, all_combos, cat_params)
            n_prev      = counts_now[combo_idx]
            print(f"[{iteration+1}/{N_ITER}]  combo #{combo_idx}  "
                  f"({mode}, beta={current_beta:.2f}, {n_prev} prior primary obs)")

            if "gp" in mode:
                gp_primaries += 1
                if gp_primaries % DIAG_EVERY == 0:
                    print_culling_diagnostic(observations, task, current_beta)

        pretty = {k: (round(v, 6) if isinstance(v, float) else v)
                  for k, v in config.items()}
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

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    primary_final   = get_primary_observations(observations)
    gp_final        = [o for o in primary_final if o["iteration"] >= N_SOBOL_TOTAL]
    sobol_final     = [o for o in primary_final if o["iteration"] < N_SOBOL_TOTAL]
    n_repeats_final = len(observations) - len(primary_final)

    def sr(obs):
        return f"{sum(1 for o in obs if o['mean_metric'] >= task.success_threshold)}/{len(obs)}" \
               f" ({100*sum(1 for o in obs if o['mean_metric'] >= task.success_threshold)/max(1,len(obs)):.1f}%)"

    print(f"Total:         {len(observations)} runs  ({len(primary_final)} primary, {n_repeats_final} repeats)")
    print(f"Sobol success: {sr(sobol_final)}")
    print(f"GP success:    {sr(gp_final)}")

    print("\nFinal culling state:")
    final_beta = beta_at(N_GP - 1, N_GP, BETA_START, BETA_END)
    print_culling_diagnostic(observations, task, final_beta)


if __name__ == "__main__":
    main()
