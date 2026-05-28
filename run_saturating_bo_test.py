"""
Saturating BO test on spirals: reuse Sobol nets, run GP phase with UCB/N_eff acquisition.

Copies the 180 Sobol-phase observations from experiments_culling_test/spirals/
(144 primaries + 36 repeats) into experiments_saturating_test/spirals/ and runs
N_GP new GP primaries with fixed beta=4.0 and N_eff denominator (h=0.2, lam=0.1).

Usage:
    .venv/bin/python run_saturating_bo_test.py [--data-dir data] [--beta 4.0]
                                               [--n-gp 144] [--n-candidates 2048]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import src.bo as _bo_module
_bo_module.N_SOBOL = 144   # match Sobol-phase size from source

from tasks import TASKS
from src.train_supervised import train_network
from src.bo import (
    get_all_combos, cat_params_for_task, _cont_params_for_task,
    suggest_next, save_state, load_state,
    build_run_counts, get_primary_observations,
    build_XY, fit_gp, compute_n_eff,
)

TASK_NAME      = "spirals"
SOURCE_DIR     = "experiments_culling_test"
OUTPUT_DIR     = "experiments_saturating_test"
STATE_FILE     = "bo_state.json"
N_SOBOL        = 144    # primary observations copied from source
N_SOBOL_TOTAL  = 180    # total iterations in Sobol phase (144 primary + 36 repeats)
DIAG_EVERY     = 10     # GP primaries between diagnostic prints


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",     default="data")
    p.add_argument("--output-dir",   default=None,
                   help="Output directory (default: experiments_saturating_test_h{h})")
    p.add_argument("--beta",         type=float, default=4.0)
    p.add_argument("--h",            type=float, default=0.2)
    p.add_argument("--lam",          type=float, default=0.1)
    p.add_argument("--n-gp",         type=int,   default=144,
                   help="Number of new GP primary observations to run")
    p.add_argument("--n-candidates", type=int,   default=2048,
                   help="Sobol grid size for acquisition evaluation")
    p.add_argument("--max-epochs",  type=int,   default=None,
                   help="Override task max_epochs (e.g. 2 for a smoke test)")
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


def run_config(task, config, run_id_base, output_dir, rdm_inputs, ds_train, ds_val,
               max_epochs_override=None):
    run_dir = Path(output_dir) / f"{run_id_base}_r0"
    print(f"    ->  {run_dir.name}")
    val_acc = train_network(
        task=task, config=config, run_dir=run_dir,
        rdm_inputs=rdm_inputs, ds_train=ds_train, ds_val=ds_val,
        max_epochs_override=max_epochs_override,
        verbose=True,
    )
    flag = "OK" if val_acc >= task.success_threshold else "FAILED"
    print(f"        val_acc={val_acc:.4f}  [{flag}]")
    return val_acc


def print_diagnostic(observations, task, gp_primary_count, h, lam):
    cont_params = _cont_params_for_task(task)
    cat_params  = cat_params_for_task(task)
    all_combos  = get_all_combos(task)
    primary_obs = get_primary_observations(observations)
    gp_obs      = [o for o in primary_obs if o["iteration"] >= N_SOBOL_TOTAL]

    # Coverage: how many GP obs went to each categorical value
    print(f"\n  --- GP diagnostic ({gp_primary_count} GP primaries) ---")
    for name, choices in cat_params:
        parts = []
        for val in choices:
            n = sum(1 for o in gp_obs if o["config"][name] == val)
            parts.append(f"{val}: {n}")
        print(f"    {name:14s}  " + "   ".join(parts))

    # N_eff stats over the GP observations
    neff_vals = [
        compute_n_eff(o["config"], primary_obs, cont_params, cat_params, h=h, lam=lam)
        for o in gp_obs
    ]
    if neff_vals:
        print(f"    N_eff over GP obs:  min={min(neff_vals):.2f}  "
              f"mean={np.mean(neff_vals):.2f}  max={max(neff_vals):.2f}")

    # Success rates
    def sr(obs_list):
        n = len(obs_list)
        if n == 0:
            return "0/0 (n/a)"
        s = sum(1 for o in obs_list if o["mean_metric"] >= task.success_threshold)
        return f"{s}/{n} ({100*s/n:.1f}%)"

    sobol_obs = [o for o in primary_obs if o["iteration"] < N_SOBOL_TOTAL]
    print(f"    Sobol success: {sr(sobol_obs)}")
    print(f"    GP success:    {sr(gp_obs)}")
    print()


def main():
    args = parse_args()

    n_gp   = args.n_gp
    n_iter = N_SOBOL_TOTAL + n_gp * 5 // 4  # 20% repeats

    task       = TASKS[TASK_NAME]()
    out_base   = args.output_dir or f"experiments_saturating_test_h{args.h:.2f}".replace(".", "")
    output_dir = Path(out_base) / TASK_NAME
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
    print(f"chance_accuracy={task.chance_accuracy:.3f}  "
          f"beta={args.beta}  h={args.h}  lam={args.lam}")
    print(f"Plan: {n_gp} GP primaries  (~{n_iter} total iterations)")
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

    for iteration in range(n_done, n_iter):
        print(f"\n{'='*60}")

        repeat_config, repeat_of_idx = _pending_repeat(observations)

        if repeat_config is not None:
            config    = repeat_config
            is_repeat = True
            print(f"[{iteration+1}/{n_iter}]  REPEAT of obs_{repeat_of_idx:04d}")
        else:
            config, combo_idx, mode = suggest_next(
                observations, task,
                beta=args.beta, h=args.h, lam=args.lam,
                n_candidates=args.n_candidates,
            )
            is_repeat = False
            counts_now = build_run_counts(get_primary_observations(observations),
                                          all_combos, cat_params)
            n_prev = counts_now[combo_idx]
            print(f"[{iteration+1}/{n_iter}]  combo #{combo_idx}  "
                  f"({mode},  {n_prev} prior primary obs)")

            if "gp" in mode:
                gp_primaries += 1
                if gp_primaries % DIAG_EVERY == 0:
                    print_diagnostic(observations, task, gp_primaries, args.h, args.lam)

        pretty = {k: (round(v, 6) if isinstance(v, float) else v)
                  for k, v in config.items()}
        print(f"  config: {json.dumps(pretty, separators=(',', ':'))}")

        val_acc = run_config(task, config, f"run_{iteration:04d}", output_dir,
                             rdm_inputs, ds_train, ds_val,
                             max_epochs_override=args.max_epochs)
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
    primary_final = get_primary_observations(observations)
    sobol_final   = [o for o in primary_final if o["iteration"] < N_SOBOL_TOTAL]
    gp_final      = [o for o in primary_final if o["iteration"] >= N_SOBOL_TOTAL]
    n_repeats     = len(observations) - len(primary_final)

    def sr(obs_list):
        n = len(obs_list)
        s = sum(1 for o in obs_list if o["mean_metric"] >= task.success_threshold)
        return f"{s}/{n} ({100*s/n:.1f}%)"

    print(f"Total: {len(observations)} runs  ({len(primary_final)} primary, {n_repeats} repeats)")
    print(f"Sobol success: {sr(sobol_final)}")
    print(f"GP success:    {sr(gp_final)}")

    print_diagnostic(observations, task, len(gp_final), args.h, args.lam)


if __name__ == "__main__":
    main()
