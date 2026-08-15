#!/usr/bin/env python3
"""
Critical period analysis (Finding #3.2): rate of representational change
between consecutive checkpoints, as a function of training time.

For each consecutive pair of checkpoints, computes 1 - Spearman_r between
the last hidden layer activation matrices. All checkpoint types with
recoverable step numbers are used:
  - step_*.npz       : step encoded in filename
  - epoch_*.npz      : step looked up in history.json
  - perf_*.npz       : step inferred from first epoch crossing threshold
                       in history.json (non-RL tasks only)
  - best.npz         : best_step from metadata (non-RL tasks only)
  - final.npz        : final_step from metadata

X-axis: step_end / final_step (end of each consecutive interval).
Y-axis: 1 - Spearman r (representational change; 0 = no change).

Output:
  output/analysis/tables/critical_period.csv
  output/analysis/figures/critical_period.pdf
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ANALYSIS  = Path(__file__).parent
REPO_ROOT = ANALYSIS.parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import TABLES_DIR, FIGURES_DIR, FINAL_DIR, RL_TASKS, TASK_NAMES, task_meta, get_depth_from_config

PROD_DIR  = REPO_ROOT / "output" / "production"
RNN_TASKS = {"adding", "mnist_rnn"}


def load_last_hidden(npz_path, task, depth):
    try:
        data = np.load(npz_path)
    except Exception:
        return None
    if task in RNN_TASKS:
        keys = sorted(k for k in data.files if k.startswith("layer_0_t_"))
        if not keys:
            return None
        return np.concatenate([data[k] for k in keys], axis=1).astype(np.float64)
    else:
        arr = data.get(f"layer_{depth - 1}")
        return arr.astype(np.float64) if arr is not None else None


def spearman_finite(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return float("nan")
    return float(spearmanr(a[mask], b[mask]).statistic)


def load_bo_perf(task):
    """Return {run_id: performance} from bo_state.json (canonical source)."""
    state_path = PROD_DIR / task / "bo_state.json"
    observations = json.load(open(state_path))
    return {f"run_{obs['iteration']:04d}_r0": obs["performance"]
            for obs in observations}


def collect_checkpoints(run_dir, task, meta):
    """
    Return {step: npz_path} for all checkpoints with recoverable step numbers.

    For RL tasks: step_*.npz + final.npz only (no epochs/history).
    For non-RL:   adds epoch_*.npz, perf_*.npz (via history.json), best.npz.
    """
    final_step = meta["final_step"]
    ckpts = {}

    # Step checkpoints — step encoded in filename
    for npz in run_dir.glob("step_*.npz"):
        step = int(npz.stem.replace("step_", ""))
        ckpts[step] = npz

    # Final checkpoint
    final_npz = run_dir / "final.npz"
    if final_npz.exists():
        ckpts[final_step] = final_npz

    if task in RL_TASKS:
        return ckpts

    # Best checkpoint
    best_npz  = run_dir / "best.npz"
    best_step = meta.get("best_step")
    if best_npz.exists() and best_step:
        ckpts[int(best_step)] = best_npz

    # Load history.json for epoch/perf step recovery
    h_path = run_dir / "history.json"
    if not h_path.exists():
        return ckpts
    history = json.load(open(h_path))
    if not history:
        return ckpts

    epoch_to_step    = {int(e["epoch"]): e["step"] for e in history}
    steps_per_epoch  = history[0]["step"] / history[0]["epoch"]

    # Epoch checkpoints
    for npz in run_dir.glob("epoch_*.npz"):
        epoch_str = npz.stem.replace("epoch_", "").replace("p", ".")
        epoch     = float(epoch_str)
        int_epoch = int(epoch)
        if float(int_epoch) == epoch and int_epoch in epoch_to_step:
            ckpts[epoch_to_step[int_epoch]] = npz
        else:
            ckpts[int(round(epoch * steps_per_epoch))] = npz

    # Perf checkpoints — find first epoch where normalised perf >= threshold
    tm      = task_meta()[task]
    chance  = tm["chance_perf"]
    max_m   = tm["max_metric"]
    use_mse = tm["metric_name"] == "val_mse"

    def norm_perf(entry):
        raw = -entry["val_loss"] if use_mse else entry["val_acc"]
        return (raw - chance) / (max_m - chance)

    for npz in run_dir.glob("perf_*.npz"):
        threshold = float(npz.stem.replace("perf_", "").replace("p", "."))
        for entry in history:
            if norm_perf(entry) >= threshold:
                ckpts[entry["step"]] = npz
                break

    return ckpts


def process_network(run_dir, task, bo_perf):
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return None
    meta       = json.load(open(meta_path))
    depth      = get_depth_from_config(meta.get("config", {}))
    final_step = int(meta.get("final_step", 0))
    if final_step == 0:
        return None

    ckpts = collect_checkpoints(run_dir, task, meta)
    if len(ckpts) < 2:
        return None

    # Load activations for all checkpoints, sorted by step
    acts = {}
    for step, npz_path in sorted(ckpts.items()):
        a = load_last_hidden(npz_path, task, depth)
        if a is not None:
            acts[step] = a

    sorted_steps = sorted(acts.keys())
    if len(sorted_steps) < 2:
        return None

    perf = bo_perf.get(run_dir.name, float("nan"))
    rows = []
    for i in range(len(sorted_steps) - 1):
        s0, s1 = sorted_steps[i], sorted_steps[i + 1]
        a0, a1 = acts[s0], acts[s1]
        if a0.shape != a1.shape:
            continue
        interval = (s1 - s0) / final_step
        if interval < 0.01:
            continue
        r = spearman_finite(a0.flatten(), a1.flatten())
        if not np.isfinite(r):
            continue
        rows.append({
            "task":        task,
            "run_id":      run_dir.name,
            "step_start":  s0,
            "step_end":    s1,
            "t_end":       round(s1 / final_step, 6),
            "one_minus_r": round((1 - r) / interval, 6),
            "performance": perf,
            "final_step":  final_step,
        })
    return rows


def load_thresholds():
    data = json.load(open(TABLES_DIR / "success_thresholds.json"))
    return {k: float(v["upper"]) for k, v in data.items() if isinstance(v, dict)}


def make_figure(df, thresholds):
    tasks   = [t for t in TASK_NAMES if t in df["task"].values]
    n_cols  = 3
    n_rows  = int(np.ceil(len(tasks) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                              squeeze=False)

    for idx, task in enumerate(tasks):
        ax    = axes[idx // n_cols][idx % n_cols]
        tdf   = df[df["task"] == task]
        thresh = thresholds.get(task)

        groups = {}
        if thresh is not None:
            groups["successful"] = ("#2166ac", tdf[tdf["performance"] >= thresh])
            groups["failed"]     = ("#d73027", tdf[tdf["performance"] <  thresh])
        else:
            groups["all"] = ("#2166ac", tdf)

        # 6 log bins [0.01→0.4], 3 uniform bins [0.4,0.6,0.8,0.99], final point at 1.0
        bin_edges = np.concatenate([[0], np.geomspace(0.01, 0.4, 7), [0.6, 0.8, 0.99]])
        rng       = np.random.default_rng(42)

        for label, (color, gdf) in groups.items():
            if gdf.empty:
                continue

            # Individual curves — sample up to 100 networks
            all_run_ids = gdf["run_id"].unique()
            sample_ids  = rng.choice(all_run_ids,
                                     size=min(100, len(all_run_ids)),
                                     replace=False)
            for rid in sample_ids:
                net = gdf[gdf["run_id"] == rid].sort_values("t_end")
                ax.plot(net["t_end"], net["one_minus_r"],
                        color=color, linewidth=0.6, alpha=0.06, zorder=1)

            # Mean line — binned region + dedicated final point at t=1.0
            gdf = gdf.copy()
            gdf["bin"] = pd.cut(gdf["t_end"], bins=bin_edges, labels=False,
                                include_lowest=True)
            means, xs = [], []
            for b, bdf in gdf.groupby("bin"):
                means.append(bdf["one_minus_r"].mean())
                xs.append(bdf["t_end"].mean())

            # Final point: all data at exactly t_end == 1.0
            final = gdf[gdf["t_end"] == 1.0]
            if not final.empty:
                means.append(final["one_minus_r"].mean())
                xs.append(1.0)

            n_nets = gdf["run_id"].nunique()
            ax.plot(xs, means, color=color, linewidth=2.25, zorder=3,
                    label=f"{label} (n={n_nets})")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 10)
        ax.set_xlabel("Training time (fraction of total)", fontsize=7)
        ax.set_ylabel("(1 − Spearman r) / interval (rate of change)", fontsize=7)
        ax.set_title(task, fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

    for idx in range(len(tasks), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(
        "Rate of representational change: (1 − Spearman r) / interval\n"
        "Mean across networks (100 individual curves shown per group)",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Only recompute these tasks, upserting into existing CSV.")
    args = parser.parse_args()
    update_tasks = set(args.tasks) if args.tasks else None

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds()
    rows = []

    for task in (t for t in TASK_NAMES if update_tasks is None or t in update_tasks):
        task_dir = PROD_DIR / task
        if not task_dir.exists():
            print(f"  {task}: not found, skipping")
            continue
        run_dirs = sorted(p for p in task_dir.iterdir()
                          if p.is_dir() and p.name.endswith("_r0"))
        bo_perf = load_bo_perf(task)
        print(f"{task}: {len(run_dirs)} networks ...", flush=True)
        for run_dir in run_dirs:
            result = process_network(run_dir, task, bo_perf)
            if result:
                rows.extend(result)

    df = pd.DataFrame(rows)

    csv_out = TABLES_DIR / "critical_period.csv"
    if update_tasks and csv_out.exists():
        old = pd.read_csv(csv_out)
        old = old[~old["task"].isin(update_tasks)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")

    fig = make_figure(df, thresholds)
    fig_out = FIGURES_DIR / "critical_period.pdf"
    fig.savefig(fig_out, bbox_inches="tight", dpi=130)
    cp_final = FINAL_DIR / "learning_dynamics/figures/critical_period"
    cp_final.mkdir(parents=True, exist_ok=True)
    fig.savefig(cp_final / "critical_period.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_out}")


if __name__ == "__main__":
    main()
