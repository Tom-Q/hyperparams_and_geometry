#!/usr/bin/env python3
"""
Step 11: RSA validity — Finding #1.1 (noise ceiling) and #1.2 (stochastic vs HP variance).

1.1 Noise ceiling
    For each task, load last-hidden-layer RDMs from successful primary networks at
    the best/final checkpoint. Compute leave-one-out Spearman correlation of each
    network's RDM with the group mean RDM.

1.2 Stochastic vs. HP-driven variance
    Within-config: Spearman correlation between repeat pairs (same config, different seed).
    Between-config: Spearman correlation between randomly sampled primary-network pairs.

Outputs:
    output/analysis/figures/f1_noise_ceiling.pdf
    output/analysis/figures/f1_variance_decomposition.pdf
    output/analysis/tables/rdm_noise_ceiling.csv   (per-network reliability, reused in Finding #2)
    output/analysis/tables/rdm_variance.csv
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ANALYSIS = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import (
    DATASET_DIR, FIGURES_DIR, RDM_DIR, TABLES_DIR, TASK_NAMES, RL_TASKS,
    metric_output_dirs,
)

TASK_DIR_OVERRIDES = {}
RNN_TASKS = {"adding", "mnist_rnn"}
NAN_TASKS = {"adding"}    # temporal RDMs contain fixed-position NaN entries
N_BETWEEN = 2000   # between-config pairs to sample for 1.2
RNG_SEED  = 42

TASK_LABELS = {
    "mnist_dual":    "MNIST dual",
    "mnist_10way":   "MNIST 10-way",
    "fashion_10way": "Fashion 10-way",
    "spirals":       "Spirals",
    "parity":        "Parity",
    "adding":        "Adding (ref)",
    "mnist_rnn":     "MNIST RNN",
    "cartpole":      "CartPole",
    "fourrooms":     "FourRooms",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_thresholds():
    """Return dict task → upper threshold (float), or {} if not set."""
    path = TABLES_DIR / "success_thresholds.json"
    if not path.exists():
        return {}
    data = json.load(open(path))
    result = {}
    for task, vals in data.items():
        if task == "_alpha":
            continue
        if isinstance(vals, dict) and vals.get("upper") is not None:
            result[task] = float(vals["upper"])
    return result


def get_ckpt_name(task):
    return "final" if task in RL_TASKS else "best"


def get_last_layer_key(ckpt_grp, task, depth, metric="cosine"):
    """Key for the primary RDM in a checkpoint group."""
    if task in RNN_TASKS:
        return f"temporal_{metric}"
    return f"layer_{max(0, int(depth) - 1)}_{metric}"


def load_rdm_vec(ckpt_grp, key):
    """Return float32 RDM vector, or None if missing/degenerate."""
    if key not in ckpt_grp:
        return None
    ds = ckpt_grp[key]
    if ds.attrs.get("degenerate", False) or len(ds) == 0:
        return None
    return ds[:].astype(np.float32)


def load_bo_repeat_pairs(task):
    """Return list of (orig_iter, rep_iter) from bo_state.json."""
    dirname = TASK_DIR_OVERRIDES.get(task, task)
    bo_path = DATASET_DIR / dirname / "bo_state.json"
    if not bo_path.exists():
        return []
    pairs = []
    for obs in json.load(open(bo_path)):
        if obs.get("is_repeat") and obs.get("repeat_of") is not None:
            pairs.append((int(obs["repeat_of"]), int(obs["iteration"])))
    return pairs


def load_task(task, success_threshold=None, metric="cosine"):
    """
    Load RDMs from HDF5 for one task.

    Returns:
        primary_rdms : dict run_id → rdm  (primary, threshold-filtered for 1.1)
        all_primary  : dict run_id → rdm  (all primary, unfiltered, for 1.2 between-config)
        within_pairs : list of (rdm_orig, rdm_rep)  (for 1.2 within-config)
        run_perf     : dict run_id → performance
    """
    h5_path = RDM_DIR / f"{task}_rdms.h5"
    if not h5_path.exists():
        print(f"  [skip] {task}: no HDF5 file")
        return {}, {}, [], {}

    ckpt = get_ckpt_name(task)
    all_rdms = {}
    run_perf = {}
    run_is_repeat = {}

    print(f"  loading {task} ...", end="", flush=True)
    with h5py.File(h5_path, "r") as h5:
        runs_grp = h5.get("runs")
        if runs_grp is None:
            print(" [no runs group]")
            return {}, {}, [], {}
        for run_id in sorted(runs_grp.keys()):
            rg = runs_grp[run_id]
            depth = int(rg.attrs.get("hp_depth", 1))
            ckpt_grp = rg.get(ckpt)
            if ckpt_grp is None:
                continue
            key = get_last_layer_key(ckpt_grp, task, depth, metric=metric)
            if key is None:
                continue
            rdm = load_rdm_vec(ckpt_grp, key)
            if rdm is None:
                continue
            all_rdms[run_id] = rdm
            run_perf[run_id] = float(rg.attrs.get("performance", float("nan")))
            run_is_repeat[run_id] = bool(rg.attrs.get("is_repeat", False))

    print(f" {len(all_rdms)} RDMs loaded")

    # Strip fixed-position NaN for NAN_TASKS (adding); raise on unexpected NaN elsewhere.
    # Adding's NaN mask is identical across all networks — strip once using the first network.
    if task in NAN_TASKS and all_rdms:
        sample = next(iter(all_rdms.values()))
        valid  = np.isfinite(sample)
        all_rdms = {rid: rdm[valid] for rid, rdm in all_rdms.items()}
    elif all_rdms:
        for rid, rdm in all_rdms.items():
            if not np.all(np.isfinite(rdm)):
                raise ValueError(
                    f"{task}/{rid}: unexpected NaN in RDM — bug in compute script")

    all_primary = {rid: rdm for rid, rdm in all_rdms.items()
                   if not run_is_repeat.get(rid, False)}

    if success_threshold is not None:
        primary_rdms = {rid: rdm for rid, rdm in all_primary.items()
                        if run_perf.get(rid, float("nan")) >= success_threshold}
    else:
        primary_rdms = dict(all_primary)

    # Build within-config pairs: both members must be successful
    repeat_iters = load_bo_repeat_pairs(task)
    within_pairs = []
    for orig_iter, rep_iter in repeat_iters:
        orig_id = f"run_{orig_iter:04d}_r0"
        rep_id  = f"run_{rep_iter:04d}_r0"
        if orig_id not in all_rdms or rep_id not in all_rdms:
            continue
        if success_threshold is not None:
            if (run_perf.get(orig_id, float("nan")) < success_threshold or
                    run_perf.get(rep_id,  float("nan")) < success_threshold):
                continue
        within_pairs.append((all_rdms[orig_id], all_rdms[rep_id]))

    return primary_rdms, all_primary, within_pairs, run_perf


# ---------------------------------------------------------------------------
# 1.1 Noise ceiling
# ---------------------------------------------------------------------------

def noise_ceiling_loo(rdm_matrix):
    """
    Spearman(rdm_i, mean_{j≠i}(rdm_j)) for each network i.
    rdm_matrix : (N, D) float32 or float64
    Returns float64 array of length N.
    """
    N = len(rdm_matrix)
    if N < 3:
        return np.full(N, np.nan)
    rdm_f32 = np.asarray(rdm_matrix, dtype=np.float32)
    # Sum in float64 without materialising the full N×D float64 matrix
    group_sum = rdm_f32.sum(axis=0, dtype=np.float64)
    results = np.zeros(N)
    for i in range(N):
        vec_i    = rdm_f32[i].astype(np.float64)
        loo_mean = (group_sum - vec_i) / (N - 1)
        r, _     = spearmanr(vec_i, loo_mean)
        results[i] = r
    return results


# ---------------------------------------------------------------------------
# 1.2 Variance decomposition
# ---------------------------------------------------------------------------

def rank_normalize_rows(mat):
    """Rank each row of mat and normalize to unit vector. Returns float64."""
    ranked = np.apply_along_axis(rankdata, 1, mat.astype(np.float64))
    ranked -= ranked.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(ranked, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1e-10, norms)
    return ranked / norms


def between_config_corrs(all_primary, n_pairs, rng):
    """
    Sample n_pairs random distinct pairs from all_primary and return their
    Spearman correlations. Uses pairwise spearmanr to avoid an N×D float64
    rank matrix (which would OOM for large temporal RDMs).
    """
    rdm_list = list(all_primary.values())
    N = len(rdm_list)
    if N < 2:
        return np.array([])
    actual = min(n_pairs, N * (N - 1) // 2)
    ia = rng.integers(0, N, actual)
    ib = rng.integers(0, N, actual)
    same = ia == ib
    ib[same] = (ib[same] + 1) % N
    return np.array([spearmanr(rdm_list[a], rdm_list[b])[0]
                     for a, b in zip(ia, ib)], dtype=np.float64)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_noise_ceiling(nc_results, thresholds):
    tasks = [t for t in TASK_NAMES if t in nc_results]
    ncols, nrows = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 9))
    axes = axes.flatten()

    for ax, task in zip(axes, tasks):
        corrs = nc_results[task]
        vp = ax.violinplot([corrs], positions=[0], showmedians=True, showextrema=True)
        vp["bodies"][0].set_facecolor("#4393c3")
        vp["bodies"][0].set_alpha(0.6)
        mean_r = np.mean(corrs)
        ax.scatter([0], [mean_r], color="#d6604d", s=40, zorder=5)
        ax.axhline(0, color="grey", lw=0.7, ls="--")
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.15, 1.08)
        ax.set_xticks([])
        thresh = thresholds.get(task)
        thresh_str = f"perf ≥ {thresh:.3f}" if thresh is not None else "all primary"
        ax.set_title(f"{TASK_LABELS.get(task, task)}\nn={len(corrs)}, {thresh_str}",
                     fontsize=9)
        ax.set_ylabel("LOO Spearman r", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.text(0.02, 0.97, f"mean={mean_r:.3f}", transform=ax.transAxes,
                fontsize=8, va="top", color="#d6604d")

    for ax in axes[len(tasks):]:
        ax.set_visible(False)

    fig.suptitle(
        "Noise ceiling — LOO Spearman r with group mean RDM\n"
        "(last hidden layer, best/final checkpoint, successful primary networks)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_variance_decomp(var_results):
    tasks = [t for t in TASK_NAMES if t in var_results]
    ncols, nrows = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 9))
    axes = axes.flatten()

    for ax, task in zip(axes, tasks):
        r = var_results[task]
        within = r["within"]
        between = r["between"]

        if len(within) == 0 and len(between) == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.set_title(TASK_LABELS.get(task, task), fontsize=9)
            continue

        data = [d for d in [within, between] if len(d) > 0]
        pos = [i for i, d in enumerate([within, between]) if len(d) > 0]

        parts = ax.violinplot(data, positions=pos, showmedians=True, showextrema=True)
        colors = ["#2166ac", "#d6604d"]
        for body, color in zip(parts["bodies"], [colors[p] for p in pos]):
            body.set_facecolor(color)
            body.set_alpha(0.6)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            ["within-config\n(stochastic)", "between-config\n(HP-driven)"],
            fontsize=7)
        ax.set_ylim(-0.25, 1.1)
        ax.set_ylabel("Spearman r", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axhline(0, color="grey", lw=0.7, ls="--")

        med_w = np.median(within) if len(within) > 0 else float("nan")
        med_b = np.median(between) if len(between) > 0 else float("nan")
        gap = med_b - med_w
        ax.set_title(
            f"{TASK_LABELS.get(task, task)}\n"
            f"within={med_w:.3f}  between={med_b:.3f}  Δ={gap:+.3f}",
            fontsize=8.5)

    for ax in axes[len(tasks):]:
        ax.set_visible(False)

    fig.suptitle(
        "Stochastic vs. HP-driven representational variance\n"
        "(within-config = same HP, different seed; between-config = random primary pairs)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RSA validity — noise ceiling and variance decomposition.")
    parser.add_argument("--task", nargs="+", default=None,
                        help="Tasks to process (default: all).")
    parser.add_argument("--metric", choices=["cosine", "pearson"], default="cosine",
                        help="RDM metric to use (default: cosine).")
    args = parser.parse_args()

    out_figures, out_tables = metric_output_dirs(args.metric)
    out_figures.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds()
    if thresholds:
        print("Success thresholds loaded.")
    else:
        print("No thresholds found — using all primary networks for 1.1.")

    rng = np.random.default_rng(RNG_SEED)
    tasks = args.task if args.task else TASK_NAMES

    nc_results  = {}   # task → float array of LOO correlations
    var_results = {}   # task → {"within": array, "between": array}
    nc_rows     = []
    var_rows    = []

    for task in tasks:
        if task not in TASK_NAMES:
            print(f"[warn] unknown task '{task}', skipping")
            continue

        threshold = thresholds.get(task)
        primary_rdms, all_primary, within_pairs, run_perf = load_task(task, threshold, metric=args.metric)

        # --- 1.1 Noise ceiling ---
        if len(primary_rdms) < 3:
            print(f"  [skip 1.1] {task}: {len(primary_rdms)} networks after filtering")
        else:
            rdm_matrix = np.array(list(primary_rdms.values()), dtype=np.float32)
            run_ids    = list(primary_rdms.keys())
            nc_corrs   = noise_ceiling_loo(rdm_matrix)
            nc_results[task] = nc_corrs
            print(f"  [1.1] {task}: N={len(primary_rdms)}, "
                  f"mean LOO r = {np.nanmean(nc_corrs):.4f}")
            for rid, corr in zip(run_ids, nc_corrs):
                nc_rows.append({
                    "task":        task,
                    "run_id":      rid,
                    "performance": run_perf.get(rid, float("nan")),
                    "loo_spearman_r": float(corr),
                })

        # --- 1.2 Variance decomposition ---
        within_corrs  = (np.array([spearmanr(a, b)[0] for a, b in within_pairs])
                         if within_pairs else np.array([]))
        between_corrs = between_config_corrs(primary_rdms, N_BETWEEN, rng)

        if np.any(~np.isfinite(within_corrs)):
            bad = np.where(~np.isfinite(within_corrs))[0].tolist()
            raise ValueError(f"{task}: NaN/Inf in within_corrs at indices {bad} — "
                             "check for constant RDMs in successful networks")
        if np.any(~np.isfinite(between_corrs)):
            raise ValueError(f"{task}: NaN/Inf in between_corrs — check RDM loading")

        var_results[task] = {"within": within_corrs, "between": between_corrs}

        med_w = f"{np.median(within_corrs):.3f}" if len(within_corrs) else "—"
        med_b = f"{np.median(between_corrs):.3f}" if len(between_corrs) else "—"
        print(f"  [1.2] {task}: {len(within_corrs)} successful repeat pairs, "
              f"within={med_w}, between={med_b}")

        for r in within_corrs:
            var_rows.append({"task": task, "pair_type": "within_config", "spearman_r": float(r)})
        for r in between_corrs:
            var_rows.append({"task": task, "pair_type": "between_config", "spearman_r": float(r)})

    # --- Save tables ---
    nc_csv = out_tables / "rdm_noise_ceiling.csv"
    pd.DataFrame(nc_rows).to_csv(nc_csv, index=False)
    print(f"\nSaved: {nc_csv}")

    var_csv = out_tables / "rdm_variance.csv"
    pd.DataFrame(var_rows).to_csv(var_csv, index=False)
    print(f"Saved: {var_csv}")

    # --- Save figures ---
    if nc_results:
        fig_nc = plot_noise_ceiling(nc_results, thresholds)
        out_nc = out_figures / "f1_noise_ceiling.pdf"
        fig_nc.savefig(out_nc, bbox_inches="tight")
        plt.close(fig_nc)
        print(f"Saved: {out_nc}")

    if var_results:
        fig_var = plot_variance_decomp(var_results)
        out_var = out_figures / "f1_variance_decomposition.pdf"
        fig_var.savefig(out_var, bbox_inches="tight")
        plt.close(fig_var)
        print(f"Saved: {out_var}")


if __name__ == "__main__":
    main()
