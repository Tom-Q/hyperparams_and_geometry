#!/usr/bin/env python3
"""
Step 11b: Temporal RSA validity — Finding #1.1 and #1.2 per timestep/phase.

Extends 11_rsa_validity.py to compute noise ceiling and stochastic/HP-driven
variance at every timestep for MNIST RNN and at every semantic phase for Adding.

MNIST RNN: loops over (layer, timestep) pairs — layer 0 (all networks) and
           layer 1 (depth=2 networks only), timesteps 0..13.

Adding:    loops over phase_1..phase_6 at layer 0. Phase RDMs must exist in
           adding_rdms.h5 (run 10b_compute_adding_phases.py first).

Outputs:
    output/analysis/figures/f1_noise_ceiling_temporal.pdf
    output/analysis/tables/rdm_noise_ceiling_temporal.csv
    output/analysis/tables/rdm_variance_temporal.csv
"""

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
    DATASET_DIR, FIGURES_DIR, RDM_DIR, TABLES_DIR,
)

TASK_DIR_OVERRIDES = {}
N_BETWEEN = 2000
RNG_SEED  = 42

ADDING_PHASE_NAMES  = ["phase_1", "phase_2", "phase_3", "phase_4", "phase_5", "phase_6"]
ADDING_PHASE_LABELS = [
    "before\nflag₁", "at\nflag₁", "between\nflags",
    "at\nflag₂", "after\nflag₂", "final",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_thresholds():
    path = TABLES_DIR / "success_thresholds.json"
    if not path.exists():
        return {}
    data = json.load(open(path))
    return {k: float(v["upper"]) for k, v in data.items()
            if k != "_alpha" and isinstance(v, dict) and v.get("upper") is not None}


def load_bo_repeat_pairs(task):
    dirname = TASK_DIR_OVERRIDES.get(task, task)
    bo_path = DATASET_DIR / dirname / "bo_state.json"
    if not bo_path.exists():
        return []
    return [(int(o["repeat_of"]), int(o["iteration"]))
            for o in json.load(open(bo_path))
            if o.get("is_repeat") and o.get("repeat_of") is not None]


def noise_ceiling_loo(rdm_matrix):
    N = len(rdm_matrix)
    if N < 3:
        return np.full(N, np.nan)
    rdm_f = rdm_matrix.astype(np.float64)
    total = rdm_f.sum(axis=0)
    results = np.zeros(N)
    for i in range(N):
        loo = (total - rdm_f[i]) / (N - 1)
        r, _ = spearmanr(rdm_f[i], loo)
        results[i] = r
    return results


def rank_normalize_rows(mat):
    ranked = np.apply_along_axis(rankdata, 1, mat.astype(np.float64))
    ranked -= ranked.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(ranked, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1e-10, norms)
    return ranked / norms


def between_config_corrs(rdm_dict, n_pairs, rng):
    rdm_list = list(rdm_dict.values())
    N = len(rdm_list)
    if N < 2:
        return np.array([])
    rn = rank_normalize_rows(np.array(rdm_list))
    actual = min(n_pairs, N * (N - 1) // 2)
    ia = rng.integers(0, N, actual)
    ib = rng.integers(0, N, actual)
    same = ia == ib
    ib[same] = (ib[same] + 1) % N
    return (rn[ia] * rn[ib]).sum(axis=1)


# ---------------------------------------------------------------------------
# MNIST RNN — per (layer, timestep)
# ---------------------------------------------------------------------------

def load_mnist_rnn_key(layer, timestep, success_threshold):
    """
    Load RDMs for layer L at timestep T.
    Returns: primary_rdms, all_primary_rdms (unfiltered non-repeats), all_rdms (incl. repeats), run_perf
    """
    h5_path = RDM_DIR / "mnist_rnn_rdms.h5"
    key = f"layer_{layer}_t_{timestep}"
    all_rdms    = {}   # all runs including repeats
    all_primary = {}   # non-repeat, regardless of perf
    primary     = {}   # non-repeat, successful only
    run_perf    = {}

    with h5py.File(h5_path, "r") as h5:
        runs_grp = h5.get("runs", {})
        for run_id in sorted(runs_grp.keys()):
            rg = runs_grp[run_id]
            is_rep = bool(rg.attrs.get("is_repeat", False))
            perf = float(rg.attrs.get("performance", float("nan")))
            run_perf[run_id] = perf
            ckpt_grp = rg.get("best")
            if ckpt_grp is None or key not in ckpt_grp:
                continue
            ds = ckpt_grp[key]
            if ds.attrs.get("degenerate", False) or len(ds) == 0:
                continue
            rdm = ds[:].astype(np.float32)
            all_rdms[run_id] = rdm
            if not is_rep:
                all_primary[run_id] = rdm
                if success_threshold is None or perf >= success_threshold:
                    primary[run_id] = rdm

    return primary, all_primary, all_rdms, run_perf


def run_mnist_rnn_temporal(rng):
    """Compute noise ceiling and variance per (layer, timestep) for MNIST RNN."""
    threshold = load_thresholds().get("mnist_rnn")
    repeat_pairs = load_bo_repeat_pairs("mnist_rnn")

    # Discover available layers and timesteps from first successful run
    n_layers = 0
    n_t = 0
    h5_path = RDM_DIR / "mnist_rnn_rdms.h5"
    with h5py.File(h5_path, "r") as h5:
        runs_grp = h5.get("runs", {})
        for run_id in sorted(runs_grp.keys()):
            ckpt = runs_grp[run_id].get("best")
            if ckpt is None:
                continue
            for k in ckpt.keys():
                if "_t_" not in k:
                    continue
                parts = k.split("_")
                try:
                    n_layers = max(n_layers, int(parts[1]) + 1)
                    n_t      = max(n_t,      int(parts[3]) + 1)
                except (IndexError, ValueError):
                    pass
            break

    print(f"  mnist_rnn: {n_layers} layers, {n_t} timesteps")

    nc_rows  = []
    var_rows = []
    results  = {}   # (layer, t) -> {"nc": array, "within": array, "between": array}

    for L in range(n_layers):
        for T in range(n_t):
            primary, all_primary, all_rdms, run_perf = load_mnist_rnn_key(L, T, threshold)

            if len(primary) < 3:
                continue

            rdm_matrix = np.array(list(primary.values()), dtype=np.float32)
            run_ids    = list(primary.keys())
            nc_corrs   = noise_ceiling_loo(rdm_matrix)

            for rid, c in zip(run_ids, nc_corrs):
                nc_rows.append({
                    "task": "mnist_rnn", "layer": L, "timestep": T,
                    "run_id": rid, "performance": run_perf.get(rid, float("nan")),
                    "loo_spearman_r": float(c),
                })

            # Within-config pairs (primary original paired with its repeat)
            within_pairs = []
            for orig_iter, rep_iter in repeat_pairs:
                orig_id = f"run_{orig_iter:04d}_r0"
                rep_id  = f"run_{rep_iter:04d}_r0"
                if orig_id not in all_rdms or rep_id not in all_rdms:
                    continue
                if threshold is None or (
                    run_perf.get(orig_id, float("nan")) >= threshold and
                    run_perf.get(rep_id,  float("nan")) >= threshold
                ):
                    within_pairs.append(
                        (all_rdms[orig_id], all_rdms[rep_id]))

            within_corrs  = (np.array([spearmanr(a, b)[0] for a, b in within_pairs])
                             if within_pairs else np.array([]))
            between_corrs = between_config_corrs(primary, N_BETWEEN, rng)

            assert not np.any(~np.isfinite(within_corrs)), \
                f"mnist_rnn L={L} T={T}: NaN in within_corrs"

            results[(L, T)] = {
                "nc": nc_corrs,
                "within": within_corrs,
                "between": between_corrs,
            }
            for r in within_corrs:
                var_rows.append({"task": "mnist_rnn", "layer": L, "timestep": T,
                                 "pair_type": "within_config", "spearman_r": float(r)})
            for r in between_corrs:
                var_rows.append({"task": "mnist_rnn", "layer": L, "timestep": T,
                                 "pair_type": "between_config", "spearman_r": float(r)})

    return results, nc_rows, var_rows, n_layers, n_t


# ---------------------------------------------------------------------------
# Adding — per phase
# ---------------------------------------------------------------------------

def load_adding_phase_key(phase_name, success_threshold):
    """Load phase RDMs for adding (includes repeats for within-config pairing)."""
    h5_path = RDM_DIR / "adding_rdms.h5"
    key = f"layer_0_{phase_name}"
    all_rdms    = {}
    all_primary = {}
    primary     = {}
    run_perf    = {}

    with h5py.File(h5_path, "r") as h5:
        runs_grp = h5.get("runs", {})
        for run_id in sorted(runs_grp.keys()):
            rg = runs_grp[run_id]
            is_rep = bool(rg.attrs.get("is_repeat", False))
            perf = float(rg.attrs.get("performance", float("nan")))
            run_perf[run_id] = perf
            ckpt_grp = rg.get("best")
            if ckpt_grp is None or key not in ckpt_grp:
                continue
            ds = ckpt_grp[key]
            if ds.attrs.get("degenerate", False) or len(ds) == 0:
                continue
            rdm = ds[:].astype(np.float32)
            all_rdms[run_id] = rdm
            if not is_rep:
                all_primary[run_id] = rdm
                if success_threshold is None or perf >= success_threshold:
                    primary[run_id] = rdm

    return primary, all_primary, all_rdms, run_perf


def run_adding_phases(rng):
    """Compute noise ceiling and variance per phase for Adding."""
    threshold    = load_thresholds().get("adding")
    repeat_pairs = load_bo_repeat_pairs("adding")

    # Check phase RDMs exist (need to have run 10b first)
    h5_path = RDM_DIR / "adding_rdms.h5"
    try:
        with h5py.File(h5_path, "r") as h5:
            sample_run = sorted(h5.get("runs", {}).keys())[0]
            ckpt       = h5["runs"][sample_run].get("best")
            has_phases = ckpt is not None and "layer_0_phase_1" in ckpt
    except BlockingIOError:
        print("  [skip adding] adding_rdms.h5 is locked (10b still running?)")
        return {}, [], []
    if not has_phases:
        print("  [skip adding] phase RDMs not found — run 10b_compute_adding_phases.py first")
        return {}, [], []

    nc_rows  = []
    var_rows = []
    results  = {}

    for pname in ADDING_PHASE_NAMES:
        primary, all_primary, all_rdms, run_perf = load_adding_phase_key(pname, threshold)

        if len(primary) < 3:
            print(f"    {pname}: only {len(primary)} networks, skipping")
            continue

        rdm_matrix = np.array(list(primary.values()), dtype=np.float32)
        run_ids    = list(primary.keys())
        nc_corrs   = noise_ceiling_loo(rdm_matrix)

        for rid, c in zip(run_ids, nc_corrs):
            nc_rows.append({
                "task": "adding", "phase": pname,
                "run_id": rid, "performance": run_perf.get(rid, float("nan")),
                "loo_spearman_r": float(c),
            })

        within_pairs = []
        for orig_iter, rep_iter in repeat_pairs:
            orig_id = f"run_{orig_iter:04d}_r0"
            rep_id  = f"run_{rep_iter:04d}_r0"
            if orig_id not in all_rdms or rep_id not in all_rdms:
                continue
            if threshold is None or (
                run_perf.get(orig_id, float("nan")) >= threshold and
                run_perf.get(rep_id,  float("nan")) >= threshold
            ):
                within_pairs.append(
                    (all_rdms[orig_id], all_rdms[rep_id]))

        within_corrs  = (np.array([spearmanr(a, b)[0] for a, b in within_pairs])
                         if within_pairs else np.array([]))
        between_corrs = between_config_corrs(primary, N_BETWEEN, rng)

        assert not np.any(~np.isfinite(within_corrs)), \
            f"adding {pname}: NaN in within_corrs"

        results[pname] = {
            "nc": nc_corrs,
            "within": within_corrs,
            "between": between_corrs,
        }
        for r in within_corrs:
            var_rows.append({"task": "adding", "phase": pname,
                             "pair_type": "within_config", "spearman_r": float(r)})
        for r in between_corrs:
            var_rows.append({"task": "adding", "phase": pname,
                             "pair_type": "between_config", "spearman_r": float(r)})

        nc_mean = np.nanmean(nc_corrs)
        mw = np.median(within_corrs) if len(within_corrs) else float("nan")
        mb = np.median(between_corrs) if len(between_corrs) else float("nan")
        print(f"    {pname}: N={len(primary)}, nc={nc_mean:.3f}, "
              f"within={mw:.3f}, between={mb:.3f}")

    return results, nc_rows, var_rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _stat_curve(results_dict, keys, stat_fn):
    """Extract a statistic curve from a dict of result arrays."""
    return np.array([stat_fn(results_dict[k]) if k in results_dict else np.nan
                     for k in keys])


def plot_temporal_figure(rnn_results, adding_results, n_layers, n_t):
    """Single figure with three panels: nc curve, within/between for each task."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    ts = np.arange(n_t)

    layer_styles = {
        0: ("-",  "#2166ac", "layer 0 (all)"),
        1: ("--", "#d6604d", "layer 1 (depth=2)"),
    }

    # ---- Row 0: MNIST RNN ----
    # Panel 0: noise ceiling per (layer, timestep)
    ax = axes[0, 0]
    for L in range(n_layers):
        sty, col, lbl = layer_styles.get(L, ("-", "grey", f"layer {L}"))
        medians, lo, hi = [], [], []
        for T in ts:
            nc = rnn_results.get((L, T), {}).get("nc", np.array([]))
            if len(nc) > 0:
                medians.append(np.median(nc))
                lo.append(np.percentile(nc, 25))
                hi.append(np.percentile(nc, 75))
            else:
                medians.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        med = np.array(medians); lo = np.array(lo); hi = np.array(hi)
        ax.fill_between(ts, lo, hi, alpha=0.15, color=col)
        ax.plot(ts, med, sty, color=col, lw=1.5, ms=4, label=lbl)
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.set_title("MNIST RNN — noise ceiling per timestep", fontsize=9, fontweight="bold")
    ax.set_xlabel("timestep", fontsize=8); ax.set_ylabel("LOO Spearman r", fontsize=8)
    ax.legend(fontsize=7); ax.set_ylim(-0.15, 1.0); ax.tick_params(labelsize=7)

    # Panel 1: within-config per timestep (last layer only)
    ax = axes[0, 1]
    for L in range(n_layers):
        sty, col, lbl = layer_styles.get(L, ("-", "grey", f"layer {L}"))
        meds = []
        for T in ts:
            w = rnn_results.get((L, T), {}).get("within", np.array([]))
            meds.append(np.median(w) if len(w) > 0 else np.nan)
        ax.plot(ts, meds, sty, color=col, lw=1.4, ms=4, label=f"{lbl} within")
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.set_title("MNIST RNN — within-config Spearman r", fontsize=9, fontweight="bold")
    ax.set_xlabel("timestep", fontsize=8); ax.set_ylabel("Spearman r", fontsize=8)
    ax.legend(fontsize=7); ax.set_ylim(-0.15, 1.0); ax.tick_params(labelsize=7)

    # Panel 2: between-config per timestep (last layer)
    ax = axes[0, 2]
    for L in range(n_layers):
        sty, col, lbl = layer_styles.get(L, ("-", "grey", f"layer {L}"))
        meds = []
        for T in ts:
            b = rnn_results.get((L, T), {}).get("between", np.array([]))
            meds.append(np.median(b) if len(b) > 0 else np.nan)
        ax.plot(ts, meds, sty, color=col, lw=1.4, ms=4, label=f"{lbl} between")
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.set_title("MNIST RNN — between-config Spearman r", fontsize=9, fontweight="bold")
    ax.set_xlabel("timestep", fontsize=8); ax.set_ylabel("Spearman r", fontsize=8)
    ax.legend(fontsize=7); ax.set_ylim(-0.15, 1.0); ax.tick_params(labelsize=7)

    # ---- Row 1: Adding phases ----
    phase_x     = np.arange(len(ADDING_PHASE_NAMES))
    phase_color = "#4dac26"

    def _phase_stat(key, stat):
        vals = [adding_results.get(p, {}).get(key, np.array([]))
                for p in ADDING_PHASE_NAMES]
        return np.array([stat(v) if len(v) > 0 else np.nan for v in vals])

    # Panel 3: noise ceiling per phase
    ax = axes[1, 0]
    med = _phase_stat("nc", np.median)
    lo  = _phase_stat("nc", lambda v: np.percentile(v, 25))
    hi  = _phase_stat("nc", lambda v: np.percentile(v, 75))
    ax.fill_between(phase_x, lo, hi, alpha=0.2, color=phase_color)
    ax.plot(phase_x, med, "o-", color=phase_color, lw=1.5, ms=5)
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.set_xticks(phase_x); ax.set_xticklabels(ADDING_PHASE_LABELS, fontsize=7)
    ax.set_title("Adding — noise ceiling per phase", fontsize=9, fontweight="bold")
    ax.set_ylabel("LOO Spearman r", fontsize=8); ax.set_ylim(-0.15, 1.0)
    ax.tick_params(labelsize=7)

    # Panel 4: within-config per phase
    ax = axes[1, 1]
    mw = _phase_stat("within", np.median)
    ax.plot(phase_x, mw, "o-", color="#2166ac", lw=1.4, ms=5, label="within-config")
    mb = _phase_stat("between", np.median)
    ax.plot(phase_x, mb, "s--", color="#d6604d", lw=1.4, ms=4, label="between-config")
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.set_xticks(phase_x); ax.set_xticklabels(ADDING_PHASE_LABELS, fontsize=7)
    ax.set_title("Adding — within vs. between variance", fontsize=9, fontweight="bold")
    ax.set_ylabel("median Spearman r", fontsize=8); ax.set_ylim(-0.25, 1.0)
    ax.legend(fontsize=7); ax.tick_params(labelsize=7)

    # Panel 5: HP-driven gap (between − within) per phase
    ax = axes[1, 2]
    gap = mb - mw
    colors_gap = [("#d6604d" if g > 0 else "#4393c3") for g in gap]
    ax.bar(phase_x, gap, color=colors_gap, width=0.5, alpha=0.7)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xticks(phase_x); ax.set_xticklabels(ADDING_PHASE_LABELS, fontsize=7)
    ax.set_title("Adding — HP-driven gap (between − within)", fontsize=9, fontweight="bold")
    ax.set_ylabel("Δ median Spearman r", fontsize=8)
    ax.tick_params(labelsize=7)

    fig.suptitle("Temporal RSA validity — noise ceiling and variance per timestep/phase\n"
                 "(successful primary networks, best checkpoint)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    # MNIST RNN temporal
    print("MNIST RNN temporal analysis ...")
    rnn_results, rnn_nc_rows, rnn_var_rows, n_layers, n_t = run_mnist_rnn_temporal(rng)
    print(f"  {len(rnn_results)} (layer, timestep) keys computed")

    # Adding phases
    print("\nAdding phase analysis ...")
    adding_results, adding_nc_rows, adding_var_rows = run_adding_phases(rng)
    print(f"  {len(adding_results)} phases computed")

    # Save tables
    all_nc_rows  = rnn_nc_rows  + adding_nc_rows
    all_var_rows = rnn_var_rows + adding_var_rows

    nc_csv  = TABLES_DIR / "rdm_noise_ceiling_temporal.csv"
    var_csv = TABLES_DIR / "rdm_variance_temporal.csv"
    pd.DataFrame(all_nc_rows).to_csv(nc_csv, index=False)
    pd.DataFrame(all_var_rows).to_csv(var_csv, index=False)
    print(f"\nSaved: {nc_csv}")
    print(f"Saved: {var_csv}")

    # Figure
    if rnn_results or adding_results:
        fig = plot_temporal_figure(rnn_results, adding_results, n_layers, n_t)
        out = FIGURES_DIR / "f1_noise_ceiling_temporal.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
