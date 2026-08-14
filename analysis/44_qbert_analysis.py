#!/usr/bin/env python3
"""
Q*bert representational geometry analysis.

Covers:
  1.1 Noise ceiling  — LOO cross-network RDM agreement (functional networks)
  1.4 Layer comparison — layer_0 vs layer_1 for depth=2 networks
  1.5 Dimensionality — participation ratio from raw activations
  3   Dynamics — crystallisation (r to final RDM) vs gradient update step
  Q   Architectural effects — cross-group vs within-group RDM agreement
      (attention on/off, residual on/off, depth 1 vs 2)

Usage:
    python analysis/44_qbert_analysis.py

Outputs:
    output/analysis/figures/qbert_noise_ceiling.pdf
    output/analysis/figures/qbert_layer_comparison.pdf
    output/analysis/figures/qbert_dimensionality.pdf
    output/analysis/figures/qbert_dynamics.pdf
    output/analysis/figures/qbert_arch_effects.pdf
    output/analysis/tables/qbert_noise_ceiling.csv
    output/analysis/tables/qbert_dimensionality.csv
    output/analysis/tables/qbert_dynamics.csv
"""

import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

ANALYSIS  = Path(__file__).parent
REPO_ROOT = ANALYSIS.parent
sys.path.insert(0, str(ANALYSIS))

from analysis_utils import FIGURES_DIR, RDM_DIR, TABLES_DIR

H5_PATH = RDM_DIR / "qbert_rdms.h5"
ACT_DIR = REPO_ROOT / "output" / "production" / "qbert"
FIG_DIR = FIGURES_DIR
TAB_DIR = TABLES_DIR

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_rdm(grp, layer, metric="pearson"):
    key = f"layer_{layer}_{metric}"
    ds = grp.get(key)
    if ds is None or ds.attrs.get("degenerate", False) or len(ds) == 0:
        return None
    return ds[:].astype(np.float64)


def spearman_r(a, b):
    if a is None or b is None:
        return np.nan
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    return float(spearmanr(a[mask], b[mask]).statistic)


def participation_ratio(acts):
    if not np.all(np.isfinite(acts)):
        return np.nan
    X_c = acts - acts.mean(axis=0)
    G   = X_c @ X_c.T
    lam = np.linalg.eigvalsh(G)
    lam = lam[lam > 1e-10 * lam.max()]
    if len(lam) == 0 or lam.sum() < 1e-12:
        return np.nan
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def step_from_ckpt_name(name):
    """'step_0001024' → 1024"""
    return int(name.split("_")[1])


def load_runs_meta(h5):
    rows = []
    for run_id, rg in h5["runs"].items():
        a = dict(rg.attrs)
        rows.append({
            "run_id":          run_id,
            "iteration":       int(a["iteration"]),
            "is_functional":   bool(a["is_functional"]),
            "depth":           int(a.get("hp_depth", 1)),
            "hidden_size":     int(a.get("hp_hidden_size", 0)),
            "use_attention":   bool(a.get("hp_use_attention", False)),
            "use_residual":    bool(a.get("hp_use_residual", False)),
            "performance":     float(a.get("performance", np.nan)),
            "max_frac_level5": float(a.get("max_frac_level5", 0.0)),
            "stop_reason":     str(a.get("stop_reason", "")),
        })
    return rows


def last_layer(depth):
    return depth - 1


def loo_noise_ceiling(rdm_dict):
    """LOO Spearman r for each network in rdm_dict. Returns list of r values."""
    run_ids = list(rdm_dict.keys())
    mat     = np.stack([rdm_dict[r] for r in run_ids])
    total   = mat.sum(axis=0)
    return [spearman_r(mat[i], (total - mat[i]) / (len(run_ids) - 1))
            for i in range(len(run_ids))]


# ── 1.1 Noise ceiling ─────────────────────────────────────────────────────────

def analysis_noise_ceiling(h5):
    print("\n── 1.1 Noise ceiling ──────────────────────────────────────────────")
    meta        = load_runs_meta(h5)
    functional  = [m for m in meta if m["is_functional"]]
    print(f"  Functional networks: {len(functional)}")

    all_rows = []
    for metric in ("pearson", "cosine"):
        rdms = {}
        for m in functional:
            rid   = m["run_id"]
            layer = last_layer(m["depth"])
            ckpt  = h5["runs"][rid].get("final") or h5["runs"][rid].get("best")
            if ckpt is None:
                continue
            rdm = load_rdm(ckpt, layer, metric)
            if rdm is not None:
                rdms[rid] = rdm

        if len(rdms) < 2:
            print(f"  [{metric}] Not enough runs — skip")
            continue

        loos = loo_noise_ceiling(rdms)
        for rid, r in zip(rdms.keys(), loos):
            all_rows.append({"run_id": rid, "metric": metric, "loo_r": r})
        print(f"  [{metric}] LOO r = {np.mean(loos):.3f} ± {np.std(loos):.3f}  "
              f"[{np.min(loos):.3f}, {np.max(loos):.3f}]")

    df = pd.DataFrame(all_rows)
    df.to_csv(TAB_DIR / "qbert_noise_ceiling.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for metric, color in [("pearson", "#2166ac"), ("cosine", "#d73027")]:
        vals = df.query("metric == @metric")["loo_r"].values
        ax.scatter([metric] * len(vals), vals, alpha=0.6, s=30, color=color, zorder=3)
        ax.plot([metric], [vals.mean()], marker="D", ms=8, color=color, zorder=4,
                label=f"{metric}  mean={vals.mean():.3f}")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_ylabel("LOO Spearman r")
    ax.set_title("Q*bert noise ceiling (functional networks, final checkpoint)")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "qbert_noise_ceiling.pdf")
    plt.close(fig)
    print(f"  Saved: qbert_noise_ceiling.pdf")


# ── 1.4 Layer comparison ──────────────────────────────────────────────────────

def analysis_layer_comparison(h5):
    print("\n── 1.4 Layer comparison (depth=2 networks) ────────────────────────")
    meta   = load_runs_meta(h5)
    depth2 = [m for m in meta if m["depth"] == 2]
    print(f"  depth=2 networks: {len(depth2)}")

    rows = []
    for metric in ("pearson", "cosine"):
        l0_rdms, l1_rdms = {}, {}
        for m in depth2:
            rid  = m["run_id"]
            ckpt = h5["runs"][rid].get("final") or h5["runs"][rid].get("best")
            if ckpt is None:
                continue
            r0 = load_rdm(ckpt, 0, metric)
            r1 = load_rdm(ckpt, 1, metric)
            if r0 is not None and r1 is not None:
                l0_rdms[rid] = r0
                l1_rdms[rid] = r1
                rows.append({"run_id": rid, "metric": metric,
                             "l0_l1_r": spearman_r(r0, r1),
                             "is_functional": m["is_functional"]})

        if len(l0_rdms) < 2:
            continue

        for lrdms, tag in [(l0_rdms, "L0"), (l1_rdms, "L1")]:
            nc = loo_noise_ceiling(lrdms)
            print(f"  [{metric}] {tag} noise ceiling = {np.mean(nc):.3f} ± {np.std(nc):.3f}")

        l01 = [r["l0_l1_r"] for r in rows if r["metric"] == metric]
        print(f"  [{metric}] L0–L1 similarity = {np.mean(l01):.3f} ± {np.std(l01):.3f}")

    if not rows:
        return
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    for ax, metric, color in zip(axes, ("pearson", "cosine"), ("#2166ac", "#d73027")):
        sub = df.query("metric == @metric").reset_index(drop=True)
        fc  = ["#1a9641" if f else "#d73027" for f in sub["is_functional"]]
        ax.bar(range(len(sub)), sub["l0_l1_r"], color=fc, alpha=0.75)
        ax.axhline(sub["l0_l1_r"].mean(), color=color, lw=1.5, ls="--",
                   label=f"mean={sub['l0_l1_r'].mean():.3f}")
        ax.set_title(f"L0–L1 similarity ({metric})")
        ax.set_xlabel("network (depth=2)")
        ax.set_ylabel("Spearman r")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "qbert_layer_comparison.pdf")
    plt.close(fig)
    print(f"  Saved: qbert_layer_comparison.pdf")


# ── 1.5 Dimensionality ────────────────────────────────────────────────────────

def analysis_dimensionality(h5):
    print("\n── 1.5 Dimensionality (participation ratio) ───────────────────────")
    meta = load_runs_meta(h5)
    rows = []

    for m in meta:
        rid      = m["run_id"]
        act_path = ACT_DIR / rid / "final.npz"
        if not act_path.exists():
            act_path = ACT_DIR / rid / "best.npz"
        if not act_path.exists():
            continue
        try:
            npz = np.load(act_path)
        except Exception as e:
            print(f"  [warn] {rid}: {e}")
            continue
        for layer in range(m["depth"]):
            key = f"layer_{layer}"
            if key not in npz:
                continue
            pr = participation_ratio(npz[key].astype(np.float64))
            rows.append({
                "run_id":          rid,
                "layer":           layer,
                "is_last_layer":   (layer == m["depth"] - 1),
                "is_functional":   m["is_functional"],
                "depth":           m["depth"],
                "hidden_size":     m["hidden_size"],
                "use_attention":   m["use_attention"],
                "use_residual":    m["use_residual"],
                "performance":     m["performance"],
                "pr":              pr,
            })

    if not rows:
        print("  No activation data found — skip")
        return
    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "qbert_dimensionality.csv", index=False)

    last = df[df["is_last_layer"]]
    print(f"  Last-layer PR: mean={last['pr'].mean():.1f}  "
          f"range=[{last['pr'].min():.1f}, {last['pr'].max():.1f}]")
    if last["is_functional"].any():
        print(f"  Functional:     {last[last['is_functional']]['pr'].mean():.1f}")
    if (~last["is_functional"]).any():
        print(f"  Non-functional: {last[~last['is_functional']]['pr'].mean():.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    ax = axes[0]
    colors = ["#1a9641" if f else "#d73027" for f in last["is_functional"]]
    ax.scatter(last["hidden_size"], last["pr"], c=colors, alpha=0.8, s=45, zorder=3)
    ax.set_xlabel("hidden_size")
    ax.set_ylabel("Participation ratio")
    ax.set_title("PR vs hidden_size (last layer, final checkpoint)")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1a9641", ms=8, label="functional"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d73027", ms=8, label="non-functional"),
    ], fontsize=8)

    ax = axes[1]
    d2 = df[df["depth"] == 2]
    if len(d2) > 0:
        l0 = d2[d2["layer"] == 0].set_index("run_id")["pr"]
        l1 = d2[d2["layer"] == 1].set_index("run_id")["pr"]
        common = l0.index.intersection(l1.index)
        colors2 = ["#1a9641" if bool(d2[d2["run_id"] == r]["is_functional"].values[0])
                   else "#d73027" for r in common]
        ax.scatter(l0[common], l1[common], c=colors2, alpha=0.8, s=45, zorder=3)
        mn = min(l0[common].min(), l1[common].min()) * 0.95
        mx = max(l0[common].max(), l1[common].max()) * 1.05
        ax.plot([mn, mx], [mn, mx], "k--", lw=0.8, alpha=0.4)
        ax.set_xlabel("Layer 0 PR")
        ax.set_ylabel("Layer 1 PR")
        ax.set_title("PR: L0 vs L1 (depth=2 networks)")
    else:
        ax.text(0.5, 0.5, "No depth=2 networks", ha="center", va="center",
                transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "qbert_dimensionality.pdf")
    plt.close(fig)
    print(f"  Saved: qbert_dimensionality.pdf")


# ── 3. Dynamics — crystallisation ─────────────────────────────────────────────

def analysis_dynamics(h5):
    print("\n── 3. Dynamics — crystallisation ──────────────────────────────────")
    meta   = load_runs_meta(h5)
    metric = "pearson"
    rows   = []

    for m in meta:
        rid   = m["run_id"]
        rg    = h5["runs"][rid]
        layer = last_layer(m["depth"])

        ref_ckpt = rg.get("final") or rg.get("best")
        if ref_ckpt is None:
            continue
        ref_rdm = load_rdm(ref_ckpt, layer, metric)
        if ref_rdm is None:
            continue

        for ckpt_name in sorted(rg.keys()):
            if not ckpt_name.startswith("step_"):
                continue
            step = step_from_ckpt_name(ckpt_name)
            rdm  = load_rdm(rg[ckpt_name], layer, metric)
            rows.append({
                "run_id":        rid,
                "step":          step,
                "spearman_r":    spearman_r(rdm, ref_rdm),
                "is_functional": m["is_functional"],
                "use_attention": m["use_attention"],
                "use_residual":  m["use_residual"],
                "depth":         m["depth"],
            })

    if not rows:
        print("  No step checkpoints found — skip")
        return
    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "qbert_dynamics.csv", index=False)

    grouped = df.groupby(["step", "is_functional"])["spearman_r"].agg(["mean", "std"]).reset_index()

    for step in [1, 64, 256, 1024]:
        sub = df[df["step"] == step]
        if len(sub):
            print(f"  step={step:5d}: mean r = {sub['spearman_r'].mean():.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    for _, sub in df.groupby("run_id"):
        color = "#2166ac" if sub["is_functional"].iloc[0] else "#aaaaaa"
        ax.plot(sub["step"], sub["spearman_r"], color=color, alpha=0.35, lw=1)

    for functional, color, label in [(True, "#2166ac", "functional"),
                                      (False, "#777777", "non-functional")]:
        g = grouped[grouped["is_functional"] == functional]
        if g.empty:
            continue
        ax.plot(g["step"], g["mean"], color=color, lw=2.5, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Gradient update step (log scale)")
    ax.set_ylabel("Spearman r (to final RDM)")
    ax.set_title("Q*bert crystallisation — similarity to final representation")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "qbert_dynamics.pdf")
    plt.close(fig)
    print(f"  Saved: qbert_dynamics.pdf")


# ── Q. Architectural effects ──────────────────────────────────────────────────

def analysis_arch_effects(h5):
    print("\n── Q. Architectural effects ────────────────────────────────────────")
    meta       = load_runs_meta(h5)
    functional = [m for m in meta if m["is_functional"]]
    metric     = "pearson"

    rdms = {}
    for m in functional:
        rid   = m["run_id"]
        layer = last_layer(m["depth"])
        ckpt  = h5["runs"][rid].get("final") or h5["runs"][rid].get("best")
        if ckpt is None:
            continue
        rdm = load_rdm(ckpt, layer, metric)
        if rdm is not None:
            rdms[rid] = {"rdm": rdm, **m}

    if len(rdms) < 4:
        print("  Not enough functional runs for arch comparison — skip")
        return

    run_ids = list(rdms.keys())
    mat     = np.array([rdms[r]["rdm"] for r in run_ids])
    n       = len(run_ids)
    corr    = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            r = spearman_r(mat[i], mat[j])
            corr[i, j] = corr[j, i] = r

    def idx(rid):
        return run_ids.index(rid)

    rows = []
    features = [("use_attention", [False, True]),
                ("use_residual",  [False, True]),
                ("depth",         [1, 2])]
    for feature, values in features:
        for val in values:
            group = [r for r in run_ids if rdms[r][feature] == val]
            other = [r for r in run_ids if rdms[r][feature] != val]
            if not group or not other:
                continue
            within = [corr[idx(a), idx(b)] for a in group for b in group if a != b]
            across = [corr[idx(a), idx(b)] for a in group for b in other]
            wm = np.nanmean(within)
            am = np.nanmean(across)
            print(f"  {feature}={val}  n={len(group):2d}  "
                  f"within={wm:.3f}  across={am:.3f}  diff={wm - am:+.3f}")
            rows.append({"feature": feature, "value": str(val),
                         "n": len(group), "within_r": wm, "across_r": am})

    if not rows:
        return
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, (feature, _) in zip(axes, features):
        sub = df[df["feature"] == feature]
        x   = np.arange(len(sub))
        w   = 0.3
        ax.bar(x - w/2, sub["within_r"], w, label="within", color="#2166ac", alpha=0.8)
        ax.bar(x + w/2, sub["across_r"], w, label="across", color="#d73027", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v}" for v in sub["value"]], fontsize=9)
        ax.set_xlabel(feature)
        ax.set_ylabel("Spearman r" if feature == "use_attention" else "")
        ax.set_ylim(0, 1)
        ax.set_title(feature)
        if feature == "use_attention":
            ax.legend(fontsize=7)
    fig.suptitle("Within vs across-group RDM agreement (functional, last layer, final)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "qbert_arch_effects.pdf")
    plt.close(fig)
    print(f"  Saved: qbert_arch_effects.pdf")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    assert H5_PATH.exists(), f"RDM file not found: {H5_PATH} — run 43_qbert_rdms.py first"

    with h5py.File(H5_PATH, "r") as h5:
        n_runs = len(h5.get("runs", {}))
        print(f"Q*bert RDMs: {H5_PATH.name}  "
              f"n_stimuli={h5['meta'].attrs.get('n_stimuli', '?')}  "
              f"n_pairs={h5['meta'].attrs.get('n_pairs', '?')}  "
              f"n_runs={n_runs}")

        analysis_noise_ceiling(h5)
        analysis_layer_comparison(h5)
        analysis_dynamics(h5)
        analysis_arch_effects(h5)

    with h5py.File(H5_PATH, "r") as h5:
        analysis_dimensionality(h5)

    print("\nDone.")


if __name__ == "__main__":
    main()
