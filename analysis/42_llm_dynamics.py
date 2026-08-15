"""
B/C  Training dynamics — Pythia (6 sizes × 16 checkpoints) and OLMo (1 size × 16 checkpoints).

Uses the 90th-percentile layer (category-structure peak) and passage half as primary stimulus
set (base models). Request half also computed for reference.

Analyses:
  B.1 / C.1  Crystallisation: r(checkpoint RDM, final RDM) vs log step
  B.2 / C.2  Rate of change: RDM dissimilarity between consecutive checkpoints / log interval
  B.3 / C.3  Trajectory MDS: joint MDS across all checkpoints × sizes
  C.5        Cross-architecture: overlay OLMo-1B vs Pythia-1B crystallisation curves

RDM gallery (B.4) was produced by 40_llm_rdms.py.

Usage:
    python analysis/42_llm_dynamics.py
    python analysis/42_llm_dynamics.py --pooling mean
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.manifold import MDS

ANALYSIS  = Path(__file__).parent
REPO_ROOT = ANALYSIS.parent
sys.path.insert(0, str(ANALYSIS))

from analysis_utils import FIGURES_DIR, TABLES_DIR, FINAL_DIR

RDM_DIR         = REPO_ROOT / "output" / "analysis" / "rdms" / "llm"
FIG_DIR         = FIGURES_DIR / "llm"
TAB_DIR         = TABLES_DIR
FINAL_LLM_DYN   = FINAL_DIR / "llm" / "figures" / "dynamics"

PYTHIA_SIZES = [
    dict(model_id="EleutherAI/pythia-70m",   n_layers=6,  label="70m",   tokens_b=0.3),
    dict(model_id="EleutherAI/pythia-160m",  n_layers=12, label="160m",  tokens_b=0.3),
    dict(model_id="EleutherAI/pythia-410m",  n_layers=24, label="410m",  tokens_b=0.3),
    dict(model_id="EleutherAI/pythia-1b",    n_layers=16, label="1b",    tokens_b=0.3),
    dict(model_id="EleutherAI/pythia-1.4b",  n_layers=24, label="1.4b",  tokens_b=0.3),
    dict(model_id="EleutherAI/pythia-2.8b",  n_layers=32, label="2.8b",  tokens_b=0.3),
]

PYTHIA_CHECKPOINTS = [
    "step1", "step2", "step4", "step8", "step16", "step32", "step64", "step128",
    "step256", "step512", "step1000", "step2000", "step8000", "step32000",
    "step64000", "step143000",
]

OLMO_CHECKPOINTS = [
    "step1000-tokens2B", "step2000-tokens4B", "step3000-tokens6B", "step4500-tokens9B",
    "step7000-tokens14B", "step11000-tokens23B", "step18000-tokens37B", "step30000-tokens62B",
    "step49000-tokens102B", "step79000-tokens165B", "step128000-tokens268B",
    "step209000-tokens438B", "step339000-tokens710B", "step551000-tokens1155B",
    "step895000-tokens1876B", "step1454000-tokens3048B",
]

OLMO_MODEL = "allenai/OLMo-1B-0724-hf"
OLMO_N_LAYERS = 16

SIZE_COLORS = {
    "70m":   "#a6cee3",
    "160m":  "#1f78b4",
    "410m":  "#b2df8a",
    "1b":    "#33a02c",
    "1.4b":  "#fb9a99",
    "2.8b":  "#e31a1c",
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def slug(model_id, revision):
    return model_id.replace("/", "__") + f"__{revision}"


def checkpoint_step(revision):
    """Extract step number from revision string."""
    m = re.match(r"step(\d+)", revision)
    assert m, f"Cannot parse step from {revision!r}"
    return int(m.group(1))


def checkpoint_tokens_b(revision):
    """Extract token count in billions from OLMo revision string, or estimate for Pythia."""
    m = re.search(r"tokens(\d+(?:\.\d+)?)([BM])", revision)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        return val if unit == "B" else val / 1000
    # Pythia: ~2.1M tokens per step
    step = checkpoint_step(revision)
    return step * 2.1e6 / 1e9


def pct_layer_label(n_layers, pct):
    """Label of the transformer block at the given percentile (1-indexed, L1…Ln)."""
    idx = round(pct / 100 * (n_layers - 1))
    return f"L{idx + 1}"


def upper_tri(rdm):
    n = rdm.shape[0]
    return rdm[np.triu_indices(n, k=1)]


def rdm_corr(a, b):
    ua, ub = upper_tri(a), upper_tri(b)
    ua = ua - ua.mean()
    ub = ub - ub.mean()
    denom = np.linalg.norm(ua) * np.linalg.norm(ub)
    if denom == 0:
        return 0.0
    return float(np.dot(ua, ub) / denom)


def load_rdm(model_id, revision, pooling, subset, layer_label):
    s = slug(model_id, revision)
    path = RDM_DIR / f"{s}_{pooling}_{subset}.npz"
    d = np.load(path)
    return d[layer_label].astype(np.float32)


# ── B.1 / C.1  Crystallisation ────────────────────────────────────────────────

def crystallisation(model_id, n_layers, checkpoints, pooling, subset, pct=90):
    """r(checkpoint_rdm, final_rdm) for each checkpoint."""
    lbl = pct_layer_label(n_layers, pct)
    final_rev = checkpoints[-1]
    final_rdm = load_rdm(model_id, final_rev, pooling, subset, lbl)
    rows = []
    for rev in checkpoints:
        rdm = load_rdm(model_id, rev, pooling, subset, lbl)
        r = rdm_corr(rdm, final_rdm)
        rows.append(dict(
            revision=rev,
            step=checkpoint_step(rev),
            tokens_b=checkpoint_tokens_b(rev),
            r_to_final=r,
        ))
    return pd.DataFrame(rows)


# ── B.2 / C.2  Rate of change ─────────────────────────────────────────────────

def rate_of_change(model_id, n_layers, checkpoints, pooling, subset, pct=90):
    lbl = pct_layer_label(n_layers, pct)
    rdms = [load_rdm(model_id, rev, pooling, subset, lbl) for rev in checkpoints]
    rows = []
    for i in range(1, len(checkpoints)):
        dissim = 1 - rdm_corr(rdms[i - 1], rdms[i])
        log_interval = np.log10(checkpoint_step(checkpoints[i])) - \
                       np.log10(checkpoint_step(checkpoints[i - 1]))
        rate = dissim / log_interval if log_interval > 0 else 0.0
        mid_step = (checkpoint_step(checkpoints[i - 1]) + checkpoint_step(checkpoints[i])) / 2
        mid_tokens = (checkpoint_tokens_b(checkpoints[i - 1]) + checkpoint_tokens_b(checkpoints[i])) / 2
        rows.append(dict(
            revision_from=checkpoints[i - 1],
            revision_to=checkpoints[i],
            mid_step=mid_step,
            mid_tokens_b=mid_tokens,
            dissim=dissim,
            rate=rate,
        ))
    return pd.DataFrame(rows)


# ── B.3 / C.3  Trajectory MDS — joint across all Pythia sizes + OLMo ──────────

def trajectory_mds_joint(pooling, subset, pct=90):
    """Joint MDS: 6 Pythia sizes × 16 checkpoints + OLMo × 16 = 112 points."""
    all_rdms, labels = [], []
    for s in PYTHIA_SIZES:
        lbl = pct_layer_label(s["n_layers"], pct)
        for rev in PYTHIA_CHECKPOINTS:
            all_rdms.append(load_rdm(s["model_id"], rev, pooling, subset, lbl))
            labels.append(dict(model=s["label"], family="pythia",
                               step=checkpoint_step(rev),
                               tokens_b=checkpoint_tokens_b(rev)))
    olmo_lbl = pct_layer_label(OLMO_N_LAYERS, pct)
    for rev in OLMO_CHECKPOINTS:
        all_rdms.append(load_rdm(OLMO_MODEL, rev, pooling, subset, olmo_lbl))
        labels.append(dict(model="OLMo-1B", family="olmo",
                           step=checkpoint_step(rev),
                           tokens_b=checkpoint_tokens_b(rev)))
    n = len(all_rdms)
    dissim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = 1 - rdm_corr(all_rdms[i], all_rdms[j])
            dissim[i, j] = dissim[j, i] = d
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
              normalized_stress=False, n_init=1)
    coords = mds.fit_transform(dissim)
    return coords, labels


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_crystallisation(cryst_pythia, cryst_olmo, subset):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    # Pythia panel
    ax = axes[0]
    for size_meta, df in cryst_pythia.items():
        ax.plot(df["step"], df["r_to_final"],
                color=SIZE_COLORS[size_meta], lw=2, marker="o", ms=4,
                label=size_meta)
    ax.set_xscale("log")
    ax.set_xlabel("Training step (log scale)")
    ax.set_ylabel("r to final checkpoint")
    ax.set_title(f"B.1 Crystallisation — Pythia ({subset})")
    ax.legend(title="Size", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", alpha=0.3)

    # OLMo panel
    ax = axes[1]
    ax.plot(cryst_olmo["step"], cryst_olmo["r_to_final"],
            color="#b07aa1", lw=2, marker="o", ms=4)
    ax.set_xscale("log")
    ax.set_xlabel("Training step (log scale)")
    ax.set_title(f"C.1 Crystallisation — OLMo-1B ({subset})")
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"BC1_crystallisation_{subset}.pdf")
    FINAL_LLM_DYN.mkdir(parents=True, exist_ok=True)
    fig.savefig(FINAL_LLM_DYN / f"BC1_crystallisation_{subset}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rate_of_change(rate_pythia, rate_olmo, subset):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    for size_meta, df in rate_pythia.items():
        ax.plot(df["mid_step"], df["rate"],
                color=SIZE_COLORS[size_meta], lw=2, marker="o", ms=4,
                label=size_meta)
    ax.set_xscale("log")
    ax.set_xlabel("Training step (log scale)")
    ax.set_ylabel("RDM change rate (dissim / log-step interval)")
    ax.set_title(f"B.2 Rate of change — Pythia ({subset})")
    ax.legend(title="Size", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.plot(rate_olmo["mid_step"], rate_olmo["rate"],
            color="#b07aa1", lw=2, marker="o", ms=4)
    ax.set_xscale("log")
    ax.set_xlabel("Training step (log scale)")
    ax.set_title(f"C.2 Rate of change — OLMo-1B ({subset})")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"BC2_rate_of_change_{subset}.pdf")
    FINAL_LLM_DYN.mkdir(parents=True, exist_ok=True)
    fig.savefig(FINAL_LLM_DYN / f"BC2_rate_of_change_{subset}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_mds_joint(coords, labels, subset):
    """Joint MDS: all Pythia sizes + OLMo in one panel. Colour = model, alpha = training progress."""
    from matplotlib.lines import Line2D

    TRAJ_COLORS = {**{s["label"]: SIZE_COLORS[s["label"]] for s in PYTHIA_SIZES},
                   "OLMo-1B": "#b07aa1"}

    fig, ax = plt.subplots(figsize=(9, 7))

    # Group indices by model
    models_in_order = [s["label"] for s in PYTHIA_SIZES] + ["OLMo-1B"]
    n_ckpts_per = {s["label"]: len(PYTHIA_CHECKPOINTS) for s in PYTHIA_SIZES}
    n_ckpts_per["OLMo-1B"] = len(OLMO_CHECKPOINTS)

    idx = 0
    for model_name in models_in_order:
        n = n_ckpts_per[model_name]
        seg = coords[idx: idx + n]
        color = TRAJ_COLORS[model_name]
        alphas = np.linspace(0.25, 1.0, n)
        sizes  = np.linspace(20, 80, n)
        for i in range(n - 1):
            ax.plot(seg[i:i+2, 0], seg[i:i+2, 1], color=color, lw=1.5,
                    alpha=(alphas[i] + alphas[i+1]) / 2)
        ax.scatter(seg[:, 0], seg[:, 1], color=color, s=sizes,
                   alpha=alphas, zorder=3)
        ax.annotate(model_name, seg[-1], fontsize=6, color=color,
                    xytext=(3, 3), textcoords="offset points", fontweight="bold")
        idx += n

    legend_elements = [
        Line2D([0], [0], color=TRAJ_COLORS[s["label"]], lw=2, label=f"Pythia-{s['label']}")
        for s in PYTHIA_SIZES
    ] + [Line2D([0], [0], color="#b07aa1", lw=2, label="OLMo-1B")]
    ax.legend(handles=legend_elements, fontsize=8, loc="best")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"B.3/C.3 Trajectory MDS — Pythia + OLMo, joint embedding ({subset})\n"
                 f"dot size/opacity = training progress (small/faint = early)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"BC3_trajectory_mds_{subset}.pdf")
    FINAL_LLM_DYN.mkdir(parents=True, exist_ok=True)
    fig.savefig(FINAL_LLM_DYN / f"BC3_trajectory_mds_{subset}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_mds_olmo(coords, steps, subset):
    fig, ax = plt.subplots(figsize=(6, 5))
    log_s = np.log10(steps)
    norm = plt.Normalize(log_s.min(), log_s.max())
    cmap = cm.plasma
    for i in range(len(coords) - 1):
        ax.plot(coords[i:i+2, 0], coords[i:i+2, 1],
                color=cmap(norm(log_s[i])), lw=1.5, alpha=0.7)
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=log_s, cmap=cmap, norm=norm, s=60, zorder=3)
    ax.annotate("start", coords[0],  fontsize=8, xytext=(4, 4),
                textcoords="offset points", color="green")
    ax.annotate("end",   coords[-1], fontsize=8, xytext=(4, 4),
                textcoords="offset points", color="red")
    plt.colorbar(sc, ax=ax, label="log10(step)")
    ax.set_title(f"C.3 Trajectory MDS — OLMo-1B ({subset})", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"C3_trajectory_mds_olmo_{subset}.pdf")
    plt.close(fig)


def plot_c5_cross_architecture(cryst_pythia_1b, cryst_olmo, subset):
    """Overlay OLMo and Pythia-1B crystallisation curves on tokens axis."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(cryst_pythia_1b["tokens_b"], cryst_pythia_1b["r_to_final"],
            color=SIZE_COLORS["1b"], lw=2, marker="o", ms=5, label="Pythia-1B")
    ax.plot(cryst_olmo["tokens_b"], cryst_olmo["r_to_final"],
            color="#b07aa1", lw=2, marker="s", ms=5, label="OLMo-1B")
    ax.set_xscale("log")
    ax.set_xlabel("Training tokens (B, log scale)")
    ax.set_ylabel("r to final checkpoint")
    ax.set_title(f"C.5 Crystallisation — Pythia-1B vs OLMo-1B ({subset})")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"C5_cross_architecture_{subset}.pdf")
    FINAL_LLM_DYN.mkdir(parents=True, exist_ok=True)
    fig.savefig(FINAL_LLM_DYN / f"C5_cross_architecture_{subset}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooling", default="last", choices=["last", "mean"])
    parser.add_argument("--pct",     default=90, type=int,
                        help="Percentile layer to use (default: 90)")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    pooling = args.pooling
    pct     = args.pct

    print(f"Pooling: {pooling},  layer: {pct}th percentile")

    for subset in ["passage", "request"]:
        print(f"\n═══ Subset: {subset} ═══════════════════════════════════════════")

        # B.1 Crystallisation — Pythia
        print("B.1 Crystallisation — Pythia")
        cryst_pythia = {}
        all_cryst_records = []
        for s in PYTHIA_SIZES:
            df = crystallisation(s["model_id"], s["n_layers"], PYTHIA_CHECKPOINTS,
                                 pooling, subset, pct)
            df["size"] = s["label"]
            df["family"] = "pythia"
            cryst_pythia[s["label"]] = df
            all_cryst_records.append(df)
            print(f"  pythia-{s['label']:5s}  r at step1={df.r_to_final.iloc[0]:.3f}  "
                  f"r at step143000={df.r_to_final.iloc[-1]:.3f}")

        # C.1 Crystallisation — OLMo
        print("C.1 Crystallisation — OLMo")
        cryst_olmo = crystallisation(OLMO_MODEL, OLMO_N_LAYERS, OLMO_CHECKPOINTS,
                                     pooling, subset, pct)
        cryst_olmo["size"] = "1b"
        cryst_olmo["family"] = "olmo"
        all_cryst_records.append(cryst_olmo)
        print(f"  OLMo-1B  r at step1000={cryst_olmo.r_to_final.iloc[0]:.3f}  "
              f"r at final={cryst_olmo.r_to_final.iloc[-1]:.3f}")

        pd.concat(all_cryst_records).to_csv(
            TAB_DIR / f"llm_bc1_crystallisation_{subset}.csv", index=False)

        # B.2 Rate of change — Pythia
        print("B.2 Rate of change — Pythia")
        rate_pythia = {}
        all_rate_records = []
        for s in PYTHIA_SIZES:
            df = rate_of_change(s["model_id"], s["n_layers"], PYTHIA_CHECKPOINTS,
                                pooling, subset, pct)
            df["size"] = s["label"]
            df["family"] = "pythia"
            rate_pythia[s["label"]] = df
            all_rate_records.append(df)

        # C.2 Rate of change — OLMo
        print("C.2 Rate of change — OLMo")
        rate_olmo = rate_of_change(OLMO_MODEL, OLMO_N_LAYERS, OLMO_CHECKPOINTS,
                                   pooling, subset, pct)
        rate_olmo["size"] = "1b"
        rate_olmo["family"] = "olmo"
        all_rate_records.append(rate_olmo)
        pd.concat(all_rate_records).to_csv(
            TAB_DIR / f"llm_bc2_rate_{subset}.csv", index=False)

        # B.3/C.3 Trajectory MDS — joint Pythia + OLMo
        print("B.3/C.3 Trajectory MDS — Pythia + OLMo (joint)")
        joint_coords, joint_labels = trajectory_mds_joint(pooling, subset, pct)

        # Figures
        print("Saving figures...")
        plot_crystallisation(cryst_pythia, cryst_olmo, subset)
        plot_rate_of_change(rate_pythia, rate_olmo, subset)
        plot_trajectory_mds_joint(joint_coords, joint_labels, subset)
        plot_c5_cross_architecture(cryst_pythia["1b"], cryst_olmo, subset)

    print("\nDone.")


if __name__ == "__main__":
    main()
