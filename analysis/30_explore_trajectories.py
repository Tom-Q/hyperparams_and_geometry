#!/usr/bin/env python3
"""
Exploratory trajectory analysis for Finding #3 — representations through learning.

For mnist_10way, fourrooms, adding: selects successful primary networks and produces:
  A. RDM gallery across checkpoints (5 networks)
  B. Similarity-to-reference curves (5 networks)
  C. Trajectory plots — PCA and UMAP (20 networks, all labeled and numbered)

Performance checkpoints are deduplicated: when multiple thresholds are crossed at
the same training step, only the highest threshold is kept; earlier ones are "no data".

Checkpoint types: performance / step / epoch (epoch omitted if absent in the data).
Reference: best checkpoint (peak validation accuracy) for supervised/RNN;
           final checkpoint (end of training) for RL.

Outputs: output/analysis/thru_learning_exploratory/
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from adjustText import adjust_text
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from umap import UMAP

ANALYSIS  = Path(__file__).parent
REPO_ROOT = ANALYSIS.parent
sys.path.insert(0, str(ANALYSIS))
from analysis_utils import RDM_DIR, TABLES_DIR, RL_TASKS

OUT_DIR   = REPO_ROOT / "output" / "analysis" / "thru_learning_exploratory"
RNN_TASKS = {"adding", "mnist_rnn"}
TASKS     = ["mnist_10way", "fourrooms", "adding"]
N_GALLERY = 5
N_UMAP    = 20
METRIC    = "cosine"

# ColorBrewer Set1 for 5 gallery/curve networks
COLORS_5 = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"]

# tab20 for 20 trajectory networks
_tab20    = plt.get_cmap("tab20")
COLORS_20 = [_tab20(i) for i in range(20)]

CKPT_ATTR = {
    "performance": "perf_ckpts",
    "step":        "step_ckpts",
    "epoch":       "epoch_ckpts",
}

REF_LABEL = {
    True:  "final (end of training)",
    False: "best (peak val. accuracy)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ref_ckpt_name(task):
    return "final" if task in RL_TASKS else "best"


def last_layer_key(task, rg):
    if task in RNN_TASKS:
        return f"temporal_{METRIC}"
    depth = int(rg.attrs.get("hp_depth", 1))
    return f"layer_{depth - 1}_{METRIC}"


def load_rdm_vec(cg, key):
    ds = cg.get(key)
    if ds is None or ds.attrs.get("degenerate", False) or len(ds) == 0:
        return None
    return ds[:].astype(np.float64)


def vec_to_matrix(vec):
    n = round((1 + np.sqrt(1 + 8 * len(vec))) / 2)
    mat = np.zeros((n, n))
    ri, ci = np.triu_indices(n, k=1)
    mat[ri, ci] = vec
    mat += mat.T
    return mat


def parse_ckpt_value(name):
    if name.startswith("perf_"):
        return float(name[5:].replace("p", "."))
    if name.startswith("step_"):
        return int(name[5:])
    if name.startswith("epoch_"):
        return float(name[6:].replace("p", "."))
    return 0.0


def spearman_finite(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return float("nan")
    return float(spearmanr(a[mask], b[mask])[0])


def strip_nan_cols(X):
    valid = np.all(np.isfinite(X), axis=0)
    return X[:, valid]


def spearman_dissimilarity_matrix(X):
    """
    Pairwise (1 - Spearman r) / 2 between rows of X.
    X must be finite (call strip_nan_cols first).
    Returns a symmetric matrix with zeros on the diagonal, values in [0, 1].
    """
    n = X.shape[0]
    # Rank-transform each row; scipy spearmanr on columns of the transposed matrix
    # gives correlations between rows of X.
    from scipy.stats import spearmanr as _spearmanr
    result = _spearmanr(X.T)
    # spearmanr returns a scalar when n==2, a SpearmanrResult otherwise
    if n == 2:
        r_mat = np.array([[1.0, result.statistic],
                          [result.statistic, 1.0]])
    else:
        r_mat = np.array(result.statistic)
    D = (1.0 - r_mat) / 2.0
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0.0, 1.0)


def deduplicate_perf_ckpts(ckpts):
    """
    For each run of consecutive identical vectors, keep only the entry with the
    highest performance level. Earlier thresholds crossed at the same training
    step carry no additional information and are dropped.
    """
    if not ckpts:
        return {}
    items = sorted(ckpts.items())   # ascending by level
    result = {}
    i = 0
    while i < len(items):
        level_i, vec_i = items[i]
        j = i + 1
        while j < len(items) and np.allclose(items[j][1], vec_i, equal_nan=True):
            j += 1
        # items[j-1] is the highest level in this group of identical vectors
        result[items[j - 1][0]] = items[j - 1][1]
        i = j
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_thresholds():
    data = json.load(open(TABLES_DIR / "success_thresholds.json"))
    return {k: float(v["upper"]) for k, v in data.items() if isinstance(v, dict)}


def select_networks(task, thresholds):
    h5_path   = RDM_DIR / f"{task}_rdms.h5"
    threshold = thresholds.get(task, -float("inf"))
    ref_name  = ref_ckpt_name(task)
    nets, n_degenerate = [], 0
    with h5py.File(h5_path, "r") as f:
        for run_id, rg in f["runs"].items():
            if rg.attrs.get("is_repeat", False):
                continue
            perf = float(rg.attrs.get("performance", float("nan")))
            if not (np.isfinite(perf) and perf >= threshold):
                continue
            # Require a valid (non-degenerate) reference checkpoint
            lkey   = last_layer_key(task, rg)
            ref_cg = rg.get(ref_name)
            if ref_cg is None:
                n_degenerate += 1
                continue
            ds = ref_cg.get(lkey)
            if ds is None or ds.attrs.get("degenerate", False) or len(ds) == 0:
                n_degenerate += 1
                continue
            nets.append((run_id, perf))
    if n_degenerate:
        print(f"  Excluded {n_degenerate} networks with degenerate/missing reference ({ref_name})")
    nets.sort(key=lambda x: x[1])
    n = len(nets)
    assert n >= N_UMAP, f"{task}: only {n} usable networks after exclusions, need {N_UMAP}"

    umap_idx    = np.round(np.linspace(0, n - 1, N_UMAP)).astype(int)
    umap_ids    = [nets[i][0] for i in umap_idx]
    gallery_idx = np.round(np.linspace(0, N_UMAP - 1, N_GALLERY)).astype(int)
    gallery_ids = [umap_ids[i] for i in gallery_idx]
    perf_map    = {nets[i][0]: nets[i][1] for i in umap_idx}
    return gallery_ids, umap_ids, perf_map


def load_network(task, run_id, h5_file):
    rg       = h5_file["runs"][run_id]
    lkey     = last_layer_key(task, rg)
    ref_name = ref_ckpt_name(task)

    nd = {
        "run_id":      run_id,
        "perf":        float(rg.attrs.get("performance", float("nan"))),
        "lkey":        lkey,
        "perf_ckpts":  {},
        "step_ckpts":  {},
        "epoch_ckpts": {},
        "ref_vec":     None,
    }

    ref_cg = rg.get(ref_name)
    if ref_cg is not None:
        nd["ref_vec"] = load_rdm_vec(ref_cg, lkey)

    for name, cg in rg.items():
        vec = load_rdm_vec(cg, lkey)
        if vec is None:
            continue
        if name.startswith("perf_"):
            nd["perf_ckpts"][parse_ckpt_value(name)] = vec
        elif name.startswith("step_"):
            nd["step_ckpts"][int(parse_ckpt_value(name))] = vec
        elif name.startswith("epoch_"):
            nd["epoch_ckpts"][parse_ckpt_value(name)] = vec

    nd["perf_ckpts"] = deduplicate_perf_ckpts(nd["perf_ckpts"])
    return nd


def load_networks(task, run_ids):
    h5_path = RDM_DIR / f"{task}_rdms.h5"
    result  = {}
    with h5py.File(h5_path, "r") as f:
        for run_id in run_ids:
            result[run_id] = load_network(task, run_id, f)
    return result


# ---------------------------------------------------------------------------
# Figure A: RDM gallery
# ---------------------------------------------------------------------------

def make_gallery(task, gallery_nets):
    is_rl    = task in RL_TASKS
    ref_text = REF_LABEL[is_rl]

    sections = []
    for ckpt_type, attr in [("performance", "perf_ckpts"),
                             ("step",        "step_ckpts"),
                             ("epoch",       "epoch_ckpts")]:
        all_keys = set()
        for nd in gallery_nets:
            all_keys.update(nd[attr].keys())
        if all_keys:
            sections.append((ckpt_type, attr, sorted(all_keys)))

    if not sections:
        return None

    n_nets   = len(gallery_nets)
    max_cols = max(len(s[2]) + 1 for s in sections)
    cell_w, cell_h = 1.4, 1.4
    gap_h    = 0.5
    n_sec    = len(sections)
    fig_w    = cell_w * max_cols + 1.2
    fig_h    = (cell_h * n_nets + gap_h) * n_sec + 0.4
    fig      = plt.figure(figsize=(fig_w, fig_h))
    sec_h_frac = 1.0 / n_sec

    for sec_idx, (ckpt_type, attr, sorted_keys) in enumerate(sections):
        col_w_frac = 1.0 / max_cols
        row_h_frac = sec_h_frac / (n_nets + 0.4)
        sec_top    = 1.0 - sec_idx * sec_h_frac

        fig.text(0.01, sec_top - 0.01, f"▸ {ckpt_type} checkpoints",
                 fontsize=9, fontweight="bold", va="top",
                 transform=fig.transFigure)

        for row, nd in enumerate(gallery_nets):
            row_top = sec_top - (row + 1.1) * row_h_frac

            fig.text(0.0, row_top + row_h_frac * 0.5,
                     f"{nd['run_id']}\nperf={nd['perf']:.3f}",
                     fontsize=5.5, va="center", ha="left",
                     transform=fig.transFigure)

            for col, key in enumerate(sorted_keys):
                left = 0.10 + col * col_w_frac
                ax   = fig.add_axes([left, row_top,
                                     col_w_frac * 0.92, row_h_frac * 0.88])
                vec  = nd[attr].get(key)
                if vec is not None:
                    ax.imshow(vec_to_matrix(vec), cmap="viridis",
                              vmin=0, vmax=1, aspect="equal",
                              interpolation="nearest")
                else:
                    ax.set_facecolor("#dddddd")
                    ax.text(0.5, 0.5, "not\nreached", ha="center", va="center",
                            fontsize=5, color="#888888", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    if ckpt_type == "performance":
                        lbl = f"perf\n{key:.3f}"
                    elif ckpt_type == "step":
                        lbl = f"step\n{key}"
                    else:
                        lbl = f"epoch\n{key}"
                    ax.set_title(lbl, fontsize=5.5, pad=2)

            # Reference column
            left = 0.10 + len(sorted_keys) * col_w_frac
            ax   = fig.add_axes([left, row_top,
                                  col_w_frac * 0.92, row_h_frac * 0.88])
            if nd["ref_vec"] is not None:
                ax.imshow(vec_to_matrix(nd["ref_vec"]), cmap="viridis",
                          vmin=0, vmax=1, aspect="equal",
                          interpolation="nearest")
            else:
                ax.set_facecolor("#ffdddd")
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        fontsize=5, color="#cc0000", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"REFERENCE\n({ref_text})", fontsize=5.5, pad=2,
                             color="darkred")
            for spine in ax.spines.values():
                spine.set_edgecolor("darkred")
                spine.set_linewidth(1.5)

    fig.suptitle(
        f"{task} — RDM gallery (last hidden layer, {METRIC} distance, vmin=0 vmax=1)\n"
        f"Rows = networks sorted by performance | columns = checkpoints in learning order",
        fontsize=8, y=1.0,
    )
    return fig


# ---------------------------------------------------------------------------
# Figure B: Similarity curves
# ---------------------------------------------------------------------------

def similarity_curve(nd, ckpt_type):
    attr    = CKPT_ATTR[ckpt_type]
    ref_vec = nd.get("ref_vec")
    if ref_vec is None:
        return [], []
    pairs = []
    for x, vec in sorted(nd[attr].items()):
        r = spearman_finite(vec, ref_vec)
        if np.isfinite(r):
            pairs.append((x, r))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def make_curves(task, gallery_nets):
    is_rl    = task in RL_TASKS
    ref_text = ("final checkpoint (end of training)" if is_rl else
                "best checkpoint (model weights at peak validation accuracy)")

    types_present = [ct for ct in ["performance", "step", "epoch"]
                     if any(nd[CKPT_ATTR[ct]] for nd in gallery_nets)]
    if not types_present:
        return None

    fig, axes = plt.subplots(1, len(types_present),
                              figsize=(5 * len(types_present), 4.5),
                              squeeze=False)
    axes = axes[0]

    for ax, ckpt_type in zip(axes, types_present):
        use_log = ckpt_type in ("step", "epoch")
        for nd, color in zip(gallery_nets, COLORS_5):
            xs, rs = similarity_curve(nd, ckpt_type)
            if not xs:
                continue
            x_arr = np.array(xs, dtype=float)
            if use_log:
                x_arr = np.log10(np.maximum(x_arr, 1e-9))
            ax.plot(x_arr, rs, color=color, linewidth=1.2,
                    marker="o", markersize=4,
                    markeredgewidth=0.5, markeredgecolor="white",
                    label=f"{nd['run_id']}  performance={nd['perf']:.3f}")

        ax.axhline(0.9, color="grey", linestyle="--", linewidth=0.8,
                   alpha=0.8, label="r = 0.9")
        ax.set_ylim(-0.15, 1.05)

        if ckpt_type == "performance":
            ax.set_xlabel("Normalised performance", fontsize=8)
            ax.set_xlim(0, 1.02)
        elif ckpt_type == "step":
            ax.set_xlabel("Training step (log₁₀ scale)", fontsize=8)
        else:
            ax.set_xlabel("Training epoch (log₁₀ scale)", fontsize=8)

        ax.set_ylabel("Spearman r with reference", fontsize=8)
        ax.set_title(f"{ckpt_type} checkpoints", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, fontsize=6.5, loc="lower center",
               ncol=min(3, len(gallery_nets) + 1),
               bbox_to_anchor=(0.5, -0.18), framealpha=0.9)

    fig.suptitle(
        f"{task} — Similarity to reference over learning\n"
        f"Reference = {ref_text}",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# Figure C: Trajectory plots (PCA and UMAP)
# ---------------------------------------------------------------------------

def build_trajectory_data(umap_nets_ordered, ckpt_type):
    attr = CKPT_ATTR[ckpt_type]
    vecs, meta = [], []
    for net_idx, nd in enumerate(umap_nets_ordered):
        ckpts = sorted(nd[attr].items())
        for seq_num, (x_val, vec) in enumerate(ckpts, start=1):
            vecs.append(vec)
            meta.append(dict(net_idx=net_idx, seq=seq_num, x_val=x_val,
                             is_ref=False, run_id=nd["run_id"], perf=nd["perf"],
                             total=len(ckpts) + 1))
        if nd.get("ref_vec") is not None:
            vecs.append(nd["ref_vec"])
            meta.append(dict(net_idx=net_idx, seq=len(ckpts) + 1,
                             x_val=float("inf"), is_ref=True,
                             run_id=nd["run_id"], perf=nd["perf"],
                             total=len(ckpts) + 1))
    return vecs, meta


def _draw_trajectory(ax, net_points, method_label, task, ckpt_type):
    """Draw all network trajectories on ax, apply adjustText, return the figure."""
    is_rl    = task in RL_TASKS
    ref_desc = REF_LABEL[is_rl]

    all_texts = []

    for net_idx, pts in sorted(net_points.items()):
        color = COLORS_20[net_idx]
        xs    = [p["x"] for p in pts]
        ys    = [p["y"] for p in pts]

        # Segments
        for i in range(len(pts) - 1):
            ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]],
                    color=color, linewidth=0.4, alpha=0.75, zorder=2)

        # Markers — start and reference only; intermediate points visible from line angles
        for p in pts:
            if p["is_ref"]:
                ax.scatter(p["x"], p["y"], facecolors="none", s=35, marker="*",
                           zorder=4, edgecolors="black", linewidths=0.6)
            elif p["seq"] == 1:
                ax.scatter(p["x"], p["y"], facecolors="none", s=22, marker="^",
                           zorder=4, edgecolors="black", linewidths=0.6)

            # Numbered label — color matches network
            t = ax.text(p["x"], p["y"], str(p["seq"]),
                        fontsize=4.5, ha="center", va="center",
                        color=color, fontweight="bold", zorder=6)
            all_texts.append(t)

    # Repel all labels away from each other and from points
    adjust_text(
        all_texts, ax=ax,
        arrowprops=dict(arrowstyle="-", color="#777777", lw=0.35, alpha=0.6),
        expand=(1.3, 1.3),
        force_text=(0.5, 0.5),
    )

    # Legend: all networks
    legend_handles = [
        plt.Line2D([0], [0], color=COLORS_20[ni], linewidth=1.2,
                   label=f"{pts[0]['run_id']}  perf={pts[0]['perf']:.3f}")
        for ni, pts in sorted(net_points.items())
    ]
    leg1 = ax.legend(handles=legend_handles, fontsize=4.5, loc="upper left",
                     bbox_to_anchor=(1.01, 1.0), framealpha=0.9,
                     title="Networks (worst→best)", title_fontsize=5.5)
    ax.add_artist(leg1)

    from matplotlib.lines import Line2D
    marker_leg = [
        Line2D([0], [0], marker="^", color="grey", ls="none", ms=5,
               label="▲ start"),
        Line2D([0], [0], marker="*", color="grey", ls="none", ms=7,
               label="★ reference"),
    ]
    ax.legend(handles=marker_leg, fontsize=5.5, loc="lower left",
              bbox_to_anchor=(1.01, 0.0), framealpha=0.9,
              title="Markers", title_fontsize=5.5)

    ax.set_xlabel("Dimension 1 (arbitrary units)", fontsize=8)
    ax.set_ylabel("Dimension 2 (arbitrary units)", fontsize=8)
    ax.tick_params(labelsize=7)

    ax.set_title(
        f"{task} — {method_label} trajectories ({ckpt_type} checkpoints)\n"
        f"▲ = start | ★ = reference ({ref_desc}) | numbers = checkpoint sequence per network",
        fontsize=7.5,
    )


def make_trajectory_fig(task, umap_nets_ordered, ckpt_type, method):
    """Produce one trajectory figure using 'mds' or 'umap', both on Spearman dissimilarities."""
    vecs, meta = build_trajectory_data(umap_nets_ordered, ckpt_type)
    if len(vecs) < 5:
        return None

    X = np.array(vecs, dtype=np.float64)
    X = strip_nan_cols(X)
    if X.shape[1] == 0:
        return None

    D = spearman_dissimilarity_matrix(X)

    if method == "mds":
        emb = MDS(n_components=2, metric="precomputed",
                  n_init=1, init="classical_mds",
                  random_state=42, normalized_stress="auto").fit_transform(D)
        method_label = "MDS"
    else:  # umap
        emb = UMAP(n_neighbors=min(10, len(X) - 1), min_dist=0.5,
                   spread=1.5, n_components=2, metric="precomputed",
                   random_state=42, verbose=False).fit_transform(D)
        method_label = "UMAP"

    # Group points by network
    net_points = defaultdict(list)
    for i, m in enumerate(meta):
        net_points[m["net_idx"]].append({**m, "x": emb[i, 0], "y": emb[i, 1]})
    for ni in net_points:
        net_points[ni].sort(key=lambda p: p["seq"])

    fig, ax = plt.subplots(figsize=(8, 7))
    _draw_trajectory(ax, net_points, method_label, task, ckpt_type)
    fig.tight_layout(rect=[0, 0, 0.72, 1.0])
    return fig


# ---------------------------------------------------------------------------
# Figure C (3D): Interactive Plotly trajectory plots — step checkpoints only
# ---------------------------------------------------------------------------

def _rgba_to_plotly(rgba):
    r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    return f"rgb({r},{g},{b})"


def make_trajectory_3d(task, umap_nets_ordered, method):
    """
    Produce an interactive 3D Plotly figure for step checkpoints.
    Returns a plotly Figure (save with write_html).
    """
    vecs, meta = build_trajectory_data(umap_nets_ordered, "step")
    if len(vecs) < 5:
        return None

    X = np.array(vecs, dtype=np.float64)
    X = strip_nan_cols(X)
    if X.shape[1] == 0:
        return None

    D = spearman_dissimilarity_matrix(X)

    if method == "mds":
        emb = MDS(n_components=3, metric="precomputed",
                  n_init=4, random_state=42,
                  normalized_stress="auto").fit_transform(D)
        method_label = "MDS"
    else:
        emb = UMAP(n_neighbors=min(10, len(X) - 1), min_dist=0.5,
                   spread=1.5, n_components=3, metric="precomputed",
                   random_state=42, verbose=False).fit_transform(D)
        method_label = "UMAP"

    net_points = defaultdict(list)
    for i, m in enumerate(meta):
        net_points[m["net_idx"]].append(
            {**m, "x": emb[i, 0], "y": emb[i, 1], "z": emb[i, 2]}
        )
    for ni in net_points:
        net_points[ni].sort(key=lambda p: p["seq"])

    is_rl    = task in RL_TASKS
    ref_desc = REF_LABEL[is_rl]
    traces   = []

    for net_idx, pts in sorted(net_points.items()):
        color    = _rgba_to_plotly(COLORS_20[net_idx])
        run_id   = pts[0]["run_id"]
        perf     = pts[0]["perf"]
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        zs = [p["z"] for p in pts]

        # Trajectory line
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=color, width=2),
            name=f"{run_id}  perf={perf:.3f}",
            legendgroup=run_id,
            showlegend=True,
            hoverinfo="skip",
        ))

        # All points — hover shows sequence and step value
        hover = [
            f"{run_id}<br>seq {p['seq']}"
            + (f"<br>step {int(p['x_val'])}" if not p["is_ref"] else "<br>reference")
            for p in pts
        ]
        symbol = ["circle"] * len(pts)
        size   = [4] * len(pts)
        for k, p in enumerate(pts):
            if p["seq"] == 1:
                symbol[k] = "circle-open"
                size[k]   = 16
            elif p["is_ref"]:
                symbol[k] = "diamond-open"
                size[k]   = 20

        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers+text",
            marker=dict(color=color, size=size, symbol=symbol,
                        line=dict(color=color, width=1)),
            text=["▲" if p["seq"] == 1 else ("★" if p["is_ref"] else str(p["seq"]))
                  for p in pts],
            textposition="top center",
            textfont=dict(color=color, size=8),
            hovertext=hover,
            hoverinfo="text",
            name=run_id,
            legendgroup=run_id,
            showlegend=False,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(f"{task} — {method_label} 3D representational trajectories "
                  f"(step checkpoints)<br>"
                  f"<sup>▲ = start  ★ = reference ({ref_desc})  "
                  f"numbers = step sequence  hover for details</sup>"),
            font=dict(size=13),
        ),
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
        ),
        legend=dict(font=dict(size=9), title="Networks (worst→best perf)"),
        width=1100,
        height=850,
        margin=dict(l=0, r=0, t=80, b=0),
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds()

    for task in TASKS:
        print(f"\n{'='*55}", flush=True)
        print(f"Task: {task}", flush=True)

        gallery_ids, umap_ids, perf_map = select_networks(task, thresholds)
        print(f"  Performance range: {min(perf_map.values()):.3f} – "
              f"{max(perf_map.values()):.3f}")

        print("  Loading data ...", flush=True)
        all_nets     = load_networks(task, umap_ids)
        gallery_nets = [all_nets[rid] for rid in gallery_ids]
        umap_nets    = [all_nets[rid] for rid in umap_ids]

        # Figure A: gallery
        print("  Gallery ...", flush=True)
        fig = make_gallery(task, gallery_nets)
        if fig:
            out = OUT_DIR / f"gallery_{task}.pdf"
            fig.savefig(out, bbox_inches="tight", dpi=130)
            plt.close(fig)
            print(f"    → {out.name}")

        # Figure B: similarity curves
        print("  Similarity curves ...", flush=True)
        fig = make_curves(task, gallery_nets)
        if fig:
            out = OUT_DIR / f"curves_{task}.pdf"
            fig.savefig(out, bbox_inches="tight", dpi=130)
            plt.close(fig)
            print(f"    → {out.name}")

        # Figure C: trajectory plots — MDS and UMAP (2D static)
        for ckpt_type in ["performance", "step", "epoch"]:
            attr = CKPT_ATTR[ckpt_type]
            if not any(nd[attr] for nd in umap_nets):
                print(f"  Trajectories [{ckpt_type}]: no data, skipping", flush=True)
                continue
            for method in ["mds", "umap"]:
                print(f"  Trajectories [{ckpt_type}] {method.upper()} ...", flush=True)
                fig = make_trajectory_fig(task, umap_nets, ckpt_type, method)
                if fig:
                    out = OUT_DIR / f"{method}_{task}_{ckpt_type}.pdf"
                    fig.savefig(out, bbox_inches="tight", dpi=130)
                    plt.close(fig)
                    print(f"    → {out.name}")

        # Figure C (3D): interactive step trajectories — MDS and UMAP
        if any(nd["step_ckpts"] for nd in umap_nets):
            for method in ["mds", "umap"]:
                print(f"  Trajectories [step] {method.upper()} 3D ...", flush=True)
                fig3d = make_trajectory_3d(task, umap_nets, method)
                if fig3d:
                    out = OUT_DIR / f"{method}_{task}_step_3d.html"
                    fig3d.write_html(str(out), include_plotlyjs="cdn")
                    print(f"    → {out.name}")

    print(f"\nDone. Figures in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
