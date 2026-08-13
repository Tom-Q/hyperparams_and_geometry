"""
A.1–A.7  Cross-model RSA analysis — 17 final-checkpoint LLMs.

Runs all seven analyses in sequence. A.1 (cross-model agreement) and A.4
(depth profiles) together determine the best layer, which is then used for
A.5–A.7. Best-layer selection is printed so it can be inspected and overridden
via --best-layer if needed.

Usage:
    python analysis/41_llm_cross_model.py
    python analysis/41_llm_cross_model.py --best-layer L24
    python analysis/41_llm_cross_model.py --pooling mean
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from scipy.spatial.distance import squareform

ANALYSIS  = Path(__file__).parent
REPO_ROOT = ANALYSIS.parent
sys.path.insert(0, str(ANALYSIS))

from analysis_utils import FIGURES_DIR, TABLES_DIR

ACT_DIR  = REPO_ROOT / "output" / "production" / "llm" / "activations"
RDM_DIR  = REPO_ROOT / "output" / "analysis" / "rdms" / "llm"
FIG_DIR  = FIGURES_DIR / "llm"
TAB_DIR  = TABLES_DIR

N_REQUEST = 128
N_PASSAGE = 128


# ── Model registry ─────────────────────────────────────────────────────────────

FINAL_MODELS = [
    dict(model_id="EleutherAI/pythia-70m",              revision="step143000",              family="pythia",  kind="base"),
    dict(model_id="EleutherAI/pythia-160m",             revision="step143000",              family="pythia",  kind="base"),
    dict(model_id="EleutherAI/pythia-410m",             revision="step143000",              family="pythia",  kind="base"),
    dict(model_id="EleutherAI/pythia-1b",               revision="step143000",              family="pythia",  kind="base"),
    dict(model_id="EleutherAI/pythia-1.4b",             revision="step143000",              family="pythia",  kind="base"),
    dict(model_id="EleutherAI/pythia-2.8b",             revision="step143000",              family="pythia",  kind="base"),
    dict(model_id="meta-llama/Llama-3.2-1B",            revision="main",                    family="llama",   kind="base"),
    dict(model_id="meta-llama/Llama-3.2-1B-Instruct",   revision="main",                    family="llama",   kind="instruct"),
    dict(model_id="Qwen/Qwen2.5-1.5B",                  revision="main",                    family="qwen",    kind="base"),
    dict(model_id="Qwen/Qwen2.5-1.5B-Instruct",         revision="main",                    family="qwen",    kind="instruct"),
    dict(model_id="HuggingFaceTB/SmolLM2-1.7B",         revision="main",                    family="smollm", kind="base"),
    dict(model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",revision="main",                    family="smollm", kind="instruct"),
    dict(model_id="google/gemma-2-2b",                  revision="main",                    family="gemma",   kind="base"),
    dict(model_id="google/gemma-2-2b-it",               revision="main",                    family="gemma",   kind="instruct"),
    dict(model_id="allenai/OLMo-1B-0724-hf",            revision="step1454000-tokens3048B", family="olmo",    kind="base"),
    dict(model_id="allenai/OLMo-2-0425-1B-SFT",         revision="main",                    family="olmo",    kind="sft"),
    dict(model_id="allenai/OLMo-2-0425-1B-Instruct",    revision="main",                    family="olmo",    kind="instruct"),
]

BASE_INSTRUCT_PAIRS = [
    ("meta-llama/Llama-3.2-1B",     "meta-llama/Llama-3.2-1B-Instruct"),
    ("Qwen/Qwen2.5-1.5B",           "Qwen/Qwen2.5-1.5B-Instruct"),
    ("HuggingFaceTB/SmolLM2-1.7B",  "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ("google/gemma-2-2b",            "google/gemma-2-2b-it"),
    ("allenai/OLMo-1B-0724-hf",      "allenai/OLMo-2-0425-1B-Instruct"),
]

FAMILY_COLORS = {
    "pythia":  "#4e79a7",
    "llama":   "#f28e2b",
    "qwen":    "#e15759",
    "smollm":  "#76b7b2",
    "gemma":   "#59a14f",
    "olmo":    "#b07aa1",
}

KIND_MARKERS = {"base": "o", "instruct": "s", "sft": "^"}


def slug(model_id, revision):
    return model_id.replace("/", "__") + f"__{revision}"


def short_name(model_id, revision=None):
    name = model_id.split("/")[-1]
    if revision and revision != "main" and "step" in revision:
        name += f" ({revision})"
    return name


# ── Layer selection ─────────────────────────────────────────────────────────────

def pct_layer_label(layer_labels, pct):
    """Label of the transformer block at the given percentile (0–100).
    Percentile is over transformer blocks only (L1…Ln); 'emb' excluded.
    pct=100 returns the last transformer block."""
    transformer = [l for l in layer_labels if l != "emb"]
    idx = round(pct / 100 * (len(transformer) - 1))
    return transformer[idx]


def layer_set_a1(layer_labels):
    """10th, 50th, 90th percentile + last 3 transformer layers (deduplicated)."""
    transformer = [l for l in layer_labels if l != "emb"]
    pct_layers = [pct_layer_label(layer_labels, p) for p in [10, 50, 90]]
    last3 = transformer[-3:]
    seen, result = set(), []
    for l in pct_layers + last3:
        if l not in seen:
            seen.add(l)
            result.append(l)
    return result


def layer_set_a2(layer_labels):
    """10th, 30th, 50th, 70th, 90th percentile + last (deduplicated)."""
    pct_layers = [pct_layer_label(layer_labels, p) for p in [10, 30, 50, 70, 90]]
    last = [l for l in layer_labels if l != "emb"][-1]
    seen, result = set(), []
    for l in pct_layers + [last]:
        if l not in seen:
            seen.add(l)
            result.append(l)
    return result


def normalized_depth(layer_label, layer_labels):
    """Normalized position in [0, 1]: emb=0, last transformer block=1."""
    transformer = [l for l in layer_labels if l != "emb"]
    if layer_label == "emb":
        return 0.0
    idx = transformer.index(layer_label)
    return (idx + 1) / len(transformer)


# ── RDM utilities ───────────────────────────────────────────────────────────────

def upper_tri(rdm):
    """Return the strictly upper-triangular elements as a 1D array."""
    n = rdm.shape[0]
    idx = np.triu_indices(n, k=1)
    return rdm[idx]


def rdm_corr(a, b):
    """Pearson r between upper triangles of two RDMs."""
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


def load_activation(model_id, revision, pooling, layer_label):
    s = slug(model_id, revision)
    path = ACT_DIR / f"{s}.npz"
    d = np.load(path)
    return d[f"{pooling}_{layer_label}"].astype(np.float32)  # (256, d_model)


def get_layer_labels(model_id, revision):
    s = slug(model_id, revision)
    path = RDM_DIR / f"{s}_last_full.npz"
    d = np.load(path)
    return list(d["layer_labels"])


# ── Ideal RDMs ─────────────────────────────────────────────────────────────────

def binary_ideal_rdm(labels):
    """0 if same label, 1 if different."""
    a = np.array(labels)
    return (a[:, None] != a[None, :]).astype(np.float32)


def load_stimuli():
    path = REPO_ROOT / "docs" / "stimuli_v5.1.json"
    with open(path) as f:
        data = json.load(f)
    return data["stimuli"]


def build_ideal_rdms(stimuli):
    req = stimuli[:N_REQUEST]
    pas = stimuli[N_REQUEST:]
    return {
        "request_theme": binary_ideal_rdm([s["theme"] for s in req]),
        "request_task":  binary_ideal_rdm([s["task"]  for s in req]),
        "passage_theme": binary_ideal_rdm([s["theme"] for s in pas]),
        "passage_task":  binary_ideal_rdm([s["task"]  for s in pas]),
    }


# ── A.1 Cross-model agreement ───────────────────────────────────────────────────

def compute_cross_model_agreement(pooling, layer_label, subset):
    """Leave-one-out: each model vs mean of others. Returns mean ± SD over 17 models."""
    rdms = np.stack([
        load_rdm(m["model_id"], m["revision"], pooling, subset, layer_label)
        for m in FINAL_MODELS
    ])  # (17, N, N)
    n = len(rdms)
    scores = []
    for i in range(n):
        others = np.concatenate([rdms[:i], rdms[i+1:]], axis=0)
        mean_rdm = others.mean(axis=0)
        scores.append(rdm_corr(rdms[i], mean_rdm))
    return np.mean(scores), np.std(scores), scores


def run_a1(pooling):
    print("\n── A.1 Cross-model agreement ──────────────────────────────────────")
    records = []
    for m in FINAL_MODELS:
        layer_labels = get_layer_labels(m["model_id"], m["revision"])
        m["_layer_labels"] = layer_labels
        m["_a1_layers"] = layer_set_a1(layer_labels)

    # Compute agreement at each candidate layer for each subset
    # We iterate over the union of candidate layers across all models (by percentile)
    candidate_pcts = [10, 50, 90]          # percentile layers
    subsets = ["request", "passage", "full"]

    for pct in candidate_pcts:
        for subset in subsets:
            # For each model, get the layer at this percentile and load its RDM
            # Then pool cross-model agreement using each model's own percentile layer
            # (since layers differ in number, we compute model-specific percentile layers
            #  and then do LOO using same-percentile layers across models)
            rdms = []
            layer_lbls_used = []
            for m in FINAL_MODELS:
                lbl = pct_layer_label(m["_layer_labels"], pct)
                rdms.append(load_rdm(m["model_id"], m["revision"], pooling, subset, lbl))
                layer_lbls_used.append(lbl)
            rdms = np.stack(rdms)  # (17, N, N)
            loo_scores = []
            for i in range(len(rdms)):
                others = np.concatenate([rdms[:i], rdms[i+1:]], axis=0)
                loo_scores.append(rdm_corr(rdms[i], others.mean(axis=0)))
            mean_r, sd_r = np.mean(loo_scores), np.std(loo_scores)
            print(f"  pct={pct:3d}  subset={subset:8s}  r={mean_r:.3f} ± {sd_r:.3f}")
            for i, m in enumerate(FINAL_MODELS):
                records.append(dict(
                    model=short_name(m["model_id"]),
                    family=m["family"], kind=m["kind"],
                    pct=pct, subset=subset,
                    layer=layer_lbls_used[i],
                    agreement=loo_scores[i],
                ))

    # Also last 3 layers (absolute, not percentile)
    # Use each model's own last-3 layers; for cross-model, align by position from end
    for pos_from_end, pos_label in enumerate(["Lout", "Lout-1", "Lout-2"]):
        for subset in subsets:
            rdms = []
            layer_lbls_used = []
            for m in FINAL_MODELS:
                transformer = [l for l in m["_layer_labels"] if l != "emb"]
                lbl = transformer[-(pos_from_end + 1)]
                rdms.append(load_rdm(m["model_id"], m["revision"], pooling, subset, lbl))
                layer_lbls_used.append(lbl)
            rdms = np.stack(rdms)
            loo_scores = []
            for i in range(len(rdms)):
                others = np.concatenate([rdms[:i], rdms[i+1:]], axis=0)
                loo_scores.append(rdm_corr(rdms[i], others.mean(axis=0)))
            mean_r, sd_r = np.mean(loo_scores), np.std(loo_scores)
            print(f"  {pos_label:8s}  subset={subset:8s}  r={mean_r:.3f} ± {sd_r:.3f}")
            for i, m in enumerate(FINAL_MODELS):
                records.append(dict(
                    model=short_name(m["model_id"]),
                    family=m["family"], kind=m["kind"],
                    pct=pos_label, subset=subset,
                    layer=layer_lbls_used[i],
                    agreement=loo_scores[i],
                ))

    df = pd.DataFrame(records)
    df.to_csv(TAB_DIR / "llm_a1_cross_model_agreement.csv", index=False)
    return df


# ── A.2 Category structure ──────────────────────────────────────────────────────

def run_a2(pooling, ideal_rdms):
    print("\n── A.2 Category structure ─────────────────────────────────────────")
    records = []

    for m in FINAL_MODELS:
        layer_labels = m["_layer_labels"]
        layers = layer_set_a2(layer_labels)
        for lbl in layers:
            depth = normalized_depth(lbl, layer_labels)
            for subset, ideal_key_prefix in [("request", "request"), ("passage", "passage")]:
                rdm = load_rdm(m["model_id"], m["revision"], pooling, subset, lbl)
                for level in ["theme", "task"]:
                    ideal = ideal_rdms[f"{ideal_key_prefix}_{level}"]
                    tau, pval = kendalltau(upper_tri(rdm), upper_tri(ideal))
                    records.append(dict(
                        model=short_name(m["model_id"]),
                        family=m["family"], kind=m["kind"],
                        layer=lbl, depth=depth, subset=subset, level=level,
                        tau=tau, pval=pval,
                    ))

    df = pd.DataFrame(records)
    df.to_csv(TAB_DIR / "llm_a2_category_structure.csv", index=False)

    # Print summary at last layer
    for subset in ["request", "passage"]:
        for level in ["theme", "task"]:
            sub = df[(df.subset == subset) & (df.level == level)]
            last_layer = sub.groupby("model", sort=False).apply(lambda x: x.loc[x.depth.idxmax()])
            mean_tau = last_layer["tau"].mean()
            print(f"  {subset:8s}  {level:5s}  τ at Lout: {mean_tau:.3f} (mean over 17 models)")

    return df


# ── A.3 Effective dimensionality ────────────────────────────────────────────────

def participation_ratio(acts):
    """(n, d) → scalar participation ratio of covariance eigenspectrum."""
    cov = np.cov(acts.T)                     # (d, d)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 0]
    return float(eigvals.sum() ** 2 / (eigvals ** 2).sum())


def run_a3(pooling):
    print("\n── A.3 Effective dimensionality ───────────────────────────────────")
    records = []

    for m in FINAL_MODELS:
        layer_labels = m["_layer_labels"]
        layers = layer_set_a2(layer_labels)   # same layer set as A.2
        for lbl in layers:
            depth = normalized_depth(lbl, layer_labels)
            acts = load_activation(m["model_id"], m["revision"], pooling, lbl)
            for subset, slc in [("request", slice(0, N_REQUEST)),
                                 ("passage", slice(N_REQUEST, None))]:
                pr = participation_ratio(acts[slc])
                records.append(dict(
                    model=short_name(m["model_id"]),
                    family=m["family"], kind=m["kind"],
                    layer=lbl, depth=depth, subset=subset, pr=pr,
                ))

    df = pd.DataFrame(records)
    df.to_csv(TAB_DIR / "llm_a3_dimensionality.csv", index=False)

    for subset in ["request", "passage"]:
        sub = df[df.subset == subset]
        last = sub.groupby("model", sort=False).apply(lambda x: x.loc[x.depth.idxmax()])
        print(f"  {subset:8s}  PR at Lout: {last['pr'].mean():.1f} ± {last['pr'].std():.1f}")

    return df


# ── A.4 Depth profiles ──────────────────────────────────────────────────────────

def run_a4(pooling, ideal_rdms):
    print("\n── A.4 Depth profiles (all layers) ────────────────────────────────")
    records = []

    for m in FINAL_MODELS:
        layer_labels = m["_layer_labels"]
        for lbl in layer_labels:
            depth = normalized_depth(lbl, layer_labels)
            for subset in ["request", "passage"]:
                rdm = load_rdm(m["model_id"], m["revision"], pooling, subset, lbl)
                # Category structure
                for level in ["theme", "task"]:
                    ideal = ideal_rdms[f"{subset}_{level}"]
                    tau, _ = kendalltau(upper_tri(rdm), upper_tri(ideal))
                    records.append(dict(
                        model=short_name(m["model_id"]),
                        family=m["family"], kind=m["kind"],
                        layer=lbl, depth=depth, subset=subset,
                        metric="tau_" + level, value=float(tau),
                    ))
                # Mean dissimilarity
                records.append(dict(
                    model=short_name(m["model_id"]),
                    family=m["family"], kind=m["kind"],
                    layer=lbl, depth=depth, subset=subset,
                    metric="mean_dist", value=float(upper_tri(rdm).mean()),
                ))

    df = pd.DataFrame(records)
    df.to_csv(TAB_DIR / "llm_a4_depth_profiles.csv", index=False)
    return df


# ── Best-layer selection ────────────────────────────────────────────────────────

def select_best_layer(df_a1, df_a4, override=None):
    """Print and return the best layer label (position-from-end notation)."""
    if override:
        print(f"\n── Layer selection: override → {override}")
        return override

    # From A.1: highest mean cross-model agreement on passage, at Lout/Lout-1/Lout-2 or pct layers
    passage_a1 = df_a1[df_a1.subset == "passage"].groupby("pct")["agreement"].mean()
    best_pct_by_agreement = passage_a1.idxmax()

    # From A.4: highest mean τ_theme across all models, averaging request+passage
    tau = df_a4[df_a4.metric == "tau_theme"].groupby(["model", "depth"])["value"].mean().reset_index()
    # For each model, find depth of max τ
    best_depths = tau.groupby("model").apply(lambda x: x.loc[x.value.idxmax(), "depth"])
    median_best_depth = float(best_depths.median())

    print(f"\n── Layer selection ────────────────────────────────────────────────")
    print(f"  Cross-model agreement (A.1): best at pct={best_pct_by_agreement}")
    print(f"  Category structure τ (A.4):  median best depth = {median_best_depth:.2f}")

    # Use Lout as the summary layer for A.5-A.7 (deepest, theory-neutral tie-breaker)
    # and report which agrees with A.1/A.4 criterion
    selected = "Lout"
    print(f"  → Using {selected} (last transformer layer) for A.5–A.7")
    print(f"    Override with --best-layer if you prefer a different layer.")
    return selected


def resolve_layer_label(model_id, revision, position_or_label):
    """Resolve 'Lout', 'Lout-1', 'Lout-2', or an absolute label like 'L24'."""
    layer_labels = get_layer_labels(model_id, revision)
    transformer = [l for l in layer_labels if l != "emb"]
    if position_or_label == "Lout":
        return transformer[-1]
    elif position_or_label == "Lout-1":
        return transformer[-2]
    elif position_or_label == "Lout-2":
        return transformer[-3]
    else:
        assert position_or_label in layer_labels, \
            f"Layer {position_or_label!r} not found in {layer_labels}"
        return position_or_label


# ── A.5 Base vs instruct comparison ────────────────────────────────────────────

def run_a5(pooling, best_layer_pos, ideal_rdms):
    print("\n── A.5 Base vs instruct comparison ────────────────────────────────")
    records = []

    for base_id, inst_id in BASE_INSTRUCT_PAIRS:
        base_m = next(m for m in FINAL_MODELS if m["model_id"] == base_id)
        inst_m = next(m for m in FINAL_MODELS if m["model_id"] == inst_id)

        base_lbl = resolve_layer_label(base_id, base_m["revision"], best_layer_pos)
        inst_lbl = resolve_layer_label(inst_id, inst_m["revision"], best_layer_pos)

        for subset in ["request", "passage"]:
            rdm_base = load_rdm(base_id, base_m["revision"], pooling, subset, base_lbl)
            rdm_inst = load_rdm(inst_id, inst_m["revision"], pooling, subset, inst_lbl)

            # RDM similarity between base and instruct
            sim = rdm_corr(rdm_base, rdm_inst)

            # Category structure for each
            for level in ["theme", "task"]:
                ideal = ideal_rdms[f"{subset}_{level}"]
                tau_base, _ = kendalltau(upper_tri(rdm_base), upper_tri(ideal))
                tau_inst, _ = kendalltau(upper_tri(rdm_inst), upper_tri(ideal))
                records.append(dict(
                    pair=f"{short_name(base_id)} / {short_name(inst_id)}",
                    family=base_m["family"],
                    subset=subset, level=level,
                    tau_base=float(tau_base), tau_inst=float(tau_inst),
                    base_instruct_rdm_sim=sim,
                ))

        pair_label = short_name(base_id).split("-")[0]
        print(f"  {pair_label}: RDM sim request={rdm_corr(load_rdm(base_id, base_m['revision'], pooling, 'request', base_lbl), load_rdm(inst_id, inst_m['revision'], pooling, 'request', inst_lbl)):.3f}  passage={rdm_corr(load_rdm(base_id, base_m['revision'], pooling, 'passage', base_lbl), load_rdm(inst_id, inst_m['revision'], pooling, 'passage', inst_lbl)):.3f}")

    df = pd.DataFrame(records)
    df.to_csv(TAB_DIR / "llm_a5_base_instruct.csv", index=False)
    return df


# ── A.6 Summary table ───────────────────────────────────────────────────────────

def run_a6(df_a1, df_a2, df_a3, best_layer_pos):
    print("\n── A.6 Summary table ───────────────────────────────────────────────")

    def last_layer_val(df, model, subset, col, metric=None):
        sub = df[df.model == model]
        if "subset" in sub.columns:
            sub = sub[sub.subset == subset]
        if metric is not None:
            sub = sub[sub.metric == metric] if "metric" in sub.columns else sub[sub.level == metric]
        if sub.empty:
            return np.nan
        return float(sub.loc[sub.depth.idxmax(), col])

    records = []
    for m in FINAL_MODELS:
        name = short_name(m["model_id"])
        a1_passage = df_a1[(df_a1.model == name) & (df_a1.subset == "passage") & (df_a1.pct == "Lout")]["agreement"]
        a1_request = df_a1[(df_a1.model == name) & (df_a1.subset == "request") & (df_a1.pct == "Lout")]["agreement"]

        rec = dict(
            model=name, family=m["family"], kind=m["kind"],
            n_layers=len([l for l in m["_layer_labels"] if l != "emb"]),
        )
        for subset in ["request", "passage"]:
            a1_sub = df_a1[(df_a1.model == name) & (df_a1.subset == subset) & (df_a1.pct == "Lout")]
            rec[f"agreement_{subset}"] = float(a1_sub["agreement"].iloc[0]) if not a1_sub.empty else np.nan

            a2_sub = df_a2[(df_a2.model == name) & (df_a2.subset == subset)]
            for level in ["theme", "task"]:
                last = a2_sub[a2_sub.level == level]
                if not last.empty:
                    rec[f"tau_{level}_{subset}"] = float(last.loc[last.depth.idxmax(), "tau"])

            a3_sub = df_a3[(df_a3.model == name) & (df_a3.subset == subset)]
            if not a3_sub.empty:
                rec[f"pr_{subset}"] = float(a3_sub.loc[a3_sub.depth.idxmax(), "pr"])

        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(TAB_DIR / "llm_a6_summary.csv", index=False)
    print(df[["model", "family", "kind", "agreement_request", "agreement_passage",
              "tau_theme_request", "tau_theme_passage"]].to_string(index=False))
    return df


# ── A.7 Cross-model RSA matrix ──────────────────────────────────────────────────

def run_a7(pooling, best_layer_pos):
    print("\n── A.7 Cross-model RSA matrix ──────────────────────────────────────")
    from sklearn.manifold import MDS

    records = []
    n = len(FINAL_MODELS)
    names = [short_name(m["model_id"]) for m in FINAL_MODELS]

    for subset in ["request", "passage", "full"]:
        mat = np.zeros((n, n))
        rdms = []
        for m in FINAL_MODELS:
            lbl = resolve_layer_label(m["model_id"], m["revision"], best_layer_pos)
            rdms.append(load_rdm(m["model_id"], m["revision"], pooling, subset, lbl))
        for i in range(n):
            for j in range(i, n):
                r = rdm_corr(rdms[i], rdms[j])
                mat[i, j] = mat[j, i] = r

        dissim = 1 - mat
        np.fill_diagonal(dissim, 0)

        df_mat = pd.DataFrame(mat, index=names, columns=names)
        df_mat.to_csv(TAB_DIR / f"llm_a7_crossmodel_{subset}.csv")

        for i, m in enumerate(FINAL_MODELS):
            for j, m2 in enumerate(FINAL_MODELS):
                records.append(dict(
                    model_a=names[i], model_b=names[j],
                    family_a=m["family"], family_b=m2["family"],
                    kind_a=m["kind"], kind_b=m2["kind"],
                    subset=subset, r=mat[i, j],
                ))

    return pd.DataFrame(records)


# ── Figures ─────────────────────────────────────────────────────────────────────

def plot_a1(df_a1):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    pct_order = [10, 50, 90, "Lout-2", "Lout-1", "Lout"]
    x_ticks = list(range(len(pct_order)))

    for ax, subset in zip(axes, ["request", "passage", "full"]):
        sub = df_a1[df_a1.subset == subset]
        means = [sub[sub.pct == p]["agreement"].mean() for p in pct_order]
        sds   = [sub[sub.pct == p]["agreement"].std()  for p in pct_order]
        ax.bar(x_ticks, means, yerr=sds, capsize=4, color="#4e79a7", alpha=0.8)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(p) for p in pct_order], fontsize=8)
        ax.set_xlabel("Layer position")
        ax.set_title(f"Subset: {subset}")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Cross-model agreement (r)")
    fig.suptitle("A.1 Cross-model agreement across layer positions", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "A1_cross_model_agreement.pdf")
    plt.close(fig)


def plot_a4(df_a4):
    models = [short_name(m["model_id"]) for m in FINAL_MODELS]
    n_models = len(models)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols

    for subset in ["request", "passage"]:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5),
                                  squeeze=False)
        for ax_idx, (m, name) in enumerate(zip(FINAL_MODELS, models)):
            ax = axes[ax_idx // n_cols][ax_idx % n_cols]
            sub = df_a4[(df_a4.model == name) & (df_a4.subset == subset)]
            depths = sub[sub.metric == "tau_theme"]["depth"].values
            sort_idx = np.argsort(depths)
            for metric, color, label in [
                ("tau_theme", "#4e79a7", "τ theme"),
                ("tau_task",  "#f28e2b", "τ task"),
            ]:
                vals = sub[sub.metric == metric]["value"].values
                ax.plot(depths[sort_idx], vals[sort_idx], color=color, lw=1.5, label=label)
            ax.axhline(0, color="gray", lw=0.5, ls="--")
            ax.set_title(name, fontsize=7)
            ax.set_xlabel("Norm. depth", fontsize=7)
            ax.set_ylim(-0.1, 0.4)
            if ax_idx == 0:
                ax.legend(fontsize=6)

        for ax_idx in range(n_models, n_rows * n_cols):
            axes[ax_idx // n_cols][ax_idx % n_cols].set_visible(False)

        fig.suptitle(f"A.4 Category structure depth profile — {subset}", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"A4_depth_profiles_{subset}.pdf")
        plt.close(fig)


def plot_a7(pooling, best_layer_pos):
    from sklearn.manifold import MDS

    n = len(FINAL_MODELS)
    names = [short_name(m["model_id"]) for m in FINAL_MODELS]
    colors = [FAMILY_COLORS[m["family"]] for m in FINAL_MODELS]
    markers = [KIND_MARKERS.get(m["kind"], "D") for m in FINAL_MODELS]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    for col, subset in enumerate(["request", "passage", "full"]):
        mat = np.zeros((n, n))
        rdms = [load_rdm(m["model_id"], m["revision"], pooling, subset,
                         resolve_layer_label(m["model_id"], m["revision"], best_layer_pos))
                for m in FINAL_MODELS]
        for i in range(n):
            for j in range(i, n):
                r = rdm_corr(rdms[i], rdms[j])
                mat[i, j] = mat[j, i] = r

        # Heatmap
        ax = axes[0, col]
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=5)
        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=5)
        ax.set_title(f"RDM correlations — {subset}", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # MDS
        ax = axes[1, col]
        dissim = np.clip(1 - mat, 0, None)
        np.fill_diagonal(dissim, 0)
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress=False)
        coords = mds.fit_transform(dissim)
        for i, (m, name) in enumerate(zip(FINAL_MODELS, names)):
            ax.scatter(coords[i, 0], coords[i, 1],
                       color=FAMILY_COLORS[m["family"]],
                       marker=KIND_MARKERS.get(m["kind"], "D"),
                       s=60, zorder=3)
            ax.annotate(name, coords[i], fontsize=5, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_title(f"MDS — {subset}", fontsize=8)
        ax.set_xlabel("MDS1", fontsize=7)
        ax.set_ylabel("MDS2", fontsize=7)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=fam)
        for fam, c in FAMILY_COLORS.items()
    ] + [
        Line2D([0], [0], marker=mk, color="gray", markersize=8, label=kind)
        for kind, mk in KIND_MARKERS.items()
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=len(legend_elements),
               fontsize=7, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(f"A.7 Cross-model RSA ({pooling}-pool, {best_layer_pos})", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(FIG_DIR / "A7_cross_model_matrix.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_a5(df_a5):
    pairs = df_a5["pair"].unique()
    subsets = ["request", "passage"]
    levels  = ["theme", "task"]

    fig, axes = plt.subplots(len(subsets), len(levels),
                              figsize=(len(levels) * 4, len(subsets) * 3.5),
                              squeeze=False)

    x = np.arange(len(pairs))
    width = 0.35

    for row, subset in enumerate(subsets):
        for col, level in enumerate(levels):
            ax = axes[row, col]
            sub = df_a5[(df_a5.subset == subset) & (df_a5.level == level)].set_index("pair")
            tau_base = [sub.loc[p, "tau_base"] if p in sub.index else 0 for p in pairs]
            tau_inst = [sub.loc[p, "tau_inst"] if p in sub.index else 0 for p in pairs]
            ax.bar(x - width/2, tau_base, width, label="base",    color="#4e79a7", alpha=0.85)
            ax.bar(x + width/2, tau_inst, width, label="instruct", color="#f28e2b", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([p.split("/")[0] for p in pairs], rotation=30, ha="right", fontsize=8)
            ax.set_title(f"{subset} × {level}", fontsize=9)
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_ylabel("Kendall τ")
            if row == 0 and col == 0:
                ax.legend()

    fig.suptitle("A.5 Base vs instruct — category structure", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "A5_base_instruct.pdf")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooling",    default="last", choices=["last", "mean"])
    parser.add_argument("--best-layer", default=None,
                        help="Override automatic layer selection, e.g. 'Lout' or 'L24'")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    stimuli = load_stimuli()
    ideal_rdms = build_ideal_rdms(stimuli)

    # Pre-fetch layer labels for all models (stored in FINAL_MODELS dicts)
    for m in FINAL_MODELS:
        if "_layer_labels" not in m:
            m["_layer_labels"] = get_layer_labels(m["model_id"], m["revision"])

    df_a1 = run_a1(args.pooling)
    df_a2 = run_a2(args.pooling, ideal_rdms)
    df_a3 = run_a3(args.pooling)
    df_a4 = run_a4(args.pooling, ideal_rdms)

    best_layer = select_best_layer(df_a1, df_a4, override=args.best_layer)

    df_a5 = run_a5(args.pooling, best_layer, ideal_rdms)
    df_a6 = run_a6(df_a1, df_a2, df_a3, best_layer)
    df_a7 = run_a7(args.pooling, best_layer)

    print("\n── Saving figures ──────────────────────────────────────────────────")
    plot_a1(df_a1)
    plot_a4(df_a4)
    plot_a5(df_a5)
    plot_a7(args.pooling, best_layer)

    print("Done.")


if __name__ == "__main__":
    main()
