"""Build RDMs from extracted LLM activations.

For each of the 122 activation npz files, computes Pearson correlation distance
RDMs (1 − r) using last-token activations at every layer. Three stimulus subsets
per layer: full (256×256), request (128×128), passage (128×128).

Output per model slug:
  rdms/{slug}_last_full.npz      — keys: {layer_label}: (256, 256) float32
  rdms/{slug}_last_request.npz   — keys: {layer_label}: (128, 128) float32
  rdms/{slug}_last_passage.npz   — keys: {layer_label}: (128, 128) float32
  rdms/{slug}_mean_full.npz      — mean-pool variants (same structure)
  rdms/{slug}_mean_request.npz
  rdms/{slug}_mean_passage.npz
  rdms/{slug}_last_alllayers.npz — keys: full, request, passage — single RDM from
  rdms/{slug}_mean_alllayers.npz   all layers concatenated per stimulus

Also saves a layer_labels array in each npz for convenience.

Visualisation:
  figures/rdm_{slug}_last_Lout.png   — final-layer full RDM for every model
  figures/rdm_gallery_pythia.pdf     — 6 rows × 16 cols
  figures/rdm_gallery_olmo.pdf       — 1 row × 16 cols

Usage:
    python scripts/build_llm_rdms.py [--skip-existing]
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import numpy as np
from scipy.spatial.distance import pdist, squareform

REPO_ROOT   = Path(__file__).parent.parent
ACT_DIR     = REPO_ROOT / "output" / "production" / "llm" / "activations"
RDM_DIR     = REPO_ROOT / "output" / "analysis" / "rdms" / "llm"
FIG_DIR         = REPO_ROOT / "output" / "analysis" / "figures" / "llm"
FINAL_LLM_RDM   = REPO_ROOT / "output" / "analysis_final" / "llm" / "figures" / "rdm_gallery"

N_REQUEST = 128   # rows 0–127
N_PASSAGE = 128   # rows 128–255


# ── RDM computation ───────────────────────────────────────────────────────────

def pearson_rdm(mat: np.ndarray) -> np.ndarray:
    """(n, d) → (n, n) Pearson correlation distance matrix."""
    # mean-centre across stimuli dimension then cosine = Pearson
    m = mat - mat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    m = m / norms
    rdm = 1.0 - (m @ m.T)
    np.fill_diagonal(rdm, 0.0)
    return rdm.astype(np.float32)


def build_rdms_for_file(act_path: Path, skip_existing: bool) -> dict[str, Path]:
    slug = act_path.stem
    out_paths = {
        "last_full":      RDM_DIR / f"{slug}_last_full.npz",
        "last_request":   RDM_DIR / f"{slug}_last_request.npz",
        "last_passage":   RDM_DIR / f"{slug}_last_passage.npz",
        "mean_full":      RDM_DIR / f"{slug}_mean_full.npz",
        "mean_request":   RDM_DIR / f"{slug}_mean_request.npz",
        "mean_passage":   RDM_DIR / f"{slug}_mean_passage.npz",
        "last_alllayers": RDM_DIR / f"{slug}_last_alllayers.npz",
        "mean_alllayers": RDM_DIR / f"{slug}_mean_alllayers.npz",
    }

    if skip_existing and all(p.exists() for p in out_paths.values()):
        print(f"[skip] {slug}")
        return out_paths

    data = np.load(act_path)
    layer_labels = list(data["layer_labels"])

    per_layer: dict[str, dict[str, np.ndarray]] = {k: {} for k in out_paths if "alllayers" not in k}
    all_layer_acts: dict[str, list[np.ndarray]] = {"last": [], "mean": []}

    for lbl in layer_labels:
        for pooling in ("last", "mean"):
            acts = data[f"{pooling}_{lbl}"]          # (256, d_model)
            req  = acts[:N_REQUEST]                   # (128, d_model)
            pas  = acts[N_REQUEST:]                   # (128, d_model)

            per_layer[f"{pooling}_full"][lbl]    = pearson_rdm(acts)
            per_layer[f"{pooling}_request"][lbl] = pearson_rdm(req)
            per_layer[f"{pooling}_passage"][lbl] = pearson_rdm(pas)

            all_layer_acts[pooling].append(acts)     # accumulate for concat

    lbl_arr = np.array(layer_labels)
    for key, path in out_paths.items():
        if "alllayers" in key:
            continue
        np.savez_compressed(str(path), layer_labels=lbl_arr, **per_layer[key])

    # All-layers-concatenated RDMs: (256, n_layers * d_model)
    for pooling in ("last", "mean"):
        cat = np.concatenate(all_layer_acts[pooling], axis=1)  # (256, n_layers*d_model)
        np.savez_compressed(
            str(out_paths[f"{pooling}_alllayers"]),
            full    = pearson_rdm(cat),
            request = pearson_rdm(cat[:N_REQUEST]),
            passage = pearson_rdm(cat[N_REQUEST:]),
        )

    print(f"[done] {slug}  ({len(layer_labels)} layers)")
    return out_paths


# ── Visualisation helpers ─────────────────────────────────────────────────────

def lout_label(layer_labels: list[str]) -> str:
    """Last transformer block label (not 'emb')."""
    return layer_labels[-1]


def rdm_image(ax, rdm: np.ndarray, title: str = ""):
    ax.imshow(rdm, cmap="viridis", vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=6, pad=2)


def save_individual_pngs(rdm_paths: dict[str, dict[str, Path]]):
    """Save one PNG per model: final-layer full RDM (last-token)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for slug, paths in rdm_paths.items():
        png_path = FIG_DIR / f"rdm_{slug}_last_Lout.png"
        if png_path.exists():
            continue
        d = np.load(paths["last_full"])
        layer_labels = list(d["layer_labels"])
        lbl = lout_label(layer_labels)
        rdm = d[lbl]
        fig, ax = plt.subplots(figsize=(4, 4))
        rdm_image(ax, rdm, title=f"{slug}\n{lbl}")
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)


# ── Gallery PDFs ──────────────────────────────────────────────────────────────

PYTHIA_SIZES = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b", "pythia-2.8b"]

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


def slug_for(model_id: str, revision: str) -> str:
    return model_id.replace("/", "__") + f"__{revision}"


def make_gallery_pdf(
    pdf_path: Path,
    rows: list[str],          # model_id strings (row labels)
    checkpoints: list[str],   # revision strings (column labels)
    rdm_paths: dict[str, dict[str, Path]],
    subset: str = "last_full",
):
    n_rows = len(rows)
    n_cols = len(checkpoints)
    cell = 1.2
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell, n_rows * cell + 0.4),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.05, wspace=0.05, top=0.95, bottom=0.02, left=0.12, right=0.99)

    for r, model_id in enumerate(rows):
        row_label = model_id.split("/")[-1]
        axes[r, 0].set_ylabel(row_label, fontsize=5, rotation=0, labelpad=40, va="center")
        for c, ckpt in enumerate(checkpoints):
            ax = axes[r, c]
            s = slug_for(model_id, ckpt)
            if s in rdm_paths and subset in rdm_paths[s]:
                d = np.load(rdm_paths[s][subset])
                layer_labels = list(d["layer_labels"])
                lbl = lout_label(layer_labels)
                rdm_image(ax, d[lbl])
            else:
                ax.set_visible(False)
            if r == 0:
                ax.set_title(ckpt, fontsize=4, rotation=45, ha="left", pad=2)

    pdf = pdf_backend.PdfPages(pdf_path)
    pdf.savefig(fig, dpi=150)
    pdf.close()
    FINAL_LLM_RDM.mkdir(parents=True, exist_ok=True)
    fig.savefig(FINAL_LLM_RDM / (pdf_path.stem + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[pdf] {pdf_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip RDM files that already exist (default: True)")
    parser.add_argument("--no-skip", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    RDM_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    act_files = sorted(ACT_DIR.glob("*.npz"))
    assert act_files, f"No activation files found in {ACT_DIR}"
    print(f"Found {len(act_files)} activation files")

    # Build RDMs
    all_rdm_paths: dict[str, dict[str, Path]] = {}
    for act_path in act_files:
        slug = act_path.stem
        paths = build_rdms_for_file(act_path, args.skip_existing)
        all_rdm_paths[slug] = paths

    # Individual PNGs
    print("Saving individual PNGs...")
    save_individual_pngs(all_rdm_paths)

    # Gallery PDFs
    print("Building Pythia gallery PDF...")
    make_gallery_pdf(
        FIG_DIR / "rdm_gallery_pythia.pdf",
        rows=[f"EleutherAI/{s}" for s in PYTHIA_SIZES],
        checkpoints=PYTHIA_CHECKPOINTS,
        rdm_paths=all_rdm_paths,
    )

    print("Building OLMo gallery PDF...")
    make_gallery_pdf(
        FIG_DIR / "rdm_gallery_olmo.pdf",
        rows=["allenai/OLMo-1B-0724-hf"],
        checkpoints=OLMO_CHECKPOINTS,
        rdm_paths=all_rdm_paths,
    )

    print("Done.")


if __name__ == "__main__":
    main()
