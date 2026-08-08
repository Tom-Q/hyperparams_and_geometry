"""Extract last-token hidden-state activations from a Pythia (or other HF) model.

For each of 128 stimuli, runs a forward pass and saves the last-token representation
at selected layers. Processes one stimulus at a time (no batching) to minimise memory.

Model weights are downloaded into a temporary directory and deleted after extraction,
so only one checkpoint is on disk at a time (~140 MB for pythia-70m, ~5.6 GB for pythia-2.8b).

Usage — Pythia training dynamics (Lout only, one checkpoint):
    python scripts/extract_llm_activations.py \\
        --model-id EleutherAI/pythia-70m \\
        --revision step143000 \\
        --layers lout

Usage — final-checkpoint cross-model analysis (emb + L50 + L75 + Lout):
    python scripts/extract_llm_activations.py \\
        --model-id EleutherAI/pythia-70m \\
        --revision step143000 \\
        --layers all

Usage — force CPU (required for pythia-2.8b):
    python scripts/extract_llm_activations.py \\
        --model-id EleutherAI/pythia-2.8b \\
        --revision step143000 \\
        --layers lout \\
        --device cpu
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

STIMULI_PATH = REPO_ROOT / "data" / "stimuli_v1.1.json"
OUTPUT_DIR   = REPO_ROOT / "output" / "production" / "llm" / "activations"


def model_slug(model_id: str, revision: str) -> str:
    """Filesystem-safe identifier: slashes → __, revision appended."""
    return model_id.replace("/", "__") + f"__{revision}"


def select_layer_indices(n_layers: int, mode: str) -> dict[str, int]:
    """
    Map layer labels to hidden_states tuple indices.
    hidden_states[0] = embedding output
    hidden_states[1..n_layers] = transformer block outputs
    """
    if mode == "lout":
        return {"Lout": n_layers}
    else:  # "all"
        return {
            "emb": 0,
            "L50": round(0.5 * n_layers),
            "L75": round(0.75 * n_layers),
            "Lout": n_layers,
        }


def load_stimuli(path: Path) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    return [s["text"] for s in data["stimuli"]]


def extract(model_id: str, revision: str, device: str, layers_mode: str, dry_run: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    slug = model_slug(model_id, revision)
    out_path = OUTPUT_DIR / f"{slug}.npz"
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return

    stimuli = load_stimuli(STIMULI_PATH)
    print(f"Stimuli: {len(stimuli)}")
    print(f"Model:   {model_id}  revision={revision}")
    print(f"Device:  {device}  layers={layers_mode}")

    if dry_run:
        print("[dry-run] exiting")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hf_llm_") as tmp:
        print(f"Downloading to {tmp} ...")
        t0 = time.time()

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, cache_dir=tmp,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=tmp,
            torch_dtype=torch.bfloat16,
        ).to(device)
        model.eval()

        n_layers = model.config.num_hidden_layers
        layer_map = select_layer_indices(n_layers, layers_mode)
        print(f"Layers ({n_layers} transformer blocks): {layer_map}")
        print(f"Download+load: {time.time()-t0:.1f}s")

        d_model = model.config.hidden_size
        n_stimuli = len(stimuli)

        # activations[layer_label][stimulus_idx] = (d_model,) float32
        all_acts = {label: np.zeros((n_stimuli, d_model), dtype=np.float32)
                    for label in layer_map}

        t1 = time.time()
        for i, text in enumerate(stimuli):
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # hidden_states: tuple of (1, seq_len, d_model), one per layer+emb
            for label, idx in layer_map.items():
                # last token, all hidden units; cast to float32 for storage
                vec = outputs.hidden_states[idx][0, -1, :].float().cpu().numpy()
                all_acts[label][i] = vec

            if (i + 1) % 32 == 0 or i == n_stimuli - 1:
                elapsed = time.time() - t1
                print(f"  {i+1}/{n_stimuli}  ({elapsed:.1f}s)", flush=True)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"Extraction: {time.time()-t1:.1f}s")

    # tmp dir (and downloaded weights) are now deleted

    layer_labels = list(layer_map.keys())
    layer_indices = np.array(list(layer_map.values()), dtype=np.int32)

    np.savez_compressed(
        str(out_path),
        **{f"hidden_{label}": all_acts[label] for label in layer_labels},
        layer_labels  = np.array(layer_labels),
        layer_indices = layer_indices,
        n_layers      = np.int32(n_layers),
        d_model       = np.int32(d_model),
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved {out_path.name}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id",  required=True)
    parser.add_argument("--revision",  default="main",
                        help="HF revision: 'main' or e.g. 'step143000' for Pythia checkpoints")
    parser.add_argument("--layers",    choices=["lout", "all"], default="lout",
                        help="lout: final block only (training dynamics); all: emb+L50+L75+Lout (cross-model)")
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()

    extract(args.model_id, args.revision, args.device, args.layers, args.dry_run)


if __name__ == "__main__":
    main()
