"""Extract hidden-state activations from a Pythia (or other HF) model.

For each of 256 stimuli (128 request + 128 passage), runs a forward pass and
saves activations at every layer (embedding + all transformer blocks).
Processes one stimulus at a time (no batching) to minimise memory.

Model weights are downloaded into a temporary directory and deleted after
extraction, so only one checkpoint is on disk at a time.

Output npz keys per model:
  last_emb, last_L1 … last_L{n}   (256, d_model) float32  — last-token
  mean_emb, mean_L1 … mean_L{n}   (256, d_model) float32  — mean-pool
  token_counts                     (256,)          int32   — tokens per stimulus
  nll                              (256,)          float32 — mean per-token NLL
  layer_labels, n_layers, d_model  — metadata

Usage — Pythia checkpoint:
    python scripts/extract_llm_activations.py \\
        --model-id EleutherAI/pythia-70m \\
        --revision step143000

Usage — cross-family model:
    python scripts/extract_llm_activations.py \\
        --model-id meta-llama/Llama-3.2-1B

Usage — force CPU (required for pythia-2.8b and large cross-family models):
    python scripts/extract_llm_activations.py \\
        --model-id EleutherAI/pythia-2.8b \\
        --revision step143000 \\
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

STIMULI_PATH = REPO_ROOT / "docs" / "stimuli_v5.1.json"
OUTPUT_DIR   = REPO_ROOT / "output" / "production" / "llm" / "activations"


def model_slug(model_id: str, revision: str) -> str:
    """Filesystem-safe identifier: slashes → __, revision appended."""
    return model_id.replace("/", "__") + f"__{revision}"


def load_stimuli(path: Path) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    return [s["text"] for s in data["stimuli"]]


def extract(model_id: str, revision: str, device: str, dry_run: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    slug = model_slug(model_id, revision)
    out_path = OUTPUT_DIR / f"{slug}.npz"
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return

    stimuli = load_stimuli(STIMULI_PATH)
    print(f"Stimuli: {len(stimuli)}")
    print(f"Model:   {model_id}  revision={revision}")
    print(f"Device:  {device}")

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
        d_model  = model.config.hidden_size

        # hidden_states[0] = embedding, hidden_states[1..n_layers] = transformer blocks
        layer_labels = ["emb"] + [f"L{i}" for i in range(1, n_layers + 1)]
        print(f"Layers: {n_layers} transformer blocks + embedding = {len(layer_labels)} total")
        print(f"Download+load: {time.time()-t0:.1f}s")

        n_stimuli = len(stimuli)
        last_acts = {lbl: np.zeros((n_stimuli, d_model), dtype=np.float32) for lbl in layer_labels}
        mean_acts = {lbl: np.zeros((n_stimuli, d_model), dtype=np.float32) for lbl in layer_labels}
        token_counts = np.zeros(n_stimuli, dtype=np.int32)
        nll          = np.zeros(n_stimuli, dtype=np.float32)

        t1 = time.time()
        for i, text in enumerate(stimuli):
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True,
                                labels=inputs["input_ids"])

            token_counts[i] = inputs["input_ids"].shape[1]
            nll[i] = outputs.loss.item()

            # hidden_states: tuple of (1, seq_len, d_model) tensors
            for j, lbl in enumerate(layer_labels):
                hs = outputs.hidden_states[j][0]  # (seq_len, d_model)
                last_acts[lbl][i] = hs[-1, :].float().cpu().numpy()
                mean_acts[lbl][i] = hs.mean(0).float().cpu().numpy()

            if (i + 1) % 32 == 0 or i == n_stimuli - 1:
                print(f"  {i+1}/{n_stimuli}  ({time.time()-t1:.1f}s)", flush=True)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"Extraction: {time.time()-t1:.1f}s")

    # tmp dir (and downloaded weights) are now deleted

    np.savez_compressed(
        str(out_path),
        **{f"last_{lbl}": last_acts[lbl] for lbl in layer_labels},
        **{f"mean_{lbl}": mean_acts[lbl] for lbl in layer_labels},
        token_counts = token_counts,
        nll          = nll,
        layer_labels = np.array(layer_labels),
        n_layers     = np.int32(n_layers),
        d_model      = np.int32(d_model),
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved {out_path.name}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", default="main",
                        help="HF revision: 'main' or e.g. 'step143000' for Pythia checkpoints")
    parser.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    extract(args.model_id, args.revision, args.device, args.dry_run)


if __name__ == "__main__":
    main()
