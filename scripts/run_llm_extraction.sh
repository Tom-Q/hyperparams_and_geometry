#!/bin/bash
# Sequential LLM activation extraction — all layers saved at every run.
# Skips files that already exist (handled inside extract_llm_activations.py).
# Run from repo root with the venv active:
#   source .venv/bin/activate && bash scripts/run_llm_extraction.sh
set -e

PY=python

# ── Final-checkpoint cross-model analysis ─────────────────────────────────────

$PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-70m   --revision step143000
$PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-160m  --revision step143000
$PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-410m  --revision step143000
$PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-1b    --revision step143000
$PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-1.4b  --revision step143000
$PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-2.8b  --revision step143000 --device cpu

# Base variants (passage stimuli)
$PY scripts/extract_llm_activations.py --model-id meta-llama/Llama-3.2-1B
$PY scripts/extract_llm_activations.py --model-id Qwen/Qwen2.5-1.5B                  --device cpu
$PY scripts/extract_llm_activations.py --model-id HuggingFaceTB/SmolLM2-1.7B         --device cpu
$PY scripts/extract_llm_activations.py --model-id google/gemma-2-2b                  --device cpu

# Instruct variants (request stimuli)
$PY scripts/extract_llm_activations.py --model-id meta-llama/Llama-3.2-1B-Instruct
$PY scripts/extract_llm_activations.py --model-id Qwen/Qwen2.5-1.5B-Instruct         --device cpu
$PY scripts/extract_llm_activations.py --model-id HuggingFaceTB/SmolLM2-1.7B-Instruct --device cpu
$PY scripts/extract_llm_activations.py --model-id google/gemma-2-2b-it                --device cpu

# OLMo base + instruct final checkpoints
$PY scripts/extract_llm_activations.py --model-id allenai/OLMo-1B-0724-hf     --revision step1454000-tokens3048B
$PY scripts/extract_llm_activations.py --model-id allenai/OLMo-2-0425-1B-SFT
$PY scripts/extract_llm_activations.py --model-id allenai/OLMo-2-0425-1B-Instruct

# ── Pythia training dynamics (15 checkpoints × 6 sizes) ───────────────────────
# step143000 is already covered above.

CHECKPOINTS="step1 step2 step4 step8 step16 step32 step64 step128 step256 step512 step1000 step2000 step8000 step32000 step64000"

for CKPT in $CHECKPOINTS; do
    $PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-70m  --revision $CKPT
    $PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-160m --revision $CKPT
    $PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-410m --revision $CKPT
    $PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-1b   --revision $CKPT
    $PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-1.4b --revision $CKPT
    $PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-2.8b --revision $CKPT --device cpu
done

# ── OLMo training dynamics (15 checkpoints, 1B base only) ─────────────────────
# step1454000 already covered above.

OLMO_CHECKPOINTS="step1000-tokens2B step2000-tokens4B step3000-tokens6B step4500-tokens9B step7000-tokens14B step11000-tokens23B step18000-tokens37B step30000-tokens62B step49000-tokens102B step79000-tokens165B step128000-tokens268B step209000-tokens438B step339000-tokens710B step551000-tokens1155B step895000-tokens1876B"

for CKPT in $OLMO_CHECKPOINTS; do
    $PY scripts/extract_llm_activations.py --model-id allenai/OLMo-1B-0724-hf --revision $CKPT
done

echo "All extractions complete."
