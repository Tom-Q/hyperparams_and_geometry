#!/bin/bash
# Sequential LLM activation extraction.
# Skips files that already exist (handled inside extract_llm_activations.py).
# Run from repo root with the venv active:
#   source .venv/bin/activate && bash scripts/run_llm_extraction.sh
set -e

PY=python

# ── Final-checkpoint cross-model analysis (--layers all: emb + L50 + L75 + Lout) ──

# Pythia series — fit on GPU up to 1.4b
$PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-70m   --revision step143000 --layers all
$PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-160m  --revision step143000 --layers all
$PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-410m  --revision step143000 --layers all
$PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-1b    --revision step143000 --layers all
$PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-1.4b  --revision step143000 --layers all
$PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-2.8b  --revision step143000 --layers all --device cpu

# Cross-family models
$PY scripts/extract_llm_activations.py --model-id meta-llama/Llama-3.2-1B       --revision main --layers all
$PY scripts/extract_llm_activations.py --model-id Qwen/Qwen2.5-1.5B             --revision main --layers all --device cpu
$PY scripts/extract_llm_activations.py --model-id HuggingFaceTB/SmolLM2-1.7B    --revision main --layers all --device cpu
$PY scripts/extract_llm_activations.py --model-id google/gemma-2-2b              --revision main --layers all --device cpu

# ── Pythia training dynamics (--layers lout only; step143000 covered above) ──────

CHECKPOINTS="step1 step2 step4 step8 step16 step32 step64 step128 step256 step512 step1000 step2000 step8000 step32000 step64000"

for CKPT in $CHECKPOINTS; do
    $PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-70m  --revision $CKPT --layers lout
    $PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-160m --revision $CKPT --layers lout
    $PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-410m --revision $CKPT --layers lout
    $PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-1b   --revision $CKPT --layers lout
    $PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-1.4b --revision $CKPT --layers lout
    $PY scripts/extract_llm_activations.py --model-id EleutherAI/pythia-2.8b --revision $CKPT --layers lout --device cpu
done

echo "All extractions complete."
