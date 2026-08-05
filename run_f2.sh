#!/usr/bin/env bash
# Run Finding #2 (scripts 20-25) in order.
# Requires Finding #1 to have been run first (reads rdm_noise_ceiling.csv,
# rdm_category_structure.csv, rdm_dimensionality.csv from Finding #1 outputs).
# Stops immediately if any script exits with a non-zero code.

set -euo pipefail

ANALYSIS="$(cd "$(dirname "$0")/analysis" && pwd)"
PYTHON="${PYTHON:-python3}"
TOTAL_START=$(date +%s)

run_step() {
    local script="$1"
    shift
    local label
    label="$(basename "$script")"
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  ▶  $label  $*"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════"
    local t0
    t0=$(date +%s)
    "$PYTHON" "$script" "$@"
    local code=$?
    local elapsed=$(( $(date +%s) - t0 ))
    if [ $code -ne 0 ]; then
        echo ""
        echo "✗  FAILED: $label (exit $code, ${elapsed}s)"
        exit $code
    fi
    echo "  ✓  done in ${elapsed}s"
}

echo "Starting Finding #2 pipeline — $(date '+%Y-%m-%d %H:%M:%S')"

# ── Finding #2: HP effects on representations ─────────────────────────────────
# 20 upserts rdm_per_network_stats.csv for cifar10 only; 21-25 regenerate
# their figures from the full merged CSV (no expensive HDF5 re-loading)
run_step "$ANALYSIS/20_hp_effects.py" --tasks cifar10
run_step "$ANALYSIS/21_latent_vars.py"
run_step "$ANALYSIS/22_rdm_pca.py"
run_step "$ANALYSIS/23_cca.py"
run_step "$ANALYSIS/24_layer_comparison.py"
run_step "$ANALYSIS/25_umap.py"

TOTAL=$(( $(date +%s) - TOTAL_START ))
echo ""
echo "════════════════════════════════════════════════════════"
echo "  All steps completed successfully in ${TOTAL}s"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════"
