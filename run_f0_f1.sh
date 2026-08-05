#!/usr/bin/env bash
# Run Finding #0 (scripts 01-08) and Finding #1 (scripts 11-18) in order.
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

echo "Starting Finding #0 + #1 pipeline — $(date '+%Y-%m-%d %H:%M:%S')"

# ── Finding #0: HP sampling coverage ─────────────────────────────────────────
run_step "$ANALYSIS/01_disk_inventory.py"
run_step "$ANALYSIS/02_performance_lorenz.py"
run_step "$ANALYSIS/03_summary_table.py"
run_step "$ANALYSIS/04_marginal_coverage.py"
run_step "$ANALYSIS/05_joint_coverage.py"
run_step "$ANALYSIS/06_concentration_lorenz.py"
run_step "$ANALYSIS/07_heatmaps.py"
run_step "$ANALYSIS/08_hp_umap.py"

# ── Finding #1: RSA validity ──────────────────────────────────────────────────
run_step "$ANALYSIS/11_rsa_validity.py"
run_step "$ANALYSIS/11b_temporal_validity.py"
run_step "$ANALYSIS/12_category_models.py"
run_step "$ANALYSIS/12b_adding_category_models.py"
run_step "$ANALYSIS/12c_mnist_rnn_category_models.py"
run_step "$ANALYSIS/13_rdm_gallery.py"
run_step "$ANALYSIS/14_category_structure.py"
run_step "$ANALYSIS/16_dimensionality.py"
run_step "$ANALYSIS/15_layer_comparison.py"
run_step "$ANALYSIS/17_crosstask_rsa.py"
run_step "$ANALYSIS/18_performance_rdm_relationship.py"

TOTAL=$(( $(date +%s) - TOTAL_START ))
echo ""
echo "════════════════════════════════════════════════════════"
echo "  All steps completed successfully in ${TOTAL}s"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════"
