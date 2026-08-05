#!/usr/bin/env bash
# Run Finding #3 (scripts 31-37) in order.
# Reads directly from HDF5 trajectory data and success_thresholds.json —
# does not depend on Finding #1 or #2 outputs.
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

echo "Starting Finding #3 pipeline — $(date '+%Y-%m-%d %H:%M:%S')"

# ── Finding #3: Representations over the course of learning ──────────────────
# 31-35 upsert existing CSVs for cifar10 only; 36-37 produce per-task files
# so are naturally incremental (just run for all tasks, only cifar10 is new)
run_step "$ANALYSIS/31_spearman_stability.py" --tasks cifar10
run_step "$ANALYSIS/32_crystallisation_time.py" --tasks cifar10
run_step "$ANALYSIS/33_critical_period.py" --tasks cifar10
run_step "$ANALYSIS/34_change_vs_performance.py" --tasks cifar10
run_step "$ANALYSIS/35_overfitting.py" --tasks cifar10
run_step "$ANALYSIS/36_trajectory_mds.py"
run_step "$ANALYSIS/37_rdm_gallery_thru_learning.py"

TOTAL=$(( $(date +%s) - TOTAL_START ))
echo ""
echo "════════════════════════════════════════════════════════"
echo "  All steps completed successfully in ${TOTAL}s"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════"
