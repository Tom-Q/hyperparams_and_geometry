#!/usr/bin/env bash
# RF surrogate smoke test on spirals.
#
# 99 Sobol-phase runs (66 primary + 33 repeats) then 99 RF-phase runs
# (66 primary + 33 repeats) = 198 total training runs.
# Pattern: P, P, R, P, P, R, ... (every other primary is repeated)
# Results go to experiments_rf_test/spirals/ to keep them separate from
# the earlier GP run in experiments/spirals/.
#
# Usage:
#   bash run_rf_test.sh
#   bash run_rf_test.sh --beta 4.0

set -euo pipefail

PYTHON=".venv/bin/python"
TASK="spirals"
N_ITER=198
OUTPUT_DIR="experiments_rf_test"
BETA=8.0

echo "========================================"
echo "RF surrogate test: spirals"
echo "  n_iter:     $N_ITER  (99 sobol + 99 RF, each ~2/3 primary + 1/3 repeat)"
echo "  output_dir: $OUTPUT_DIR/$TASK"
echo "  beta:       $BETA"
echo "========================================"
echo ""

$PYTHON run_bo.py \
    --task        "$TASK"       \
    --n-iter      "$N_ITER"     \
    --output-dir  "$OUTPUT_DIR" \
    --beta        "$BETA"       \
    "${@}"
