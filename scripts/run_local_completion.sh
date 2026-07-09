#!/bin/bash
# Run mnist_dual and adding simultaneously to completion.
# One thread each to avoid CPU contention.
# Logs go to each task's output directory.

set -e
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

VENV=.venv/bin/python

echo "Launching mnist_dual (360 iterations remaining)..."
$VENV run_bo.py \
    --task        mnist_dual \
    --n-iter      1000 \
    --n-sobol     200 \
    --output-dir  output/production \
    --beta        4.0 \
    --h           0.162 \
    > output/production/mnist_dual/run_local.log 2>&1 &
PID_DUAL=$!

echo "Launching adding (948 iterations remaining)..."
$VENV run_bo.py \
    --task        adding \
    --n-iter      1000 \
    --n-sobol     200 \
    --output-dir  output/production \
    --beta        4.0 \
    --h           0.147 \
    > output/production/adding/run_local.log 2>&1 &
PID_ADD=$!

echo "mnist_dual PID: $PID_DUAL"
echo "adding     PID: $PID_ADD"
echo ""
echo "Monitor with:"
echo "  tail -f output/production/mnist_dual/run_local.log"
echo "  tail -f output/production/adding/run_local.log"

wait $PID_DUAL && echo "mnist_dual done." || echo "mnist_dual FAILED (exit $?)."
wait $PID_ADD  && echo "adding done."     || echo "adding FAILED (exit $?)."
