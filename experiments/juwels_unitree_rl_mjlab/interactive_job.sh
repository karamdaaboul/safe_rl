#!/usr/bin/env bash
set -euo pipefail

TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
NUM_CPUS="${NUM_CPUS:-8}"

echo "=== Requesting interactive node on JUWELS Booster ==="
echo "    Time:  ${TIME_LIMIT}  |  CPUs: ${NUM_CPUS}"
echo
echo "After allocation, either run:"
echo "  bash run_in_container.sh"
echo "or run commands manually inside the allocation."
echo

salloc \
    --account=hai_1075 \
    --partition=booster \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${NUM_CPUS}" \
    --gres=gpu:1 \
    --time="${TIME_LIMIT}"
