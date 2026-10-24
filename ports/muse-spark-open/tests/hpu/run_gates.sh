#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Device gates for the Muse Spark Open port. Start the server with the recipe's command first (the Launch block of
# docs/recipes/muse-spark-open.md), then run this in the same container or anywhere that reaches it:
#
#   MUSE_URL=http://127.0.0.1:8000 ports/muse-spark-open/tests/hpu/run_gates.sh [gate ...] [--greedy-ref FILE]
#
# Prints one PASS or FAIL line per gate (gates.py says what each checks) and exits non-zero if any gate fails.
# All gates take about 40 minutes on 8 Gaudi 2 cards, most of it in the three 258K-token needle prompts.
# Details go to muse_gates.json and the greedy ids to muse_greedy.json in the current directory
# (MUSE_GATES_OUT, MUSE_GREEDY_OUT to change).
#
# Kernel switch A/B: run once with MUSE_SPARK_MXFP4_KERNELS=0 in the server's environment, restart the server
# with MUSE_SPARK_MXFP4_KERNELS=1, and run again with --greedy-ref <the first run's muse_greedy.json>.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL="${MUSE_URL:-http://127.0.0.1:8000}"

for _ in $(seq 1 60); do
  curl -fsS "${URL}/health" > /dev/null 2>&1 && break
  sleep 10
done
curl -fsS "${URL}/health" > /dev/null || { echo "FAIL server: ${URL}/health does not answer"; exit 1; }

MUSE_URL="${URL}" exec python3 "${HERE}/gates.py" --out "${MUSE_GATES_OUT:-muse_gates.json}" \
  --greedy-out "${MUSE_GREEDY_OUT:-muse_greedy.json}" "$@"
