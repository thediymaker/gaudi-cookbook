#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Device gates for the Beam-1.0 port. Start the server with the recipe's command first (the Launch block
# of docs/recipes/beam-1-0-501b-a23b-fp8.md), then run this in the same container or anywhere that
# reaches it:
#
#   BEAM_URL=http://127.0.0.1:8000 ports/beam-1-0/tests/hpu/run_gates.sh [gate ...]
#
# Prints one PASS or FAIL line per gate (see gates.py for what each checks) and exits non-zero if any
# gate fails. All gates take about 30 minutes on 8 Gaudi 2 cards, most of it in the 250K-token prompts.
# Details go to beam_gates.json in the current directory (BEAM_GATES_OUT to change).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL="${BEAM_URL:-http://127.0.0.1:8000}"

for _ in $(seq 1 60); do
  curl -fsS "${URL}/health" > /dev/null 2>&1 && break
  sleep 10
done
curl -fsS "${URL}/health" > /dev/null || { echo "FAIL server: ${URL}/health does not answer"; exit 1; }

BEAM_URL="${URL}" exec python3 "${HERE}/gates.py" --out "${BEAM_GATES_OUT:-beam_gates.json}" "${@:-all}"
