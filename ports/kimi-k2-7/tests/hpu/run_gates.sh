#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Serving gates for the Kimi-K2.7-Code port. Start the server with the recipe's command first (the Launch block of
# docs/recipes/kimi-k2-7-code.md), then run this in the same container or anywhere that reaches it:
#
#   BASE=http://127.0.0.1:8000 ports/kimi-k2-7/tests/hpu/run_gates.sh [OUT_DIR] [REF_DIR]
#
# Gates (one PASS or FAIL line each; exit 1 if any fails):
#   golden    12 fixed prompts with known answers, greedy: all correct
#   tools     tool call through the kimi_k2 parser, non-streaming and streaming
#   needles   needle recall at 32K, 64K and 120K tokens (the served window is 131,072), 3 depths each
#   logprobs  200 requests with top_logprobs=20 and CJK / emoji output: every position complete and finite
#   tf        teacher-forced scoring of fixed code text (recorded; compared when REF_DIR is given)
# With REF_DIR (the OUT_DIR of an earlier run, for example with KIMI_MXK=0 on the same server build), golden and
# teacher-forced results are also compared: both sides all correct, top-1 agreement >= 0.95, NLL within 0.02.
# The kernel parity check (kernels/mxfp4-moe/tests/run_parity.sh --variant kimi) needs a free card and runs before
# the server starts; see the port README. About 25 minutes on 8 Gaudi 2 cards, most of it in the needles.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BASE="${BASE:-http://127.0.0.1:8000}"
OUT="${1:-kimi_gates}"
REF="${2:-}"
mkdir -p "$OUT"
G="python3 $HERE/kimi_gates.py"

for _ in $(seq 1 60); do
  curl -fsS "$BASE/health" > /dev/null 2>&1 && break
  sleep 10
done
curl -fsS "$BASE/health" > /dev/null || { echo "FAIL server: $BASE/health does not answer"; exit 1; }

rc=0
gate() {
  "$@" > "$OUT/last.log" 2>&1
  local r=$?
  cat "$OUT/last.log" >> "$OUT/gates.log"
  grep -E '^(PASS|FAIL)' "$OUT/last.log" || echo "FAIL $2: no result (see $OUT/gates.log)"
  [ "$r" -eq 0 ] || rc=1
}
gate $G golden "$OUT/golden.json"
gate $G tools
gate $G needle 32000,64000,120000
gate $G logprobs 200
gate $G tf "$OUT/tf.json"
if [ -n "$REF" ]; then
  gate $G compare "$REF/golden.json" "$OUT/golden.json"
  gate $G compare "$REF/tf.json" "$OUT/tf.json"
fi
exit $rc
