#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Serving gates for the GLM-5.2 and GLM-5.3 MXFP4 port. Start the server with the recipe's command first (the Launch
# block of docs/recipes/glm-5-3-mxfp4.md or docs/recipes/glm-5-2-mxfp4.md), then run this in the same container or
# anywhere that reaches it:
#
#   BASE=http://127.0.0.1:8000 ports/glm-5-mxfp4/tests/hpu/run_gates.sh [OUT_DIR] [REF_DIR]
#
# Gates (one PASS or FAIL line each; exit 1 if any fails; details in glm5_gates.py):
#   needles   3 needles at 10 / 50 / 90% depth, two fillers, 16K to 120K tokens (the served window is 131,072)
#   gsm8k     first 50 GSM8K test questions, reasoning low, greedy: >= 47 / 50
#   tools     tool_choice required / named / auto / none and a JSON schema, streaming and not (glm47 parser)
#   mtp       MTP acceptance on natural chat prompts at the server's default sampling: >= 0.70
#   logprobs  40 sampled requests with top_logprobs=5 under MTP: every token has finite logprobs
#   speed     decode tok/s at 1 and 4 streams (~2K-token prompt, greedy, 256 tokens): recorded
#   ttft      time to first token of an uncached ~2K and ~8K-token prompt, median of 3: recorded
#   stress    4 decoding streams while 64K- and 100K-token prompts prefill: everything finishes and is right
#   greedy    16 fixed prompts, 192 greedy tokens each: recorded in OUT_DIR/greedy.json
# With REF_DIR (the OUT_DIR of an earlier run on the same server build with switches that are documented to keep
# greedy outputs bit-identical, for example GLM_ROUTER_FAST, GLM_EARLY_DRAFT and GLM_SPEC_DEVICE_REJECT off), the
# compare gate checks that both greedy files are token-identical.
# About an hour on 8 Gaudi 2 cards, most of it in the needles and GSM8K.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BASE="${BASE:-http://127.0.0.1:8000}"
OUT="${1:-glm5_gates}"
REF="${2:-}"
mkdir -p "$OUT"
G="python3 $HERE/glm5_gates.py"

for _ in $(seq 1 120); do
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
gate $G greedy "$OUT/greedy.json"
gate $G tools
gate $G mtp
gate $G logprobs 40
gate $G speed
gate $G ttft
gate $G gsm8k
gate $G needles 16384,32768,65536,98304,122880
gate $G stress
if [ -n "$REF" ]; then
  gate $G compare "$REF/greedy.json" "$OUT/greedy.json"
fi
exit $rc
