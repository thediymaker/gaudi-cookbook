#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Device gates for the MiMo-V2.6-Flash port, against a server started with the recipe's command. One PASS or FAIL
# line per gate; exit code 1 if any gate failed.
#
#   URL=http://localhost:8000 bash run_gates.sh
#
# Gates: decode speed at 1 / 4 / 16 streams, tool calls (short, and after a 32K-token document), window-sized KV
# against the stock KV layout, needles at 128K / 256K / 500K, and four 500K-token prompts at once.
#
# The KV gate compares greedy outputs with a reference recorded on the stock layout: start a second server with
# the same command and MIMO_SWA_KV_GROUPS=0, run
#   bash run_gates.sh --record-kv-reference stock-kv.json
# and pass KV_REFERENCE=stock-kv.json to the full run. Without it the gate prints SKIP.
# The whole run takes about an hour and a half on one 8-card node, most of it the 500K prefills.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
G="python3 ${HERE}/gates.py"
OUT="${GATES_OUT:-/tmp/mimo-gates}"
mkdir -p "$OUT"

if [[ "${1:-}" == "--record-kv-reference" ]]; then
  exec $G kv-record "${2:?output file}"
fi

fail=0
run() { "$@" || fail=1; }

run $G decode --streams 1,4,16
run $G tools
if [[ -n "${KV_REFERENCE:-}" ]]; then
  run $G kv-record "$OUT/kv-groups.json"
  run $G kv-diff "$KV_REFERENCE" "$OUT/kv-groups.json"
else
  echo "SKIP kv-identical: set KV_REFERENCE (see the header of this script)"
fi
run $G needles --tokens 131072
run $G needles --tokens 262144
run $G needles --tokens 500000
run $G concurrent --tokens 500000 --seeds 21,22,23,24
exit $fail
