#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Device gates for the MiMo-V2.6 port, against a server started with a recipe's command. One PASS or FAIL line per
# gate; exit code 1 if any gate failed.
#
#   URL=http://localhost:8000 bash run_gates.sh                 # MiMo-V2.6-Flash-RL recipe (512K)
#   URL=http://localhost:8000 PROFILE=pro bash run_gates.sh     # MiMo-V2.6-Pro-RL recipe (256K)
#
# Gates, Flash: decode speed at 1 / 4 / 16 streams (floor 20 tok/s per stream), tool calls (short, and after a
# 32K-token document), window-sized KV against the stock KV layout, needles at 128K / 256K / 500K, and four
# 500K-token prompts at once.
# Gates, Pro: decode speed at 1 / 4 / 8 streams (floor 12 tok/s per stream: the stock MoE op stays under it at 4 and
# 8 streams, so a server with the kernels off fails it), tool calls, window-sized KV against the stock layout,
# needles at 128K and 240K, and four 240K-token prompts at once. Run tests/hpu/mxk_ops.py on one card first.
#
# The KV gate compares greedy outputs with a reference recorded on the stock layout: start a second server with
# the same command and MIMO_SWA_KV_GROUPS=0, run
#   bash run_gates.sh --record-kv-reference stock-kv.json
# and pass KV_REFERENCE=stock-kv.json to the full run. Without it the gate prints SKIP.
# The Flash run takes about an hour and a half on one 8-card node, most of it the 500K prefills; the Pro run about
# an hour, most of it the 240K prefills.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
G="python3 ${HERE}/gates.py"
OUT="${GATES_OUT:-/tmp/mimo-gates}"
PROFILE="${PROFILE:-flash}"
mkdir -p "$OUT"

if [[ "${1:-}" == "--record-kv-reference" ]]; then
  exec $G kv-record "${2:?output file}"
fi

case "$PROFILE" in
  flash) STREAMS=1,4,16 FLOOR=20 NEEDLES=(131072 262144 500000) CONCURRENT=500000 ;;
  pro) STREAMS=1,4,8 FLOOR=12 NEEDLES=(131072 245760) CONCURRENT=245760 ;;
  *) echo "PROFILE must be flash or pro, got '$PROFILE'" >&2; exit 2 ;;
esac

fail=0
run() { "$@" || fail=1; }

run $G decode --streams "$STREAMS" --min-tok-s "$FLOOR"
run $G tools
if [[ -n "${KV_REFERENCE:-}" ]]; then
  run $G kv-record "$OUT/kv-groups.json"
  run $G kv-diff "$KV_REFERENCE" "$OUT/kv-groups.json"
else
  echo "SKIP kv-identical: set KV_REFERENCE (see the header of this script)"
fi
for n in "${NEEDLES[@]}"; do
  run $G needles --tokens "$n"
done
run $G concurrent --tokens "$CONCURRENT" --seeds 21,22,23,24
exit $fail
