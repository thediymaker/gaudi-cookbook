#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Device gates for a DeepSeek-V4.1-Flash server started with the recipe's command. Run inside the container (or
# anywhere that reaches the server), after the warm-up has finished, on an otherwise idle server:
#
#   ports/deepseek-v4-1-flash/tests/hpu/run_gates.sh [--log SERVER_LOG] [--kernel on|off] [--out DIR]
#
# Prints one PASS or FAIL line per gate and exits non-zero if any gate fails:
#   SPEED       decode tok/s at 1 and 4 streams (reported; first, while the server is idle)
#   PPL         perplexity vs the fp32 CPU oracle (data/oracle_1017.json, for the -1017 weights)
#   DECODE-TF   decode path vs prefill path, teacher-forced
#   NEEDLES     a secret at half depth in ~13K- and ~45K-token prompts
#   ISOLATION   concurrent secrets with a shared cached prefix, repeats and aborts: no cross-request leak
#   STRUCTURED  json_schema (thinking off and on), tool_choice required and auto, long plain prompts alongside
# With --log (the server's output):
#   KERNEL      the server runs the MXFP4 decode kernels and the Sinkhorn kernel when --kernel on, neither when off
#   TAGS        with DSV41_TAG_CHECK=count, the last "tag check" line reports 0 violating rows
# --out DIR keeps each gate's output (DIR/<gate>.log); kernel_ab.py compares two such directories (kernel off, on).
#
# Env: DSV41_GATE_URL (default http://localhost:8000), PYTHON (default python3).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3}
LOG="" KERNEL="" OUT=""
while [ $# -gt 0 ]; do
  case $1 in
    --log) LOG=$2; shift 2 ;;
    --kernel) KERNEL=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
case $KERNEL in ""|on|off) ;; *) echo "--kernel takes on or off" >&2; exit 2 ;; esac
OUT=${OUT:-$(mktemp -d)}
mkdir -p "$OUT"
FAILED=()

gate() {  # gate <name> <script> [args...]
  local name=$1; shift
  echo "== $name"
  "$PY" "$HERE/$@" 2>&1 | tee "$OUT/$name.log"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || FAILED+=("$name")
}

gate speed speed_gate.py --conc 1,4 --tokens 300
gate ppl ppl_gate.py
gate decode-tf decode_tf_gate.py --max-tokens 200 --concurrent 4
gate needles needle_gate.py --words 12000,40500 --depth 0.5
gate isolation isolation_gate.py --n 24 --workers 8 --min-words 600 --max-words 3000 --shared-prefix-words 4000 \
  --repeat --abort-frac 0.2 --seed 21
gate structured structured_gate.py --rounds 2 --per-round 6 --conc 12

if [ -n "$LOG" ]; then
  if [ -n "$KERNEL" ]; then
    echo "== kernel"
    on_moe=$(grep -c "MXFP4 decode kernels ON" "$LOG")
    off_moe=$(grep -c "MXFP4 decode kernels OFF" "$LOG")
    on_hc=$(grep -c "mHC Sinkhorn TPC kernel" "$LOG")
    if [ "$KERNEL" = on ]; then
      ok=$(( on_moe > 0 && off_moe == 0 && on_hc > 0 ))
    else
      ok=$(( on_moe == 0 && on_hc == 0 ))
    fi
    detail="expected $KERNEL: MoE kernel ON lines $on_moe, OFF lines $off_moe, Sinkhorn kernel lines $on_hc"
    if [ "$ok" -eq 1 ]; then res="KERNEL PASS $detail"; else res="KERNEL FAIL $detail"; FAILED+=(kernel); fi
    echo "$res" | tee "$OUT/kernel.log"
  fi
  last=$(grep "tag check after" "$LOG" | tail -1)
  if [ -n "$last" ]; then
    echo "== tags"
    rows=$(sed -E 's/.*: ([0-9]+) violating rows.*/\1/' <<<"$last")
    if [ "$rows" = 0 ]; then echo "TAGS PASS ${last##*dsv41: }"; else echo "TAGS FAIL ${last##*dsv41: }"; FAILED+=(tags); fi
  fi
fi

echo "== results in $OUT"
if [ ${#FAILED[@]} -eq 0 ]; then echo "ALL GATES PASS"; exit 0; fi
echo "FAILED: ${FAILED[*]}"
exit 1
