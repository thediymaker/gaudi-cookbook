#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# GLM-5.3-Flash device gates against a running server started with the recipe's command (8 Gaudi 2 cards).
# Prints one PASS or FAIL line per gate and exits non-zero if any gate fails.
#
#   ports/glm-5-3-flash/tests/hpu/run_gates.sh [--base URL] [--model NAME] [GATE ...]
#
# Gates (default: all, in this order): golden needles tools images decode prefill. See gates.py for each bar.
# The whole set took about 8 minutes on a fresh server (each gate warms its shapes before measuring).
# The kernel gate runs without a server, on one free card: run_gates.sh sinkhorn (sinkhorn_parity.py).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ARGS=()
GATES=()
while [ $# -gt 0 ]; do
  case $1 in
    --base|--model) ARGS+=("$1" "$2"); shift 2 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) GATES+=("$1"); shift ;;
  esac
done
[ ${#GATES[@]} -gt 0 ] || GATES=(golden needles tools images decode prefill)
if [ "${GATES[*]}" = sinkhorn ]; then
  exec python3 "$HERE/sinkhorn_parity.py"
fi
python3 "$HERE/gates.py" "${ARGS[@]}" "${GATES[@]}"
