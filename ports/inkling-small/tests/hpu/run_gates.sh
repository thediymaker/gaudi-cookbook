#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Device gates for the Inkling-Small port. Run inside the port image, next to a server started with the
# recipe's command. Prints one PASS or FAIL line per gate and exits non-zero if any gate fails.
#
#   INKLING_MODEL_DIR=<checkpoint dir> [SERVER_LOG=<server log>] [BASE=http://localhost:8000] \
#     [MODEL=thinkingmachines/Inkling-Small] bash ports/inkling-small/tests/hpu/run_gates.sh [quick|long|all]
#
# quick (default): install and CPU checks, the serving gates in inkling_gates.py, then the engine log.
# long: the long-context tier only (200K / 250K needles, 3 x 100K resident, prefill share); about 20 min.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="$(cd "$HERE/../.." && pwd)"
TIER="${1:-quick}"
FAIL=0
line() { echo "$1 $2${3:+ | $3}"; [ "$1" = PASS ] || FAIL=1; }

if [ "$TIER" != long ]; then
    # Plugin, config, KV layout, runner patch, parsers, rel-bias and conv-state checks (CPU, needs pytest:
    # pip install pytest). test_inkling_patch.py is for source copies, not the installed image.
    out=$(cd /tmp && INKLING_PORT_IMAGE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest -q -p no:cacheprovider \
          "$PORT"/tests/test_inkling_{stack,parsers,ops,conv_state}.py 2>&1 | tail -1)
    if echo "$out" | grep -q " passed" && ! echo "$out" | grep -qE "failed|error|skipped"; then
        line PASS preflight "$out"
    else
        line FAIL preflight "$out (set INKLING_MODEL_DIR so nothing skips)"
    fi
    python3 "$HERE/inkling_gates.py" || FAIL=1
    # After the serving gates: the decode window is logged on the first decode step.
    if [ -n "${SERVER_LOG:-}" ]; then
        grep -q "KV groups: \['FullAttentionSpecx42'\]" "$SERVER_LOG" \
            && line PASS log/kv-groups "one FullAttention group" || line FAIL log/kv-groups "not in $SERVER_LOG"
        pool=$(grep -o "KV pool: [0-9]* blocks.*max_model_len [0-9]*" "$SERVER_LOG" | tail -1)
        [ -n "$pool" ] && line PASS log/kv-pool "$pool" || line FAIL log/kv-pool "not in $SERVER_LOG"
        win=$(grep -o "decode sliding_block_size=[0-9]*" "$SERVER_LOG" | tail -1)
        [ "$win" = "decode sliding_block_size=5" ] && line PASS log/decode-window "$win" \
            || line FAIL log/decode-window "${win:-not logged} (want 5 with INKLING_DECODE_WINDOW_PLUS1=1)"
    fi
fi
if [ "$TIER" = long ] || [ "$TIER" = all ]; then
    python3 "$HERE/inkling_gates.py" long || FAIL=1
fi
exit $FAIL
