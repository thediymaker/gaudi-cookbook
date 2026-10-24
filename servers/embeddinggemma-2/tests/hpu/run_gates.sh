#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# EmbeddingGemma 2 device gates on one card. Inside the server image:
#   tests/hpu/run_gates.sh [--url http://host:8000]
# First the in-process parity gates (they need the card to themselves, so they are skipped with --url), then the
# server's API and throughput gates. One PASS or FAIL line per gate; exit 0 only if all pass.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../../../common/gate_lib.sh"
status=0
if [ -z "$URL" ]; then
  (cd "$HERE/../.." && python3 tests/hpu/parity.py) || status=1
fi
start_server "$HERE/../.." 900 python3 server.py
python3 "$HERE/api_gates.py" --url "$URL" "${GATE_ARGS[@]}" || status=1
exit $status
