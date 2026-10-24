#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Orpheus device gates against a running pod (vLLM on the card, the sidecar on port 8000):
#   tests/hpu/run_gates.sh [--url http://host:8000] [--whisper openai/whisper-large-v3]
# Prints one PASS or FAIL line per gate; exit 0 only if all pass.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../../../common/gate_lib.sh"
URL=${URL:-http://127.0.0.1:8000}
python3 "$COMMON/tts_gates.py" --url "$URL" --model orpheus-3b --voice tara --streams 8 --min-speed 1.5 \
  --max-first-audio 0.5 "${GATE_ARGS[@]}"
