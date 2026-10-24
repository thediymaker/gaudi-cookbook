#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Chatterbox device gates on one card, for both models. Inside the server image:
#   tests/hpu/run_gates.sh [--url http://host:8000] [--whisper openai/whisper-large-v3]
# Prints one PASS or FAIL line per gate; exit 0 only if all pass.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../../../common/gate_lib.sh"
start_server "$HERE/../.." 2400 python3 server.py
status=0
echo "== chatterbox-turbo"
python3 "$COMMON/tts_gates.py" --url "$URL" --model chatterbox-turbo --streams 4 --min-speed 5.0 \
  --max-first-audio 0.3 "${GATE_ARGS[@]}" || status=1
echo "== chatterbox"
python3 "$COMMON/tts_gates.py" --url "$URL" --model chatterbox --streams 4 --min-speed 3.0 \
  --max-first-audio 0.5 "${GATE_ARGS[@]}" || status=1
exit $status
