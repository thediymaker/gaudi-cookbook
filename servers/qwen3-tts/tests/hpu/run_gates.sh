#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Qwen3-TTS device gates on one card. Inside the server image:
#   tests/hpu/run_gates.sh [--url http://host:8000] [--whisper openai/whisper-large-v3]
# Prints one PASS or FAIL line per gate; exit 0 only if all pass.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../../../common/gate_lib.sh"
start_server "$HERE/../.." 1200 python3 server.py
python3 "$COMMON/tts_gates.py" --url "$URL" --model qwen3-tts --voice ryan --streams 16 \
  --min-speed 4.0 --max-first-audio 0.3 "${GATE_ARGS[@]}"
