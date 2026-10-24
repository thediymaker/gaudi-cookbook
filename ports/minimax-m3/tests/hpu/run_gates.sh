#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Device gates for the MiniMax-M3 port. Run inside the port container while the recipe's server is up:
#
#   ports/minimax-m3/tests/hpu/run_gates.sh [BASE_URL] [MODEL]      (defaults: http://127.0.0.1:8000, amd/MiniMax-M3-MXFP4)
#   SERVER_LOG=/path/to/server.log ports/minimax-m3/tests/hpu/run_gates.sh    also checks the expert path in the log
#
# Prints one PASS or FAIL line per gate and exits non-zero if any fails. Takes about 25 minutes on a warm
# server (the 125K needle alone is a few minutes of prefill); GATES_SKIP=needles,speed skips gates.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="$(cd "${HERE}/../.." && pwd)"
COOKBOOK="$(cd "${PORT}/../.." && pwd)"
BASE="${1:-http://127.0.0.1:8000}"
MODEL="${2:-amd/MiniMax-M3-MXFP4}"
fail=0

if python3 "${PORT}/patches/swigluoai_fp8_moe.py" --check >/dev/null &&
    python3 "${PORT}/patches/reasoning_prompt_scope.py" --check >/dev/null &&
    python3 "${COOKBOOK}/fixes/apply_all.py" --check --profile v0.26.0 >/dev/null; then
    echo "PASS install: port patches and the v0.26.0 fix profile are applied"
else
    echo "FAIL install: a port patch or a v0.26.0 fix is not applied"
    fail=1
fi

if [ -n "${SERVER_LOG:-}" ]; then
    if grep -q "\[minimax-m3:fp8-moe\] SwiGLU-OAI FP8 experts: mode=${VLLM_M3_FP8_MOE:-native}" "${SERVER_LOG}"; then
        echo "PASS expert-path: the server log shows mode=${VLLM_M3_FP8_MOE:-native}"
    else
        echo "FAIL expert-path: no '[minimax-m3:fp8-moe] ... mode=${VLLM_M3_FP8_MOE:-native}' line in ${SERVER_LOG}"
        fail=1
    fi
fi

python3 "${HERE}/gates.py" --base "${BASE}" --model "${MODEL}" --skip "${GATES_SKIP:-}" || fail=1
exit "${fail}"
