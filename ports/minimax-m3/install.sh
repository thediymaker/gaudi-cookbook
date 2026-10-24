#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Install the MiniMax-M3 port into a stock vllm-gaudi v0.26.0 image (Gaudi software 1.24.1).
# The only build step; the Dockerfile runs it from /opt/gaudi-cookbook. Stops at the first step that does not
# apply, so an image is never left half-installed.
#
#   1. check the image: vllm-gaudi b059f9c and vLLM 568afb3a, nothing else (v0.26.0 carries the official HPU
#      MiniMax-M3 model and the minimax_m3_py tool parser);
#   2. patch the compressed-tensors FP8 MoE method: SwiGLU-OAI experts on Habana's fused FP8 kernel;
#   3. patch vLLM's minimax_m3 reasoning parser: prompt markers no longer end reasoning when streaming;
#   4. re-check both patches and import the patched modules.
# The cookbook's v0.26.0 fix profile is applied at container start, as for any recipe (see the recipe).
set -euo pipefail

PORT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_GAUDI_COMMIT=b059f9c
VLLM_COMMIT=568afb3a

echo "== minimax-m3: checking the image"
python3 - "$VLLM_GAUDI_COMMIT" "$VLLM_COMMIT" <<'PY'
import importlib.metadata as md
import importlib.util
import os
import subprocess
import sys

want = {"vllm_gaudi": sys.argv[1], "vllm": sys.argv[2]}
for pkg, commit in want.items():
    try:
        version = md.version(pkg)
    except md.PackageNotFoundError:
        sys.exit(f"minimax-m3: {pkg} is not installed")
    found = f"+g{commit}" in version or f".g{commit}" in version
    if not found:
        # An exact release tag has no commit in its version string; read the source checkout instead.
        spec = importlib.util.find_spec(pkg)
        src = os.path.dirname(os.path.dirname(spec.origin)) if spec and spec.origin else ""
        head = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True).stdout
        found = head.startswith(commit)
    if not found:
        sys.exit(f"minimax-m3: {pkg} {version} is not commit {commit}; build the stock v0.26.0 image first")
    print(f"minimax-m3: {pkg} {version} ({commit})")
PY

echo "== minimax-m3: patches"
python3 "${PORT}/patches/swigluoai_fp8_moe.py"
python3 "${PORT}/patches/reasoning_prompt_scope.py"

echo "== minimax-m3: build checks"
python3 "${PORT}/patches/swigluoai_fp8_moe.py" --check
python3 "${PORT}/patches/reasoning_prompt_scope.py" --check
python3 -c "import vllm_gaudi.ops.hpu_compressed_tensors as m; assert hasattr(m, '_m3_fp8_moe_call')"
python3 -c "import vllm.reasoning.minimax_m3_reasoning_parser as m; \
assert hasattr(m.MiniMaxM3ReasoningParser, '_current_turn_token_ids')"
echo "== minimax-m3: installed"
