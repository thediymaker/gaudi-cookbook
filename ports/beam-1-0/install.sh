#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Install the Beam-1.0 port into a stock vllm-gaudi releases/v0.30.0 image (Gaudi software 1.24.1).
# The only build step; the Dockerfile runs it from /opt/gaudi-cookbook. Stops at the first step that
# does not apply, so an image is never left half-installed.
#
#   1. check the image: vllm-gaudi 058f107 and vLLM ced6857a, nothing else;
#   2. apply the cookbook's v0.30.0 fix profile: the port's runner patches are written against the fixed
#      runner (they extend swa-decode-window and so-row-owners). The recipe's launch step runs the same
#      command again and finds every fix applied;
#   3. install the vendored files of vLLM PR #62052 (the response-template parser framework and grammar);
#   4. patch the runner: exact sliding windows for model type "beam", grammar bitmask on the device;
#   5. install the beam_hpu plugin (model, load format `beam`, parsers and frontend);
#   6. run the build gates (no device needed).
set -euo pipefail

PORT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COOKBOOK="$(cd "${PORT}/../.." && pwd)"
VLLM_GAUDI_COMMIT=058f107
VLLM_COMMIT=ced6857a

echo "== beam-1-0: checking the image"
python3 - "$VLLM_GAUDI_COMMIT" "$VLLM_COMMIT" <<'EOF'
import importlib.metadata as md
import sys

want = {"vllm_gaudi": sys.argv[1], "vllm": sys.argv[2]}
for pkg, commit in want.items():
    try:
        version = md.version(pkg)
    except md.PackageNotFoundError:
        sys.exit(f"beam-1-0: {pkg} is not installed")
    # setuptools-scm versions carry the commit: 0.28.1rc1.dev944+gced6857af...
    if f"+g{commit}" not in version and f".g{commit}" not in version:
        sys.exit(f"beam-1-0: {pkg} {version} is not commit {commit}; build the stock releases/v0.30.0 image first")
    print(f"beam-1-0: {pkg} {version}")
EOF

echo "== beam-1-0: cookbook fixes, v0.30.0 profile"
python3 "${COOKBOOK}/fixes/apply_all.py" --profile v0.30.0

echo "== beam-1-0: vendored vLLM PR #62052 files"
python3 "${PORT}/patches/vllm_pr_62052.py"

echo "== beam-1-0: runner patches"
python3 "${PORT}/patches/beam_sliding_window.py"
python3 "${PORT}/patches/grammar_device_mask.py"

echo "== beam-1-0: beam_hpu plugin"
find "${PORT}/beam_hpu" -name __pycache__ -prune -exec rm -rf {} +
PIP_NO_INDEX=1 python3 -m pip install --no-deps --no-build-isolation "${PORT}/beam_hpu"

echo "== beam-1-0: build gates"
python3 "${PORT}/tests/build_gates.py"
echo "== beam-1-0: installed"
