#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Install the Ripple-65B port into a stock vllm-gaudi releases/v0.30.0 image. The only build step: the Dockerfile
# runs it, and so can a running container started from the stock image. Stops at the first step that does not
# apply.
#
#   bash ports/ripple-65b/install.sh            # from the repository root, or any directory
#
# Steps:
#   1. check the installed vllm-gaudi and vLLM are the commits this port was written against
#   2. apply the cookbook's v0.30.0 fix profile plus swa-window-context (the runner patches below build on both)
#   3. patch vLLM's DelegatingParser as vLLM PR #65745 does (reasoning that opens after content)
#   4. patch the HPU runner: exact sliding windows for model type ricursive, hooks for window-sized KV groups
#   5. install the ripple_hpu plugin (model, config, parsers, runner hooks)
#   6. build gates (tests/build_gates.py): patches applied, plugin registered, chat template, parsers, window masks
#
# The recipe's launch step runs fixes/apply_all.py again with the same fixes; they then report "applied".
set -euo pipefail
PORT=$(cd "$(dirname "$0")" && pwd)
COOKBOOK=$(cd "$PORT/../.." && pwd)
VLLM_GAUDI_COMMIT=058f107
VLLM_COMMIT=ced6857a
step() { echo "[ripple-65b install] $*"; }

step "1/6 base image: vllm-gaudi $VLLM_GAUDI_COMMIT, vLLM $VLLM_COMMIT"
python3 - "$VLLM_GAUDI_COMMIT" "$VLLM_COMMIT" <<'EOF'
import importlib.metadata
import importlib.util
import os
import subprocess
import sys


def commit(pkg, dist):
    # a source install (as in docs/getting-started.md) has a git checkout; a wheel carries +g<hash> in its version
    spec = importlib.util.find_spec(pkg)
    src = os.path.dirname(list(spec.submodule_search_locations)[0]) if spec else None
    if src:
        r = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.strip()
    ver = importlib.metadata.version(dist)
    return ver.split("+g", 1)[1].split(".")[0] if "+g" in ver else ver


bad = []
for pkg, dist, want in (("vllm_gaudi", "vllm_gaudi", sys.argv[1]), ("vllm", "vllm", sys.argv[2])):
    got = commit(pkg, dist)
    if not got.startswith(want):
        bad.append(f"{pkg} is at {got}, the port needs {want}")
if bad:
    sys.exit("[ripple-65b install] " + "; ".join(bad))
EOF

step "2/6 fix profile v0.30.0 + swa-window-context"
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.30.0 swa-window-context

step "3/6 vLLM parser: PR #65745 DelegatingParser change"
python3 "$PORT/patches/parser_streams_all.py"

step "4/6 runner patches"
python3 "$PORT/patches/runner_window.py"
python3 "$PORT/patches/swa_kv_hooks.py"

step "5/6 plugin"
# build in a copy: setuptools writes build/ and egg-info next to the sources
BUILD=$(mktemp -d)
cp -r "$PORT/ripple_hpu" "$BUILD/"
find "$BUILD" -name __pycache__ -prune -exec rm -rf {} +
PIP_NO_INDEX=1 pip install --no-deps --no-build-isolation --no-cache-dir "$BUILD/ripple_hpu"
rm -rf "$BUILD"

step "6/6 build gates"
python3 "$PORT/tests/build_gates.py"
step "done"
