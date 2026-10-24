#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Install the Kimi-K2.7-Code port into a stock vllm-gaudi releases/v0.30.0 image. The only build step: the
# Dockerfile runs it, and so can a running container started from the stock image. Stops at the first step
# that does not apply.
#
#   bash ports/kimi-k2-7/install.sh
#
# vLLM at this commit already has the model (KimiK25ForConditionalGeneration with a DeepSeek-V3 text model),
# the kimi_k2 tool and reasoning parsers, and vllm-gaudi's native MXFP4 MoE method (grouped top-k routing
# included). The port adds:
#   1. a check that the installed vllm-gaudi and vLLM are the commits it was written against
#   2. patches/mxfp4_fused_load.py: load the converted checkpoint's fused MXFP4 experts under expert parallelism
#   3. the kimi_hpu package and patches/mxk_hooks.py: the MXFP4 decode kernel hook (switch KIMI_MXK, off = stock)
#   4. the kernel build: kernels/mxfp4-moe/build.sh --variant kimi into /opt/gaudi-cookbook-kimi/mxk
#   5. a final check that every patch reports "applied" and the hook module imports
# Generic fixes are not applied here: the recipe runs fixes/apply_all.py at launch.
set -euo pipefail
PORT=$(cd "$(dirname "$0")" && pwd)
COOKBOOK=$(cd "$PORT/../.." && pwd)
KERNEL_DIR=/opt/gaudi-cookbook-kimi/mxk   # kimi_hpu/mxk.py reads it from here (KIMI_MXK_LIB)
VLLM_GAUDI_COMMIT=058f107
VLLM_COMMIT=ced6857a
step() { echo "[kimi-k2-7 install] $*"; }

step "1/5 base image: vllm-gaudi $VLLM_GAUDI_COMMIT, vLLM $VLLM_COMMIT"
python3 - "$VLLM_GAUDI_COMMIT" "$VLLM_COMMIT" <<'PY'
import importlib.metadata
import importlib.util
import os
import subprocess
import sys


def commit(pkg, dist):
    # a source install (docs/getting-started.md) has a git checkout; a wheel carries +g<hash> in its version
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
    sys.exit("[kimi-k2-7 install] " + "; ".join(bad))
PY

step "2/5 fused MXFP4 expert loading"
python3 "$PORT/patches/mxfp4_fused_load.py"

step "3/5 kimi_hpu package and the kernel hooks"
# build in a copy: setuptools writes build/ and egg-info next to the sources
BUILD=$(mktemp -d)
cp -r "$PORT/kimi_hpu" "$BUILD/"
PIP_NO_INDEX=1 pip install --no-deps --no-build-isolation --no-cache-dir "$BUILD/kimi_hpu"
rm -rf "$BUILD"
python3 "$PORT/patches/mxk_hooks.py"

step "4/5 MXFP4 decode kernels (variant kimi)"
bash "$COOKBOOK/kernels/mxfp4-moe/build.sh" --variant kimi "$KERNEL_DIR"

step "5/5 checks"
python3 "$PORT/patches/mxfp4_fused_load.py" --check
python3 "$PORT/patches/mxk_hooks.py" --check
python3 -c "import kimi_hpu.mxk as m; assert m.lib_version(), 'no kernel library in ' + m.LIB_DIR; print(m.lib_version())"
step "done"
