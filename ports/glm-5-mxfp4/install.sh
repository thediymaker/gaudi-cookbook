#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Install the GLM-5.x MXFP4 port (GLM-5.2 and GLM-5.3) into a stock vllm-gaudi releases/v0.30.0 image (Gaudi
# software 1.24.1). The only build step: the Dockerfile runs it, and so can a running container started from the
# stock image. Stops at the first step that does not apply, so an image is never left half-installed.
#
#   bash ports/glm-5-mxfp4/install.sh
#
# vLLM at this commit already has the model (GlmMoeDsaForCausalLM, the DeepSeek MTP layer, the glm47 tool and
# reasoning parsers) and vllm-gaudi's native MXFP4 MoE method. The port adds:
#   1. a check that the installed vllm-gaudi and vLLM are the commits it was written against;
#   2. the cookbook fixes: the v0.30.0 profile plus vllm-long-prefill-share and glm47-streaming-text-markers. The
#      port's patches are tested on top of the fixed sources; the recipe's launch step runs the same command again
#      and finds every fix applied;
#   3. the glm5_hpu package (DSA top-k attention, MXFP4 kernel hooks, EP x TP expert layout, decode levers);
#   4. the port patches (patches/apply_all.py: MXFP4 fused loading, FusedSDPA softmax mode, MTP and speculative
#      decoding on the HPU runner, and the hooks into glm5_hpu);
#   5. the kernel build: kernels/mxfp4-moe/build.sh --variant glm into /opt/gaudi-cookbook-glm5/mxk;
#   6. a final check that every fix and patch reports "applied" and the package finds the kernels.
set -euo pipefail
PORT=$(cd "$(dirname "$0")" && pwd)
COOKBOOK=$(cd "$PORT/../.." && pwd)
KERNEL_DIR=/opt/gaudi-cookbook-glm5/mxk   # glm5_hpu/mxfp4_moe.py reads it from here (LIB_DIR)
VLLM_GAUDI_COMMIT=058f107
VLLM_COMMIT=ced6857a
FIXES=(--profile v0.30.0 vllm-long-prefill-share glm47-streaming-text-markers)
step() { echo "[glm-5-mxfp4 install] $*"; }

step "1/6 base image: vllm-gaudi $VLLM_GAUDI_COMMIT, vLLM $VLLM_COMMIT"
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
    sys.exit("[glm-5-mxfp4 install] " + "; ".join(bad))
PY

step "2/6 cookbook fixes: ${FIXES[*]}"
python3 "$COOKBOOK/fixes/apply_all.py" "${FIXES[@]}"

step "3/6 glm5_hpu package"
# build in a copy: setuptools writes build/ and egg-info next to the sources
BUILD=$(mktemp -d)
cp -r "$PORT/glm5_hpu" "$BUILD/"
find "$BUILD" -name __pycache__ -prune -exec rm -rf {} +
PIP_NO_INDEX=1 pip install --no-deps --no-build-isolation --no-cache-dir "$BUILD/glm5_hpu"
rm -rf "$BUILD"

step "4/6 port patches"
python3 "$PORT/patches/apply_all.py"

step "5/6 MXFP4 MoE kernels (variant glm)"
bash "$COOKBOOK/kernels/mxfp4-moe/build.sh" --variant glm "$KERNEL_DIR"

step "6/6 checks"
python3 "$COOKBOOK/fixes/apply_all.py" --check "${FIXES[@]}"
python3 "$PORT/patches/apply_all.py" --check
python3 -c "import glm5_hpu.mxfp4_moe as m; assert m.lib_version(), 'no kernel library in ' + m.LIB_DIR; print(m.lib_version())"
step "done"
