#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Install the DeepSeek-V4.1-Flash port into a stock vllm-gaudi releases/v0.30.0 image. The only build step: the
# Dockerfile runs it from /opt/gaudi-cookbook, and so can a container started from the stock image with the
# repository copied in. Stops at the first step that does not apply.
#
#   bash ports/deepseek-v4-1-flash/install.sh        # from the repository root, or any directory
#
# Steps:
#   1. check that the installed vllm-gaudi and vLLM are the commits this port was written against
#   2. install the dsv41_hpu plugin (vLLM general plugin: model, quant config, runner hook, frontend corrections)
#   3. patch vllm_gaudi/__init__.py so every process registers the kernel library before the device opens
#      (patches/dsv41-kernel-env; does nothing unless DSV41_MXK=1)
#   4. build the MXFP4 decode kernels (kernels/mxfp4-moe, variant dsv41) plus the port's Sinkhorn kernel
#      (kernels/dsv41_sinkhorn) into /opt/mxfp4-moe/lib, where dsv41_hpu/mxk.py looks for them
#   5. image gate (kernels/image_gate.py): op library, kernel library version, served layer covered, env hook
#
# The release's generic fixes are not applied here: the recipe's launch step runs fixes/apply_all.py with the
# v0.30.0 profile, and the plugin refuses to start without the fixes it depends on.
set -euo pipefail
PORT=$(cd "$(dirname "$0")" && pwd)
COOKBOOK=$(cd "$PORT/../.." && pwd)
LIB_DIR=${DSV41_MXK_LIB:-/opt/mxfp4-moe/lib}
VLLM_GAUDI_COMMIT=058f107
VLLM_COMMIT=ced6857a
step() { echo "[deepseek-v4-1-flash install] $*"; }

step "1/5 base image: vllm-gaudi $VLLM_GAUDI_COMMIT, vLLM $VLLM_COMMIT"
python3 - "$VLLM_GAUDI_COMMIT" "$VLLM_COMMIT" <<'PYEOF'
import importlib.metadata
import importlib.util
import os
import subprocess
import sys


def commit(pkg, dist):
    # a source install (as in docs/getting-started.md) has a git checkout; a wheel carries +g<hash> in its version
    spec = importlib.util.find_spec(pkg)
    if spec is None:
        return "not installed"
    src = os.path.dirname(list(spec.submodule_search_locations)[0])
    r = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True)
    if r.returncode == 0:
        return r.stdout.strip()
    ver = importlib.metadata.version(dist)
    return ver.split("+g", 1)[1].split(".")[0] if "+g" in ver else ver


bad = []
for pkg, dist, want in (("vllm_gaudi", "vllm_gaudi", sys.argv[1]), ("vllm", "vllm", sys.argv[2])):
    got = commit(pkg, dist)
    print(f"[deepseek-v4-1-flash install] {pkg} {got}")
    if not got.startswith(want):
        bad.append(f"{pkg} is at {got}, the port needs {want}")
if bad:
    sys.exit("[deepseek-v4-1-flash install] " + "; ".join(bad))
PYEOF

step "2/5 plugin"
# build in a copy: setuptools writes build/ and egg-info next to the sources
BUILD=$(mktemp -d)
cp -r "$PORT/dsv41_hpu" "$BUILD/"
PIP_NO_INDEX=1 pip install --no-deps --no-build-isolation --no-cache-dir "$BUILD/dsv41_hpu"
rm -rf "$BUILD"
python3 -c "import importlib.metadata as m; eps = m.entry_points(group='vllm.general_plugins'); \
assert any(e.name == 'dsv41_hpu' for e in eps), 'dsv41_hpu entry point missing'; import dsv41_hpu"

step "3/5 kernel env hook"
python3 "$PORT/patches/dsv41-kernel-env/apply.py"
python3 "$PORT/patches/dsv41-kernel-env/apply.py" --check

step "4/5 MXFP4 decode kernels (variant dsv41) and the Sinkhorn kernel"
bash "$COOKBOOK/kernels/mxfp4-moe/build.sh" --variant dsv41 --extra-kernel "$PORT/kernels/dsv41_sinkhorn" "$LIB_DIR"

step "5/5 image gate"
DSV41_MXK_LIB=$LIB_DIR python3 "$PORT/kernels/image_gate.py"
step "done"
