#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Install the Muse Spark Open port into a stock vllm-gaudi releases/v0.30.0 image. The only build step: the
# Dockerfile runs it, and so can a running container started from the stock image. Stops at the first step
# that does not apply.
#
#   bash ports/muse-spark-open/install.sh            # from the repository root, or any directory
#
# Steps:
#   1. check the installed vllm-gaudi and vLLM are the commits this port was written against
#   2. apply the cookbook's v0.30.0 fix profile (the runner patches below are written on top of it)
#   3. install the vendored files of vLLM PR #61879 (config, parsers, processor) into the vLLM tree
#   4. patch the HPU runner: exact sliding windows for muse_spark, hooks for window-sized KV groups
#   5. install the muse_spark_hpu plugin and the hook that registers the kernel library at process start
#   6. build the MXFP4 decode kernels (kernels/mxfp4-moe, variant muse) into /opt/muse-spark-open/mxfp4-kernels
#   7. build gates (tests/build_gates.py): patches applied, plugin registered, window masks, kernel build
#
# The recipe's launch step runs fixes/apply_all.py again with the same profile plus its opt-in fixes; the profile
# part then reports "applied".
set -euo pipefail
PORT=$(cd "$(dirname "$0")" && pwd)
COOKBOOK=$(cd "$PORT/../.." && pwd)
KERNEL_DIR=/opt/muse-spark-open/mxfp4-kernels   # muse_spark_hpu/mxfp4_kernels.py reads it from here
VLLM_GAUDI_COMMIT=058f107
VLLM_COMMIT=ced6857a
step() { echo "[muse-spark-open install] $*"; }

step "1/7 base image: vllm-gaudi $VLLM_GAUDI_COMMIT, vLLM $VLLM_COMMIT"
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
    sys.exit("[muse-spark-open install] " + "; ".join(bad))
EOF

step "2/7 fix profile v0.30.0"
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.30.0

step "3/7 vendored files of vLLM PR #61879"
python3 "$PORT/patches/vllm_pr_61879.py"

step "4/7 runner patches"
python3 "$PORT/patches/runner_window.py"
python3 "$PORT/patches/swa_kv_hooks.py"

step "5/7 plugin and kernel env hook"
# build in a copy: setuptools writes build/ and egg-info next to the sources
BUILD=$(mktemp -d)
cp -r "$PORT/muse_spark_hpu" "$BUILD/"
PIP_NO_INDEX=1 pip install --no-deps --no-build-isolation --no-cache-dir "$BUILD/muse_spark_hpu"
rm -rf "$BUILD"
python3 "$PORT/patches/kernel_env_hook.py"

step "6/7 MXFP4 decode kernels (variant muse)"
bash "$COOKBOOK/kernels/mxfp4-moe/build.sh" --variant muse "$KERNEL_DIR"

step "7/7 build gates"
python3 "$PORT/tests/build_gates.py" --kernels
step "done"
