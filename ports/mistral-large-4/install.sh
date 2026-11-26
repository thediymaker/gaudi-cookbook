#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Install the Mistral Large 4 port into a stock vllm-gaudi releases/v0.30.0 image. The only build step: the
# Dockerfile runs it, and so can a running container started from the stock image. Stops at the first step that
# does not apply.
#
#   bash ports/mistral-large-4/install.sh            # from the repository root, or any directory
#
# Steps:
#   1. check the installed vllm-gaudi and vLLM are the commits this port was written against
#   2. mistral-common 1.13.0 (--no-deps, sha256-pinned): the image's 1.11.7 lacks the low / medium reasoning efforts
#      and the request grammar API the vendored parser calls; 1.13.0's own requirements are all met by the image
#   3. the vendored files of vLLM PR #64534 (params.json adapter, tokenizer, renderer, /tokenize, parser) over the
#      stock v0.30.0 files, each only over the recorded stock bytes
#   4. the mistral_large4_hpu plugin and the hook that registers the kernel libraries at process start
#   5. the MXFP4 MoE kernels (kernels/mxfp4-moe, variant ml4) into /opt/gaudi-cookbook-ml4/mxk
#   6. the model-only decode-step kernels (kernels/k1) into /opt/gaudi-cookbook-ml4/k1
#   7. build gates (tests/build_gates.py --kernels): files, plugin registration, config, quant claim, kernels
# The release's generic fixes are not applied here: the recipe's launch step runs fixes/apply_all.py with the
# v0.30.0 profile.
set -euo pipefail
PORT=$(cd "$(dirname "$0")" && pwd)
COOKBOOK=$(cd "$PORT/../.." && pwd)
OPT=/opt/gaudi-cookbook-ml4        # mistral_large4_hpu/mxk.py and k1ix.py read the libraries from here
VLLM_GAUDI_COMMIT=058f107
VLLM_COMMIT=ced6857a
MC_VERSION=1.13.0
MC_SHA256=6304455b69ae9ca02cddb9bb3bb888ab8dcbe3945a3cafe3368897795611993d   # mistral_common-1.13.0-py3-none-any.whl
step() { echo "[mistral-large-4 install] $*"; }

step "1/7 base image: vllm-gaudi $VLLM_GAUDI_COMMIT, vLLM $VLLM_COMMIT"
python3 - "$VLLM_GAUDI_COMMIT" "$VLLM_COMMIT" <<'EOF'
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
    sys.exit("[mistral-large-4 install] " + "; ".join(bad))
EOF

step "2/7 mistral-common $MC_VERSION"
REQ=$(mktemp)
echo "mistral-common==$MC_VERSION --hash=sha256:$MC_SHA256" > "$REQ"
pip install --no-deps --no-cache-dir --require-hashes --only-binary=:all: -r "$REQ"
rm -f "$REQ"
python3 -c "import mistral_common as m, sys; sys.exit(0 if m.__version__ == '$MC_VERSION' else 'mistral-common ' + m.__version__)"

step "3/7 vendored files of vLLM PR #64534"
# find_spec, not import: importing vllm logs to stdout
VLLM_ROOT=$(python3 -c "import importlib.util as u, os; print(os.path.dirname(list(u.find_spec('vllm').submodule_search_locations)[0]))")
python3 "$PORT/patches/install_pr_files.py" "$PORT/vendor" "$VLLM_ROOT"

step "4/7 plugin and kernel env hook"
# build in a copy: setuptools writes build/ and egg-info next to the sources
BUILD=$(mktemp -d)
cp -r "$PORT/ml4_hpu" "$BUILD/"
PIP_NO_INDEX=1 pip install --no-deps --no-build-isolation --no-cache-dir "$BUILD/ml4_hpu"
rm -rf "$BUILD"
python3 "$PORT/patches/kernel_env_hook.py"

step "5/7 MXFP4 MoE kernels (variant ml4)"
bash "$COOKBOOK/kernels/mxfp4-moe/build.sh" --variant ml4 "$OPT/mxk"

step "6/7 decode-step kernels"
bash "$PORT/kernels/k1/build_k1.sh" "$OPT/k1"

step "7/7 build gates"
python3 "$PORT/tests/build_gates.py" --kernels
step "done"
