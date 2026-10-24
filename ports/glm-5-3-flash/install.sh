#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Install the GLM-5.3-Flash port into a stock vllm-gaudi v0.26.0 image. The only build step; ports/glm-5-3-flash/
# Dockerfile runs it, and it can run in a container started from the stock image the same way:
#
#   bash ports/glm-5-3-flash/install.sh
#
# Steps, each fatal on failure:
#   1. check that vllm-gaudi and vLLM are the v0.26.0 release commits (source checkouts, as in docs/getting-started.md);
#   2. apply the port's source patches (patches/*.py, fail closed, idempotent);
#   3. build the mHC Sinkhorn TPC kernel (kernels/build.sh, via kernels/mxfp4-moe/build.sh);
#   4. install the model plugin (glm5_next_hpu/) in editable mode, so it finds the kernel build next to it;
#   5. re-check every patch and the plugin's own patch check.
# The release's generic fixes are not applied here: the recipe's launch command runs fixes/apply_all.py.
set -euo pipefail
PORT=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3}

VLLM_GAUDI_COMMIT=b059f9c   # vllm-gaudi v0.26.0
VLLM_COMMIT=568afb3a        # its VLLM_STABLE_COMMIT

log() { echo "[glm-5-3-flash install] $*"; }

# 1. release commits, read from the source checkouts without importing vLLM
check_commit() {  # check_commit <package> <expected commit prefix>
  local pkg=$1 want=$2 dir head
  dir=$("$PY" -c "import importlib.util,os,sys;s=importlib.util.find_spec('$pkg');\
sys.exit(1) if s is None else print(os.path.dirname(list(s.submodule_search_locations)[0]))") || {
    log "$pkg is not installed"; exit 1; }
  head=$(git -C "$dir" rev-parse HEAD 2>/dev/null) || {
    log "$pkg at $dir is not a git checkout; build the image as in docs/getting-started.md"; exit 1; }
  case $head in
    "$want"*) log "$pkg $head" ;;
    *) log "$pkg is at $head, expected $want (v0.26.0); this port is only verified on that release"; exit 1 ;;
  esac
}
check_commit vllm_gaudi "$VLLM_GAUDI_COMMIT"
check_commit vllm "$VLLM_COMMIT"

# 2. source patches (exit 2 = the installed source differs from what the patch expects)
PATCHES=(nope_mla mla_model_type cache_group_idx mhc_kernel_env)
for p in "${PATCHES[@]}"; do
  "$PY" "$PORT/patches/$p.py" || { log "patch $p failed"; exit 1; }
done

# 3. kernel
"$PORT/kernels/build.sh" "$PORT/kernels/dist"

# 4. plugin
"$PY" -m pip install --no-build-isolation --no-deps -q -e "$PORT/glm5_next_hpu"

# 5. verify
for p in "${PATCHES[@]}"; do
  "$PY" "$PORT/patches/$p.py" --check >/dev/null || { log "patch $p is not applied after install"; exit 1; }
done
"$PY" -c "from glm5_next_hpu.patch_check import missing_patches as m; import sys; sys.exit(1 if m() else 0)" || {
  log "plugin patch check failed"; exit 1; }
log "done"
