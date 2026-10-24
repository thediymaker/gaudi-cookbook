#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Build the GLM-5.3-Flash mHC Sinkhorn kernel inside the serving image (install.sh runs this):
#
#   ports/glm-5-3-flash/kernels/build.sh [OUT_DIR]       default OUT_DIR: ports/glm-5-3-flash/kernels/dist
#
# The kernel library glue and build live once, in kernels/mxfp4-moe. This port needs none of its MXFP4 MoE kernels,
# so it builds the empty variant plus one model-only kernel (glm_mhc_sinkhorn/). Output: libmxk_tpc.so (this kernel and
# every stock GUID, forwarded, for GC_KERNEL_PATH), libmxk_op_lazy.so (torch.ops.custom_op.glm_mhc_sinkhorn) and
# MANIFEST. Compiled files are build outputs; do not commit them.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
OUT=${1:-$HERE/dist}
# tpc-clang records the build directory in the ELF; a fixed path keeps the kernel ELF reproducible.
export MXK_BUILD_DIR=${MXK_BUILD_DIR:-/tmp/glm_mhc_tpc_build}
exec "$ROOT/kernels/mxfp4-moe/build.sh" --variant none --extra-kernel "$HERE/glm_mhc_sinkhorn" "$OUT"
