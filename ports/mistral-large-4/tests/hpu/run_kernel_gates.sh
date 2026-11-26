#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# One-card kernel gates of the Mistral Large 4 port, in the installed port image with one free card (run them before
# the server starts, or in a separate one-card container):
#
#   ports/mistral-large-4/tests/hpu/run_kernel_gates.sh [OUT_DIR]
#
#   parity   kernels/mxfp4-moe/tests/run_parity.sh --variant ml4 on the built library: PARITY PASS (v8 and the
#            grouped path, grouped bit-identical to v8), PREFILL PASS, SIG_DEVICE PASS
#   k1       ML4_INDEX_KERNEL: the fused decode indexer returns the native path's indices, eager and in an HPU graph
#   k2       ML4_ATTN_KERNEL: the sparse-attention gather gives the native path's bits
#   r5, r6   ML4_R5_* / ML4_R6_* node-cut kernels: each bitwise equal to the native functions
# Prints one PASS or FAIL line per gate; exit 1 if any fails.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COOKBOOK="$(cd "$HERE/../../../.." && pwd)"
OUT="${1:-ml4_kernel_gates}"
LIB="${ML4_MXK_LIB:-/opt/gaudi-cookbook-ml4/mxk}"
mkdir -p "$OUT"
rc=0
one() {
  local name=$1; shift
  if "$@" > "$OUT/$name.log" 2>&1; then echo "PASS $name"; else echo "FAIL $name (see $OUT/$name.log)"; rc=1; fi
}
one parity bash "$COOKBOOK/kernels/mxfp4-moe/tests/run_parity.sh" --variant ml4 --lib "$LIB" --out "$OUT/parity"
one k1 env ML4_INDEX_KERNEL=1 python3 "$HERE/kernel_k1_hpu.py"
one k2 env ML4_ATTN_KERNEL=1 ML4_INDEX_KERNEL=1 python3 "$HERE/kernel_k2_hpu.py"
one r5 python3 "$HERE/kernel_r5_hpu.py"
one r6 python3 "$HERE/kernel_r6_hpu.py"
exit $rc
