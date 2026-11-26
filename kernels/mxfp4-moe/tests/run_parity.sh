#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Full one-card parity run of one kernel variant, one implementation per process. Inside a vllm-gaudi container
# with one Gaudi 2 card and a build of that variant:
#   tests/run_parity.sh --variant <gptoss|glm|dsv41|muse|kimi|ml4> --lib <build dir> [--checkpoint <dir>] [--out <dir>]
# (any other parity_hpu.py option is passed through). Exit 0 only if the CPU tests (when pytest is installed), the
# parity comparison (PARITY PASS, and PREFILL PASS for glm and ml4) and the identical-signature check (SIG_DEVICE
# PASS) all pass. The kernels and the signature check run once per decode path of the build (v8, then each v10 path
# of a v10 build, and ml4's grouped path), and PARITY PASS then also requires every other path to be bit-identical
# to the v8 path.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
P="python3 $HERE/parity_hpu.py"
if python3 -c "import pytest" 2>/dev/null; then python3 -m pytest -q "$HERE"; fi
$P --impl ref "$@"
$P --impl stock "$@"
PATHS=$($P --impl paths "$@")
for p in $PATHS; do $P --impl ours --path "$p" "$@"; done
$P --impl cmp "$@"
for p in $PATHS; do $P --impl sig --path "$p" "$@"; done
