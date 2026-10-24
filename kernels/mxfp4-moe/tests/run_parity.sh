#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Full one-card parity run of one kernel variant, one implementation per process. Inside a vllm-gaudi container
# with one Gaudi 2 card and a build of that variant:
#   tests/run_parity.sh --variant <gptoss|glm|dsv41|muse|kimi> --lib <build dir> [--checkpoint <dir>] [--out <dir>]
# (any other parity_hpu.py option is passed through). Exit 0 only if the CPU tests (when pytest is installed), the
# parity comparison (PARITY PASS, and PREFILL PASS for glm) and the identical-signature check (SIG_DEVICE PASS)
# all pass.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
P="python3 $HERE/parity_hpu.py"
if python3 -c "import pytest" 2>/dev/null; then python3 -m pytest -q "$HERE"; fi
$P --impl ref "$@"
$P --impl stock "$@"
$P --impl ours "$@"
$P --impl cmp "$@"
$P --impl sig "$@"
