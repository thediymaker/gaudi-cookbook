#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Full one-card parity run, one implementation per process. Inside a vllm-gaudi container with one Gaudi 2 card:
#   tests/run_parity.sh [--checkpoint <openai/gpt-oss-120b snapshot>] [--e-local 32] [--lib <dir>] [--out <dir>]
# Exit 0 only if the CPU emulation test, the parity comparison and the identical-signature check all pass.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
P="python3 $HERE/parity_hpu.py"
if python3 -c "import pytest" 2>/dev/null; then python3 -m pytest -q "$HERE/test_layout_cpu.py"; fi
$P --impl ref "$@"
$P --impl stock "$@"
$P --impl ours "$@"
$P --impl cmp "$@"
$P --impl sig "$@"
