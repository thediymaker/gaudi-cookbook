#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Builds the Orpheus image from a stock vllm-gaudi v0.26.0 image: applies the v0.26.0 fix profile plus penalty-shapes
# to the installed vLLM and vllm-gaudi, then installs the sidecar's two packages. Fails closed.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
. "$HERE/../common/install_lib.sh"
check_base
# Exit 2 means the installed vLLM or vllm-gaudi is not the v0.26.0 release the fixes are written for.
set +e
python3 "$ROOT/fixes/apply_all.py" --check --profile v0.26.0 penalty-shapes
rc=$?
set -e
if [ "$rc" -eq 2 ]; then echo "install: this image is not stock vllm-gaudi v0.26.0" >&2; exit 1; fi
python3 "$ROOT/fixes/apply_all.py" --profile v0.26.0 penalty-shapes
$PIP --no-deps -r "$HERE/requirements.txt"
check_torch_kept
python3 -c "import snac, soundfile, httpx, fastapi, vllm"
echo "install: orpheus image ready"
