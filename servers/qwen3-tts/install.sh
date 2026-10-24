#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Installs the Qwen3-TTS server's dependencies into the Gaudi 1.24.1 pytorch-installer image. Fails closed.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../common/install_lib.sh"
check_base
# sox: the qwen-tts package imports the sox binding, which needs the binary.
apt-get update && apt-get install -y --no-install-recommends sox && rm -rf /var/lib/apt/lists/*
$PIP -r "$HERE/requirements.txt"
$PIP --no-deps --extra-index-url https://download.pytorch.org/whl/cpu -r "$HERE/requirements-nodeps.txt"
check_torch_kept
python3 -c "import qwen_tts, torchaudio, soundfile, fastapi"
echo "install: qwen3-tts server ready"
