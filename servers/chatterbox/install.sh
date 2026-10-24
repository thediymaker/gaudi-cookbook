#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Installs the Chatterbox server's dependencies into the Gaudi 1.24.1 pytorch-installer image. Fails closed.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../common/install_lib.sh"
check_base
$PIP -r "$HERE/requirements.txt"
$PIP --no-deps --extra-index-url https://download.pytorch.org/whl/cpu -r "$HERE/requirements-nodeps.txt"
check_torch_kept
python3 -c "import chatterbox.tts, chatterbox.tts_turbo, perth, soundfile, fastapi"
echo "install: chatterbox server ready"
