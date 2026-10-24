#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Installs the EmbeddingGemma 2 server's dependencies into the Gaudi 1.24.1 pytorch-installer image. Fails closed.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../common/install_lib.sh"
check_base
$PIP -r "$HERE/requirements.txt"
check_torch_kept
python3 -c "import av, PIL, fastapi; from transformers.models.embedding_gemma2 import EmbeddingGemma2Model"
echo "install: embeddinggemma-2 server ready"
