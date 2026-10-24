# SPDX-License-Identifier: Apache-2.0
# EmbeddingGemma 2 server on the stock Gaudi 1.24.1 PyTorch image (no vLLM). Build from the repository root:
#   docker build -f servers/embeddinggemma-2/Dockerfile -t <registry>/embeddinggemma-2-gaudi:<tag> .
ARG BASE=vault.habana.ai/gaudi-docker/1.24.1/ubuntu24.04/habanalabs/pytorch-installer-2.11.0:latest
FROM ${BASE}
COPY servers /opt/gaudi-cookbook/servers
RUN /opt/gaudi-cookbook/servers/embeddinggemma-2/install.sh
ENV PT_HPU_LAZY_MODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/gaudi-cookbook/servers/common:${PYTHONPATH}
WORKDIR /opt/gaudi-cookbook/servers/embeddinggemma-2
EXPOSE 8000
CMD ["python3", "server.py"]
