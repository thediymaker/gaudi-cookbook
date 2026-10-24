# SPDX-License-Identifier: Apache-2.0
# Orpheus 3B: one image for both containers of the pod, built FROM a stock vllm-gaudi v0.26.0 image (built as in
# docs/getting-started.md). The vLLM container runs serve_llm.sh on one card; the sidecar container runs the speech
# server on the CPU only (give it no card). Build from the repository root:
#   docker build -f servers/orpheus/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.26.0-image> \
#     -t <registry>/orpheus-gaudi:<tag> .
ARG BASE
FROM ${BASE}
COPY fixes /opt/gaudi-cookbook/fixes
COPY servers /opt/gaudi-cookbook/servers
RUN /opt/gaudi-cookbook/servers/orpheus/install.sh
ENV PYTHONPATH=/opt/gaudi-cookbook/servers/common:${PYTHONPATH}
WORKDIR /opt/gaudi-cookbook/servers/orpheus
# vLLM container:    ./serve_llm.sh <model>
# sidecar container: python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
