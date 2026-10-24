# Kimi-K2.7-Code port image. BASE is a stock vllm-gaudi releases/v0.30.0 image built as in
# docs/getting-started.md (vllm-gaudi 058f107, vLLM ced6857a). Build from the repository root:
#   docker build -f ports/kimi-k2-7/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> -t <your-kimi-k2-7-image> .
ARG BASE
FROM ${BASE}
COPY . /opt/gaudi-cookbook
RUN bash /opt/gaudi-cookbook/ports/kimi-k2-7/install.sh
ENV COOKBOOK=/opt/gaudi-cookbook
