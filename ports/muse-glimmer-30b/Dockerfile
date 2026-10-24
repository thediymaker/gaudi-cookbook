# Muse-Glimmer-30B port image. BASE is a stock vllm-gaudi releases/v0.30.0 image built as in
# docs/getting-started.md (vllm-gaudi 058f107, vLLM ced6857a). Build from the repository root:
#   docker build -f ports/muse-glimmer-30b/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> -t <port-image> .
ARG BASE
FROM ${BASE}
COPY . /opt/gaudi-cookbook
RUN bash /opt/gaudi-cookbook/ports/muse-glimmer-30b/install.sh
