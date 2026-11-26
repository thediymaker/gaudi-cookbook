# Mistral Large 4 port image. BASE is a stock vllm-gaudi releases/v0.30.0 image built as in docs/getting-started.md
# (vllm-gaudi 058f107, vLLM ced6857a). Build from the repository root:
#   docker build -f ports/mistral-large-4/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> -t <your-ml4-image> .
# install.sh downloads mistral-common 1.13.0 from PyPI (sha256-pinned), so the build needs network access.
ARG BASE
FROM ${BASE}
COPY . /opt/gaudi-cookbook
RUN bash /opt/gaudi-cookbook/ports/mistral-large-4/install.sh
ENV COOKBOOK=/opt/gaudi-cookbook
