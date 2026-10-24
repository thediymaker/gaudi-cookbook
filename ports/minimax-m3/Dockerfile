# MiniMax-M3 port image: a stock vllm-gaudi v0.26.0 image plus this port.
# Build from the repository root:
#   docker build -f ports/minimax-m3/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.26.0-image> -t <your-minimax-m3-image> .
ARG BASE
FROM ${BASE}

COPY . /opt/gaudi-cookbook
RUN bash /opt/gaudi-cookbook/ports/minimax-m3/install.sh
ENV COOKBOOK=/opt/gaudi-cookbook
