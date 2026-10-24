# Beam-1.0 port image: a stock vllm-gaudi releases/v0.30.0 image plus this port.
# Build from the repository root:
#   docker build -f ports/beam-1-0/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> -t <your-beam-1-0-image> .
ARG BASE
FROM ${BASE}

COPY . /opt/gaudi-cookbook
RUN bash /opt/gaudi-cookbook/ports/beam-1-0/install.sh
ENV COOKBOOK=/opt/gaudi-cookbook
