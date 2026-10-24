# GLM-5.x MXFP4 port image (GLM-5.2 and GLM-5.3): a stock vllm-gaudi releases/v0.30.0 image plus this port.
# Build from the repository root:
#   docker build -f ports/glm-5-mxfp4/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> -t <your-glm-5-mxfp4-image> .
ARG BASE
FROM ${BASE}

COPY . /opt/gaudi-cookbook
RUN bash /opt/gaudi-cookbook/ports/glm-5-mxfp4/install.sh
ENV COOKBOOK=/opt/gaudi-cookbook
