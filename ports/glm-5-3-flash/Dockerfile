# SPDX-License-Identifier: Apache-2.0
# GLM-5.3-Flash port image, built from a stock vllm-gaudi v0.26.0 image (docs/getting-started.md). From the repository
# root:
#   docker build -f ports/glm-5-3-flash/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.26.0-image> -t <port-image> .
ARG BASE
FROM ${BASE}
COPY . /opt/gaudi-cookbook
RUN bash /opt/gaudi-cookbook/ports/glm-5-3-flash/install.sh
