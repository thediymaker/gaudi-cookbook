# SPDX-License-Identifier: Apache-2.0
# MiMo-V2.6 port image (Flash and Pro): a stock vllm-gaudi v0.26.0 image plus this repository and the port's
# install step.
#   docker build -f ports/mimo-v2-6-flash/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.26.0-image> \
#     -t <your-mimo-v2-6-flash-image> .
ARG BASE
FROM ${BASE}
COPY . /opt/gaudi-cookbook
RUN bash /opt/gaudi-cookbook/ports/mimo-v2-6-flash/install.sh
