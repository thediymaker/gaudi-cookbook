# SPDX-License-Identifier: Apache-2.0
# Inkling-Small port image: the stock vllm-gaudi releases/v0.30.0 image plus this port.
#   docker build -f ports/inkling-small/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> -t <inkling-image> .
# Run from the repository root; the whole repository is copied so the fix pack is in the image too.
ARG BASE
FROM ${BASE}
COPY . /opt/gaudi-cookbook
RUN bash /opt/gaudi-cookbook/ports/inkling-small/install.sh
