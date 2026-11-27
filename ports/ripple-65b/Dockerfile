# SPDX-License-Identifier: Apache-2.0
# Ripple-65B serving image: the stock vllm-gaudi releases/v0.30.0 image plus this port.
#
#   docker build -f ports/ripple-65b/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> \
#     -t <your-ripple-65b-image> .        # from the repository root
#
# The whole repository goes to /opt/gaudi-cookbook, so the launch step can run fixes/apply_all.py from there.
ARG BASE
FROM ${BASE}
COPY . /opt/gaudi-cookbook
RUN bash /opt/gaudi-cookbook/ports/ripple-65b/install.sh
ENV COOKBOOK=/opt/gaudi-cookbook
