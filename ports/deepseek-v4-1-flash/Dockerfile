# SPDX-License-Identifier: Apache-2.0
# DeepSeek-V4.1-Flash serving image: the stock vllm-gaudi releases/v0.30.0 image plus this port.
#
#   docker build -f ports/deepseek-v4-1-flash/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> \
#     -t <your-deepseek-v4-1-flash-image> .        # from the repository root
#
# The whole repository goes to /opt/gaudi-cookbook, so the launch step can run fixes/apply_all.py and the port's
# serving scripts from there.
ARG BASE
FROM ${BASE}
COPY . /opt/gaudi-cookbook
RUN bash /opt/gaudi-cookbook/ports/deepseek-v4-1-flash/install.sh
