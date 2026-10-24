# SPDX-License-Identifier: Apache-2.0
# Muse Spark Open serving image: the stock vllm-gaudi releases/v0.30.0 image plus this port.
#
#   docker build -f ports/muse-spark-open/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> \
#     -t <your-muse-spark-open-image> .        # from the repository root
#
# The whole repository goes to /opt/gaudi-cookbook, so the launch step can run fixes/apply_all.py from there.
ARG BASE
FROM ${BASE}
COPY . /opt/gaudi-cookbook
RUN bash /opt/gaudi-cookbook/ports/muse-spark-open/install.sh
