# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2.6-Flash and MiMo-V2.6-Pro for vLLM on Intel Gaudi (vllm-gaudi v0.26.0).

Loaded as a vLLM general plugin (entry point `vllm.general_plugins: mimo_v2_hpu`) in every vLLM process. It routes
the MiMo-V2 architectures to the HPU model. vLLM rewrites the architecture of a MiMo-V2.6 checkpoint to
MiMoV2OmniForCausalLM because its config carries a vision_config; that name gets the text model too.

With MIMO_MXK=1 it also points the graph compiler at the MXFP4 MoE kernel library (mxk.setup_env). vLLM loads
general plugins in the API server process while it parses the command line, so the engine and worker processes
inherit that environment before any of them opens a device.
"""

ARCHITECTURES = ("MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM", "MiMoV2OmniForCausalLM")


def register() -> None:
    from vllm import ModelRegistry

    from . import mxk

    mxk.setup_env()
    for arch in ARCHITECTURES:
        ModelRegistry.register_model(arch, "mimo_v2_hpu.model:MiMoV2HpuForCausalLM")
