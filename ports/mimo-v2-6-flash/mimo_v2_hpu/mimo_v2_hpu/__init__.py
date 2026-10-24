# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2.6-Flash for vLLM on Intel Gaudi (vllm-gaudi v0.26.0).

Loaded as a vLLM general plugin (entry point `vllm.general_plugins: mimo_v2_hpu`) in every vLLM process. It routes
the MiMo-V2 architectures to the HPU model. vLLM rewrites the architecture of a MiMo-V2.6 checkpoint to
MiMoV2OmniForCausalLM because its config carries a vision_config; that name gets the text model too.
"""

ARCHITECTURES = ("MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM", "MiMoV2OmniForCausalLM")


def register() -> None:
    from vllm import ModelRegistry

    for arch in ARCHITECTURES:
        ModelRegistry.register_model(arch, "mimo_v2_hpu.model:MiMoV2HpuForCausalLM")
