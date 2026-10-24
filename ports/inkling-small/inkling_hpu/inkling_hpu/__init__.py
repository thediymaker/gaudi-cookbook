# SPDX-License-Identifier: Apache-2.0
"""Inkling-Small for vLLM on Intel Gaudi (vllm-gaudi releases/v0.30.0), text only.

Loaded as a vLLM general plugin (entry point `vllm.general_plugins: inkling_hpu`) in every vLLM process,
before the model config is parsed. It registers the config for model_type inkling_mm_model, the HPU model
for both Inkling architectures (in place of vLLM's CUDA-only one), and the `inkling_py` reasoning and tool
parsers.
"""

ARCHITECTURES = ("InklingForConditionalGeneration", "InklingForCausalLM")


def register() -> None:
    from vllm import ModelRegistry
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    from .config import register_config

    register_config()
    for arch in ARCHITECTURES:
        ModelRegistry.register_model(arch, "inkling_hpu.model:HPUInklingForCausalLM")
    ToolParserManager.register_lazy_module("inkling_py", "inkling_hpu.parsers", "InklingPyToolParser")
    ReasoningParserManager.register_lazy_module("inkling_py", "inkling_hpu.parsers", "InklingPyReasoningParser")
