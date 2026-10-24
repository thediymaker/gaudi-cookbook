# SPDX-License-Identifier: Apache-2.0
"""Beam-1.0 (compressed-tensors FP8) for vLLM on Intel Gaudi 2 (vllm-gaudi releases/v0.30.0).

Loaded as a vLLM general plugin (entry point `vllm.general_plugins: beam_hpu`) in every process:
API server, engine core and workers. register():
  - points architecture BeamForCausalLM at the HPU model, by module path, so registering never
    imports HPU code;
  - adds BeamConfig (from vLLM PR #62074, config.py) to vLLM's config registry for model_type "beam";
  - registers load format "beam" (loader.py; opt-in with --load-format beam);
  - installs the frontend: the `beam` reasoning and tool parsers and the request-side corrections
    (frontend.py).
"""

MODEL_ARCH = "BeamForCausalLM"
MODEL_CLASS = "beam_hpu.vllm_model:BeamHPUForCausalLM"


def register() -> None:
    from vllm import ModelRegistry
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    from . import frontend, loader  # noqa: F401  (loader registers load format "beam" on import)
    from .config import BeamConfig

    ModelRegistry.register_model(MODEL_ARCH, MODEL_CLASS)
    _CONFIG_REGISTRY["beam"] = BeamConfig
    frontend.install()
