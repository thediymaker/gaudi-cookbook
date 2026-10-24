# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash (`glm5_next`) for vLLM on Gaudi 2, as an out-of-tree plugin.

`register()` runs in every vLLM process through the `vllm.general_plugins` entry point. It

1. registers the vendored `glm5_next` config and processors with transformers (the image's transformers has none);
2. checks that the port's source patches are in the installed vLLM and vllm-gaudi (install.sh applies them at
   image build; a patch applied from here would edit files that this process has already imported);
3. maps the architecture to vLLM's hybrid attention + state config, so the KV cache spec gets a state block size;
4. installs the mHC Sinkhorn kernel when `GLM5_NEXT_MHC_SINKHORN=kernel`;
5. registers `Glm5NextForConditionalGeneration` with vLLM's model registry.
"""

from __future__ import annotations

_DONE = False


def register() -> None:
    global _DONE
    if _DONE:
        return

    from .config import register as register_config

    register_config()

    from .patch_check import require_patches

    require_patches()

    # Without this mapping vLLM computes no mamba_block_size for the KDA layers and KV cache spec construction
    # fails on `assert mamba_block_size is not None`. Qwen3.5 (also a linear-attention hybrid) uses the same config.
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP, HybridAttentionMambaModelConfig

    MODELS_CONFIG_MAP.setdefault("Glm5NextForConditionalGeneration", HybridAttentionMambaModelConfig)

    from . import mhc

    mhc.install()

    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Glm5NextForConditionalGeneration", "glm5_next_hpu.model:Glm5NextForConditionalGeneration"
    )
    _DONE = True
