# SPDX-License-Identifier: Apache-2.0
"""Muse Spark Open (AMD Quark MXFP4 checkpoint) for vLLM on Intel Gaudi 2, vllm-gaudi v0.30.0.

A vLLM general plugin (entry point `vllm.general_plugins: muse_spark_hpu`), loaded in every vLLM process.
register():
  - points the architecture MuseSparkForConditionalGeneration at the HPU model (by module path, so registration
    never imports HPU code);
  - adds the Muse Spark config classes of vLLM PR #61879 to vLLM's config registry;
  - registers the PR's reasoning parser and a forced-tool_choice subclass of its tool parser under the PR's names
    `muse` and `musespark1x`;
  - registers the `muse_spark_hpu_mxfp4` quantization config, which claims the AMD Quark MXFP4 checkpoint;
  - installs the structured-output gate and the xgrammar fixes for Muse (structured.py).

The PR files are installed into the vLLM tree by patches/vllm_pr_61879.py; without them registration fails.
The runner hooks are installed by the model constructor, inside the worker processes.

Switches (environment, read at import):
  MUSE_SPARK_MOE_TP_SPLIT=4            expert layout EP2 x TP4 at TP 8 (moe.py); the recipe's value
  MUSE_SPARK_SWA_KV=1                  window-sized KV for the 46 sliding layers (swa_kv.py)
  MUSE_SPARK_LONG_ATTN=1               long-context prompt attention without a dense bias (long_context.py)
  MUSE_SPARK_DECODE_GRAPH_MAX_BLOCKS=N decode batches above N blocks run without an HPU graph (long_context.py)
  MUSE_SPARK_DECODE_LAYER_GROUP=62     one mark_step per 62 layers in decode graphs (model.py)
  MUSE_SPARK_MXFP4_KERNELS=1           MXFP4 decode kernels instead of the stock MoE op (mxfp4_kernels.py)
  MUSE_SPARK_MEMLOG=1                  device memory log lines from TP rank 0 (runner_hooks.py)
"""

MODEL_ARCH = "MuseSparkForConditionalGeneration"
MODEL_CLASS = "muse_spark_hpu.model:MuseSparkHPUForConditionalGeneration"
PARSER_NAMES = ("muse", "musespark1x")


def register() -> None:
    from vllm import ModelRegistry
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    try:
        import vllm.transformers_utils.configs.muse_spark as C
    except ImportError as e:
        raise ImportError(
            "muse_spark_hpu: the vLLM PR #61879 files are missing from the vLLM tree (patches/vllm_pr_61879.py)"
        ) from e

    ModelRegistry.register_model(MODEL_ARCH, MODEL_CLASS)
    _CONFIG_REGISTRY.update(
        muse_spark=C.MuseSparkConfig,
        muse_spark_text=C.MuseSparkTextConfig,
        muse_spark_vision=C.MuseSparkVisionConfig,
        muse_spark_audio=C.MuseSparkAudioConfig,
    )
    for name in PARSER_NAMES:
        ReasoningParserManager.register_lazy_module(name, "vllm.reasoning.muse_reasoning_parser", "MuseReasoningParser")
        ToolParserManager.register_lazy_module(name, "muse_spark_hpu.tool_parser", "MuseSparkHPUToolParser")
    from . import quant  # noqa: F401  (registers muse_spark_hpu_mxfp4)
    from . import structured

    structured.install()
