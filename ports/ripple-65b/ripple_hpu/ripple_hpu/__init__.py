# SPDX-License-Identifier: Apache-2.0
"""Ricursive Ripple-65B-Preview for vLLM on Intel Gaudi 2, vllm-gaudi v0.30.0.

A vLLM general plugin (entry point `vllm.general_plugins: ripple_hpu`), loaded in every vLLM process.
register():
  - points the architecture RicursiveForCausalLM at the HPU model (by module path, so registration never
    imports HPU code);
  - adds RicursiveConfig (vLLM PR #65745, config.py) to vLLM's config registry for model type "ricursive", so
    no trust_remote_code is needed;
  - registers the PR's `ripple` reasoning and tool parsers (parser.py, on v0.30.0's parser engine).

The runner hooks are installed by the model constructor, inside the worker processes.

Switches (environment, read at import):
  RIPPLE_SWA_KV=1                   window-sized KV for the 64 sliding-window layers (swa_kv.py)
  RIPPLE_CONSTRAIN_AUTO_TOOLS=1     tool_choice auto gets the tool grammar with non-strict tools (parser.py)
  RIPPLE_FAST_SOFTMAX_MIN_KV=N      fast softmax for full-layer prompt attention against N+ keys (runner_hooks.py)
  RIPPLE_FSDPA_WS_GIB, _Q_TILE      FusedSDPA prompt calls above the workspace run as query tiles (runner_hooks.py)
  RIPPLE_GRAPH_BUDGET_GIB           device memory bound for decode HPU graphs (graph_budget.py)
  RIPPLE_PREFILL_INFLIGHT=K         layers a long prefill may run ahead of the device (model.py)
"""

MODEL_ARCH = "RicursiveForCausalLM"
MODEL_CLASS = "ripple_hpu.model:RippleHPUForCausalLM"
PARSER = "ripple"


def register() -> None:
    from vllm import ModelRegistry
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    from .config import RicursiveConfig

    ModelRegistry.register_model(MODEL_ARCH, MODEL_CLASS)
    _CONFIG_REGISTRY["ricursive"] = RicursiveConfig
    ReasoningParserManager.register_lazy_module(PARSER, "ripple_hpu.parser", "RippleParserReasoningAdapter")
    ToolParserManager.register_lazy_module(PARSER, "ripple_hpu.parser", "RippleEngineToolParser")
