#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install-time gates for the Ripple-65B port: run by install.sh after every step, no device or checkpoint.

The checkpoint's config.json is a fixture (tests/fixtures/), the chat template is serving/chat_template.jinja. Each
gate prints one PASS or FAIL line; any failure exits 1.

  patches       every port patch reports "applied" (--check exits 0), and so does fixes/swa-window-context
  entry_point   ripple_hpu:register is installed in the vllm.general_plugins group
  register      load_general_plugins() registered RicursiveForCausalLM (by module path), RicursiveConfig for
                model type ricursive, and the `ripple` reasoning and tool parsers
  imports       every module of the package imports
  config        the checkpoint config.json parses into RicursiveConfig: 80 layers, NoPE full layers at 4, 9, ...,
                79, window 1024, value residuals, fp32 gate, clamp 20
  template      the chat template renders tools, and an assistant turn with reasoning and a tool call renders
                <think> and <tool_call> (the parser's input layout); content format "string"
  parser        the composed ripple parser splits reasoning, content and a tool call, and tool_choice auto with
                non-strict tools gets the hermes tool grammar (RIPPLE_CONSTRAIN_AUTO_TOOLS)
  model_class   the architecture resolves to ripple_hpu.model's class
  hooks         the vllm-gaudi functions runner_hooks.py and graph_budget.py wrap keep the signatures and
                internals they rely on (ModuleFusedSDPA.forward, _fsdpa_num_q_tiles, _execute_model_generic,
                wrap_in_hpu_graph's cache dict, CachedParams counters, the block-FP8 linear)
  runner_window the sliding-window masks of the patched runner (tests/test_runner_window.py)
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import pathlib
import pkgutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PORT = HERE.parent
COOKBOOK = PORT.parent.parent
FIXTURES = HERE / "fixtures"
ARCH = "RicursiveForCausalLM"
MODEL_MODULE, MODEL_CLASS = "ripple_hpu.model", "RippleHPUForCausalLM"
PATCHES = ("parser_streams_all.py", "runner_window.py", "swa_kv_hooks.py")

results: list[tuple[str, bool, str]] = []


def gate(name, fn, *args):
    try:
        results.append((name, True, fn(*args) or "ok"))
    except Exception as e:  # every failure is reported, then exit 1
        results.append((name, False, f"{type(e).__name__}: {e}"))


def g_patches():
    for p in PATCHES:
        r = subprocess.run([sys.executable, str(PORT / "patches" / p), "--check"], capture_output=True, text=True)
        assert r.returncode == 0, f"{p}: {(r.stdout + r.stderr).strip()}"
    fix = COOKBOOK / "fixes" / "swa-window-context" / "apply.py"
    r = subprocess.run([sys.executable, str(fix), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, f"swa-window-context: {(r.stdout + r.stderr).strip()}"
    return f"{len(PATCHES)} port patches + swa-window-context applied"


def g_entry_point():
    eps = [e.value for e in importlib.metadata.entry_points(group="vllm.general_plugins")]
    assert "ripple_hpu:register" in eps, eps


def g_register():
    from vllm import ModelRegistry
    from vllm.plugins import load_general_plugins
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    load_general_plugins()
    assert ARCH in ModelRegistry.get_supported_archs(), "architecture not registered"
    assert _CONFIG_REGISTRY.get("ricursive").__name__ == "RicursiveConfig", _CONFIG_REGISTRY.get("ricursive")
    r = ReasoningParserManager.get_reasoning_parser("ripple")
    t = ToolParserManager.get_tool_parser("ripple")
    assert r.__name__ == "RippleParserReasoningAdapter" and t.__name__ == "RippleEngineToolParser", (r, t)
    assert t.structural_tag_model == "hermes"
    return "architecture, config, parsers"


def g_imports():
    import ripple_hpu

    names = [m.name for m in pkgutil.walk_packages(ripple_hpu.__path__, "ripple_hpu.")]
    for n in names:
        importlib.import_module(n)
    assert MODEL_MODULE in names, names
    return f"{len(names)} modules"


def g_config():
    from ripple_hpu.config import RicursiveConfig

    cfg = RicursiveConfig(**json.loads((FIXTURES / "config.json").read_text()))
    full = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    assert cfg.num_hidden_layers == 80 and full == list(range(4, 80, 5)), full
    assert cfg.sliding_window == 1024 and cfg.use_valres and cfg.use_gated_attention and cfg.gate_full_precision
    assert cfg.ffn_clamp_threshold == 20.0 and abs(cfg.embed_scale - 71.554) < 1e-3 and not cfg.tie_word_embeddings
    assert cfg.rope_parameters.get("rope_theta") == 10000.0, cfg.rope_parameters
    return "80 layers, 16 NoPE, window 1024"


def g_template():
    from transformers.utils.chat_template_utils import render_jinja_template
    from vllm.renderers.hf import _detect_content_format

    tmpl = (PORT / "serving" / "chat_template.jinja").read_text()
    fmt = _detect_content_format(tmpl, default="openai")
    assert fmt == "string", fmt
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "d",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }
    ]
    msgs = [
        {"role": "user", "content": "weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Need the tool.",
            "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
        },
        {"role": "tool", "content": "18C"},
    ]
    out, _ = render_jinja_template(conversations=[msgs], chat_template=tmpl, add_generation_prompt=True, tools=tools)
    s = out[0]
    assert "<tools>" in s and '"get_weather"' in s, s[:400]
    assert "<think>\nNeed the tool.\n</think>" in s and '<tool_call>\n{"name": "get_weather"' in s, s
    assert "<tool_response>\n18C\n</tool_response>" in s and s.endswith("<|im_start|>assistant\n"), s[-200:]
    return "content format string; tools, reasoning, tool call, tool response"


def g_parser():
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser.parser_manager import ParserManager

    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>")))
    cls = ParserManager.get_parser(tool_parser_name="ripple", reasoning_parser_name="ripple", enable_auto_tools=True)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    req = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "x"}], tool_choice="auto", tools=tools)
    text = (
        "<think>\nNeed the tool.\n</think>\n"
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
    )
    reasoning, content, calls = cls(tok).parse(text, req, enable_auto_tools=True)
    assert reasoning == "Need the tool." and not content, (reasoning, content)
    assert [(c.name, json.loads(c.arguments)) for c in calls] == [("get_weather", {"city": "Paris"})], calls
    so = cls(tok, req.tools).adjust_request(request=req).structured_outputs
    tag = json.loads(so.structural_tag) if so is not None and so.structural_tag else None
    assert tag and tag["format"]["type"] == "triggered_tags", so
    return "reasoning and tool call split; auto tool grammar"


def g_model_class():
    mod = importlib.import_module(MODEL_MODULE)
    assert hasattr(mod, MODEL_CLASS), MODEL_CLASS
    return f"{MODEL_MODULE}:{MODEL_CLASS}"


def g_hooks():
    import inspect

    from habana_frameworks.torch.hpu import graphs
    from vllm_gaudi.extension import ops
    from vllm_gaudi.extension.utils import ModuleFusedSDPA
    from vllm_gaudi.ops import hpu_fp8
    from vllm_gaudi.v1.worker import hpu_model_runner as hmr

    fwd = list(inspect.signature(ModuleFusedSDPA.forward).parameters)
    want = [
        "self",
        "query",
        "key",
        "value",
        "attn_mask",
        "dropout_p",
        "is_causal",
        "scale",
        "softmax_mode",
        "recompute_mode",
        "valid_sequence_lengths",
        "padding_side",
        "window_size",
        "sinks",
    ]
    assert fwd == want, fwd
    assert list(inspect.signature(ops._fsdpa_num_q_tiles).parameters) == ["attn_bias"]
    p = list(inspect.signature(hmr.HPUModelRunner._execute_model_generic).parameters)
    assert p[1:4] == ["token_ids", "position_ids", "attn_metadata"] and "warmup_mode" in p, p
    assert hasattr(hmr.HPUModelRunner, "_use_graphs")
    assert "htorch.hpu.wrap_in_hpu_graph(" in inspect.getsource(hmr._maybe_wrap_in_hpu_graph)
    src = inspect.getsource(graphs.wrap_in_hpu_graph)
    assert "cache = {}" in src and "def clear_cache" in src and "cache.clear()" in src, "wrap_in_hpu_graph cache"
    assert "cache[h] = CachedParams(" in inspect.getsource(graphs.wrapped_hpugraph_forward)
    assert isinstance(graphs.CachedParams.cache_hits, dict) and isinstance(graphs.CachedParams.skip_replay_cnt, dict)
    q = list(inspect.signature(ops.apply_block_fp8_linear_hpu).parameters)
    assert q == ["input", "layer", "block_size", "bias", "do_unpad", "force_channel_fp8"], q
    r = list(inspect.signature(ops.apply_fp8_linear_hpu).parameters)
    assert r[:3] == ["input", "weight", "weight_scale"] and "bias" in r, r
    assert "hpu_ops.apply_block_fp8_linear_hpu(" in inspect.getsource(hpu_fp8.Fp8LinearMethod.apply)
    from ripple_hpu import graph_budget as gb

    assert abs(gb.graph_gib(32, 1024, 384) - 0.789) < 0.005
    return "FusedSDPA, q tiles, graph cache, block-FP8 linear"


def g_runner_window():
    r = subprocess.run([sys.executable, str(HERE / "test_runner_window.py")], capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout + r.stderr).strip()[-600:]
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "ok"


def main() -> int:
    gate("patches", g_patches)
    gate("entry_point", g_entry_point)
    gate("register", g_register)
    gate("imports", g_imports)
    gate("config", g_config)
    gate("template", g_template)
    gate("parser", g_parser)
    gate("model_class", g_model_class)
    gate("hooks", g_hooks)
    gate("runner_window", g_runner_window)
    for name, ok, detail in results:
        print(f"[ripple-65b build_gates] {'PASS' if ok else 'FAIL'} {name}: {detail}")
    failed = [n for n, ok, _ in results if not ok]
    print(f"[ripple-65b build_gates] {len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
