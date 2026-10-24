#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build gates for the Beam port image: run by install.sh after every install step.

No device, network or checkpoint is needed: the checkpoint's config.json and chat template are
fixtures next to this file. Every gate prints one PASS or FAIL line; any failure exits 1.

  fixes          fixes/apply_all.py --check --profile v0.30.0 reports every profile fix applied
  patches        the port's three source patches report "applied" (--check exits 0)
  entry_point    beam_hpu:register is in the vllm.general_plugins entry points
  register       vLLM's plugin loader registers BeamForCausalLM, BeamConfig and the `beam` parsers
  imports        every module of the package imports
  config_quant   the checkpoint config parses into BeamConfig with compressed-tensors FP8 per-channel
                 weights and dynamic per-token activations
  content_fmt    the checkpoint chat template is detected as content format "string" (serving with
                 --chat-template-content-format openai turns every chat request into an HTTP 400)
  template       the template renders a user turn with reasoning_effort "low" and with tools
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import pkgutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = HERE.parent
COOKBOOK = PORT.parent.parent
FIXTURES = HERE / "fixtures"
ENTRY_POINT = "beam_hpu:register"
ARCH = "BeamForCausalLM"
PATCHES = ("vllm_pr_62052.py", "beam_sliding_window.py", "grammar_device_mask.py")

results: list[tuple[str, bool, str]] = []


def gate(fn):
    def run():
        try:
            results.append((fn.__name__, True, fn() or ""))
        except Exception as e:  # noqa: BLE001  (every failure becomes a FAIL line)
            results.append((fn.__name__, False, f"{type(e).__name__}: {e}"))

    return run


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], capture_output=True, text=True)


@gate
def fixes():
    r = _run(str(COOKBOOK / "fixes" / "apply_all.py"), "--check", "--profile", "v0.30.0")
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    return "v0.30.0 profile applied"


@gate
def patches():
    for name in PATCHES:
        r = _run(str(PORT / "patches" / name), "--check")
        assert r.returncode == 0, f"{name}: {(r.stdout + r.stderr).strip()}"
    return ", ".join(PATCHES)


@gate
def entry_point():
    eps = [e.value for e in importlib.metadata.entry_points(group="vllm.general_plugins")]
    assert ENTRY_POINT in eps, eps
    return ENTRY_POINT


@gate
def register():
    from vllm import ModelRegistry
    from vllm.plugins import load_general_plugins
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    load_general_plugins()
    assert ARCH in ModelRegistry.get_supported_archs(), "architecture not registered"
    assert _CONFIG_REGISTRY.get("beam").__name__ == "BeamConfig", _CONFIG_REGISTRY.get("beam")
    ReasoningParserManager.get_reasoning_parser("beam")
    ToolParserManager.get_tool_parser("beam")
    return "architecture, config, parsers"


@gate
def imports():
    import beam_hpu

    names = [m.name for m in pkgutil.walk_packages(beam_hpu.__path__, "beam_hpu.")]
    for n in names:
        importlib.import_module(n)
    assert "beam_hpu.vllm_model" in names, names
    return f"{len(names)} modules"


@gate
def config_quant():
    from beam_hpu.config import BeamConfig

    raw = json.loads((FIXTURES / "config.json").read_text())
    cfg = BeamConfig(**raw)
    assert cfg.num_hidden_layers == 52 and cfg.nope_layer_ids == [5, 11, 17, 23, 29, 35, 41, 47, 51], cfg.nope_layer_ids
    q = raw["quantization_config"]
    g = q["config_groups"]["FP8_DYNAMIC"]
    assert q["quant_method"] == "compressed-tensors"
    act = g["input_activations"]
    assert g["weights"]["strategy"] == "channel" and act["strategy"] == "token" and act["dynamic"], g
    return "BeamConfig, compressed-tensors FP8 channel/token"


@gate
def content_fmt():
    from vllm.renderers.hf import _detect_content_format

    fmt = _detect_content_format((FIXTURES / "chat_template.jinja").read_text(), default="openai")
    assert fmt == "string", fmt
    return fmt


@gate
def template():
    from transformers.utils.chat_template_utils import render_jinja_template

    tmpl = (FIXTURES / "chat_template.jinja").read_text()
    msgs = [{"role": "user", "content": "hi"}]
    params = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
    tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": params}}]
    out, _ = render_jinja_template(
        conversations=[msgs], chat_template=tmpl, add_generation_prompt=True, reasoning_effort="low", tools=tools
    )
    assert "Reasoning effort: 32" in out[0] and "type f = " in out[0], out[0][:400]
    return "effort and tools render"


def main() -> int:
    for g in (fixes, patches, entry_point, register, imports, config_quant, content_fmt, template):
        g()
    for name, ok, detail in results:
        print(f"[beam-1-0 build gate] {'PASS' if ok else 'FAIL'} {name}: {detail}")
    failed = [n for n, ok, _ in results if not ok]
    print(f"[beam-1-0 build gate] {len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
