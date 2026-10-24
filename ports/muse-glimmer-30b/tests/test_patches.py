# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Muse-Glimmer-30B port patches (stdlib only).

The apply tests need the release sources: set GAUDI_FIXES_ROOT to a directory that holds vllm/ (vLLM
ced6857a) and vllm_gaudi/ (releases/v0.30.0, 058f107), for example a copy of the image's site-packages.
They patch a temporary copy, never the given tree, and are skipped without it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

PATCHES = Path(__file__).resolve().parent.parent / "patches"
FILES = {
    "output_gate_rank.py": ["vllm/model_executor/models/muse_glimmer.py"],
    "swa_window_blocks.py": ["vllm_gaudi/v1/worker/hpu_model_runner.py"],
    "forced_tool_grammar.py": [
        "vllm/tool_parsers/muse_glimmer_tool_parser.py",
        "vllm/entrypoints/openai/chat_completion/protocol.py",
    ],
}


def load(name):
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), PATCHES / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def release_root():
    root = os.environ.get("GAUDI_FIXES_ROOT")
    if not root or not all((Path(root) / rel).is_file() for rels in FILES.values() for rel in rels):
        pytest.skip("GAUDI_FIXES_ROOT with the v0.30.0 vLLM and vllm-gaudi sources not set")
    return Path(root)


def run(script, *args):
    return subprocess.run([sys.executable, str(PATCHES / script), *args], capture_output=True, text=True)


@pytest.fixture
def copy_root(tmp_path):
    src = release_root()
    for rels in FILES.values():
        for rel in rels:
            dst = tmp_path / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / rel, dst)
    return tmp_path


@pytest.mark.parametrize("script", sorted(FILES))
def test_apply_check_idempotent(copy_root, script):
    root = str(copy_root)
    assert run(script, "--check", "--root", root).returncode == 1  # anchors match, not applied
    first = run(script, "--root", root)
    assert first.returncode == 0, first.stderr
    assert run(script, "--check", "--root", root).returncode == 0  # applied
    assert run(script, "--root", root).returncode == 0  # second run is a no-op
    target = copy_root / FILES[script][0]
    compile(target.read_text(), str(target), "exec")


@pytest.mark.parametrize("script", sorted(FILES))
def test_drift_writes_nothing(copy_root, script):
    target = copy_root / FILES[script][0]
    text = target.read_text()
    for old, new in (
        ("self.attn(q, k, v)", "self.attn(q, k, v, None)"),
        ('self._get_model_type() == "gemma4"', 'self._get_model_type() == "gemma"'),
        ("            return request\n        return super().adjust_request(request)\n", "        return request\n"),
    ):
        text = text.replace(old, new)
    target.write_text(text)
    before = target.read_text()
    res = run(script, "--root", str(copy_root))
    assert res.returncode == 2
    assert target.read_text() == before


def test_output_gate_patch_text():
    mod = load("output_gate_rank.py")
    src = "x\n" + mod.OLD + "            pass\n"
    out = mod.patch(src)
    assert mod.MARKER in out and "attn_output.view(*hidden_states.shape[:-1], -1)" in out
    with pytest.raises(mod.Drift):
        mod.patch(out.replace(mod.OLD.splitlines()[0], "changed"))


def test_swa_patch_text():
    mod = load("swa_window_blocks.py")
    src = f"a = {mod.OLD}\nb = {mod.OLD} and x\n"
    out = mod.patch(src)
    assert out.count('in ("gemma4", "muse_glimmer")') == 2 and mod.MARKER in out
    with pytest.raises(mod.Drift):
        mod.patch(f"a = {mod.OLD}\n")


def helpers_namespace():
    """Run the helper block the patch appends, with a stand-in for vllm.sampling_params."""
    mod = load("forced_tool_grammar.py")
    fake = types.ModuleType("vllm.sampling_params")
    fake.StructuredOutputsParams = lambda **kw: kw
    saved = {k: sys.modules.get(k) for k in ("vllm", "vllm.sampling_params")}
    sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    sys.modules["vllm.sampling_params"] = fake
    try:
        ns: dict = {}
        exec(compile(mod.HELPERS, "helpers", "exec"), ns)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return ns


def tool(name, parameters):
    return types.SimpleNamespace(function=types.SimpleNamespace(name=name, parameters=parameters))


IDS = {"<|message|>": 1, "<|eom|>": 2, "<|eot|>": 3, "<|start|>": 4}
WEATHER = tool(
    "get_weather",
    {
        "type": "object",
        "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
        "required": ["city"],
    },
)


def tokens_in(node):
    if isinstance(node, dict):
        found = [node["token"]] if node.get("type") == "token" else []
        return found + [t for v in node.values() for t in tokens_in(v)]
    if isinstance(node, list):
        return [t for v in node for t in tokens_in(v)]
    return []


def test_structural_tag_named():
    build = helpers_namespace()["build_structural_tag"]
    tag = build([WEATHER, tool("other", None)], "get_weather", True, IDS)
    json.dumps(tag)
    turn = tag["format"]["elements"]
    assert len(turn) == 1  # named: one call, no repetition
    reasoning, call = turn[0]["elements"]
    assert reasoning["type"] == "optional"
    assert call["elements"][0]["value"] == " to=get_weather"
    params = call["elements"][3]["elements"]
    assert params[0]["type"] == "tag" and params[0]["begin"] == '<atem:parameter name="city">'
    assert params[1]["type"] == "optional"  # unit is not required
    assert [e["value"] for e in params[1]["content"]["content"]["elements"]] == ["c", "f"]
    # The end token is a request stop token: the grammar must not spell it out.
    assert IDS["<|eot|>"] not in tokens_in(tag)


def test_structural_tag_required_parallel():
    build = helpers_namespace()["build_structural_tag"]
    tag = build([WEATHER, tool("web_search", {"properties": {"q": {}}, "required": ["q"]})], None, True, IDS)
    turn = tag["format"]["elements"]
    assert turn[0]["elements"][1]["type"] == "or" and len(turn[0]["elements"][1]["elements"]) == 2
    assert turn[1]["type"] == "star"
    assert build([WEATHER], None, False, IDS)["format"]["elements"][-1]["type"] == "sequence"
    assert build([WEATHER], "missing", True, IDS) is None
