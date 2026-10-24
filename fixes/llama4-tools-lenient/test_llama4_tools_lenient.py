# SPDX-License-Identifier: Apache-2.0
"""CPU tests for llama4-tools-lenient: the pure-Python parser, the call policy and the template.

The parser tests are stdlib only (the vLLM half of the plugin is skipped when vLLM is absent);
the template test needs jinja2. Run with: pytest fixes/llama4-tools-lenient
"""

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plugin = _load("llama4_tools_lenient_under_test", HERE / "llama4_tools_lenient.py")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {
                "type": "object",
                "properties": {"filePath": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["filePath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "parameters": {
                "type": "object",
                "properties": {"filePath": {"type": "string"}, "replaceAll": {"type": "boolean"}},
                "required": ["filePath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todowrite",
            "parameters": {"type": "object", "properties": {"todos": {"type": "array"}}, "required": ["todos"]},
        },
    },
]


def test_tag_syntax():
    content, calls = plugin.parse_output('<function=read>{"filePath": "a.py"}</function>', TOOLS, "all")
    assert calls == [("read", {"filePath": "a.py"})] and content == ""


def test_pythonic_list_with_prose_before():
    content, calls = plugin.parse_output('Let me look.\n[read(filePath="a.py", limit="20")]', TOOLS, "all")
    assert content.strip() == "Let me look."
    assert calls == [("read", {"filePath": "a.py", "limit": 20})]


def test_python_markers_and_json():
    _, calls = plugin.parse_output('<|python_start|>read(filePath="b.py")<|python_end|>', TOOLS, "all")
    assert calls == [("read", {"filePath": "b.py"})]
    _, calls = plugin.parse_output('{"name": "read", "parameters": {"filePath": "c.py"}}', TOOLS, "all")
    assert calls == [("read", {"filePath": "c.py"})]


def test_schema_repair():
    text = '[todowrite(todos="[{\'content\': \'x\'}]"), edit(filePath="a", replaceAll="true")]'
    _, calls = plugin.parse_output(text, TOOLS, "all")
    assert calls[0] == ("todowrite", {"todos": [{"content": "x"}]})
    assert calls[1] == ("edit", {"filePath": "a", "replaceAll": True})


def test_unknown_tool_and_prose_are_content():
    content, calls = plugin.parse_output("nope(x=1) is not a known tool", TOOLS, "all")
    assert calls == [] and "nope" in content
    content, calls = plugin.parse_output("Just an answer.", TOOLS, "all")
    assert calls == [] and content == "Just an answer."


def test_independent_policy():
    reads = [("read", {"filePath": "a"}), ("read", {"filePath": "b"}), ("edit", {"filePath": "a"})]
    assert plugin.apply_policy(reads, "independent") == reads[:2]
    edits = [("edit", {"filePath": "a"}), ("edit", {"filePath": "b"}), ("edit", {"filePath": "a"})]
    assert plugin.apply_policy(edits, "independent") == edits[:2]
    assert plugin.apply_policy(reads, "first") == reads[:1]
    assert plugin.apply_policy(reads, "all") == reads


def test_template_renders_history_as_python_literals():
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda v, indent=None: json.dumps(v, indent=indent)
    tmpl = env.from_string((HERE / "llama4-tools.jinja").read_text())
    messages = [
        {"role": "user", "content": "read a.py"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "read", "arguments": {"filePath": 'a "b".py', "limit": 5}}}],
        },
        {"role": "tool", "content": "line1\nline2"},
    ]
    out = tmpl.render(messages=messages, tools=TOOLS, bos_token="<|begin_of_text|>", add_generation_prompt=True)
    assert out.startswith("<|begin_of_text|>")
    assert '[read(filePath="a \\"b\\".py", limit=5)]<|eom|>' in out
    assert "<|header_start|>ipython<|header_end|>\n\nline1\nline2<|eom|>" in out
    assert out.endswith("<|header_start|>assistant<|header_end|>\n\n")
