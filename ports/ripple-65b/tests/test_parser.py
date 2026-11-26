# SPDX-License-Identifier: Apache-2.0
"""ripple_hpu parser on vLLM v0.30.0: upstream's tests/parser/engine/test_ripple.py (PR #65745) with imports pointed
at the plugin, plus checks for this port:
  * the DelegatingParser patch (patches/parser_streams_all.py) is installed and a later turn's <think> is split out
    while streaming even though the prompt closed reasoning;
  * streaming a long tool call stays within a time budget (v0.30.0 has no arg_value_extender, so each delta
    re-scans the call so far; upstream PR #67146 is the fix on main);
  * tool_choice auto with non-strict tools gets the tool grammar (RIPPLE_CONSTRAIN_AUTO_TOOLS).

Needs the port image and vLLM's own test helpers (tests/parser/engine in the vLLM source tree); skips without
them. In the image: cd <vLLM source tree> && python3 -m pytest -q <this file>
"""

import json
import time

import pytest

pytest.importorskip("vllm")
pytest.importorskip("ripple_hpu.parser")
pytest.importorskip("tests.parser.engine.replay_harness")

from tests.parser.engine.conftest import make_mock_tokenizer  # noqa: E402
from tests.parser.engine.replay_harness import (  # noqa: E402
    _test_request,
    collect_output,
    replay_streaming,
)
from vllm.parser.parser_manager import ParserManager  # noqa: E402
from vllm.reasoning import ReasoningParserManager  # noqa: E402
from vllm.tool_parsers import ToolParserManager  # noqa: E402

import ripple_hpu  # noqa: E402
from ripple_hpu.parser import RippleEngineToolParser, RippleParser, RippleParserReasoningAdapter  # noqa: E402

ripple_hpu.register()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    },
]


def _call(city: str) -> str:
    body = {"name": "get_weather", "arguments": {"city": city}}
    return f"\n<tool_call>\n{json.dumps(body)}\n</tool_call>"


# (model output, reasoning, content, tool-call cities)
CASES = {
    "direct_answer": ("17*23 = 391.", "", "17*23 = 391.", []),
    "reasoning": (
        "<think>\nCompute.\n17*23=391.\n</think>\nThe answer is 391.",
        "Compute.\n17*23=391.",
        "The answer is 391.",
        [],
    ),
    "reasoning_then_tool": (
        "<think>\nNeed the tool.\n</think>" + _call("Paris"),
        "Need the tool.",
        "",
        ["Paris"],
    ),
    "content_then_tools": (
        "Checking both." + _call("Paris") + _call("Rome"),
        "",
        "Checking both.",
        ["Paris", "Rome"],
    ),
    "truncated_reasoning": ("<think>\nStill thinking", "Still thinking", "", []),
    # Text before a tool call keeps its whitespace in both parse paths.
    "extra_newline_before_tool_content": (
        "<think>\nNeed the tool.\n</think>\n\nChecking now." + _call("Paris"),
        "Need the tool.",
        "\nChecking now.",
        ["Paris"],
    ),
}


@pytest.fixture
def tokenizer():
    return make_mock_tokenizer({"<|im_start|>": 1, "<|im_end|>": 2})


@pytest.fixture
def parser_cls():
    return ParserManager.get_parser(
        tool_parser_name="ripple",
        reasoning_parser_name="ripple",
        enable_auto_tools=True,
    )


def test_parser_names_resolve_to_engine_adapters():
    tool_parser_cls = ToolParserManager.get_tool_parser("ripple")
    assert tool_parser_cls is RippleEngineToolParser
    assert tool_parser_cls.structural_tag_model == "hermes"
    assert ReasoningParserManager.get_reasoning_parser("ripple") is RippleParserReasoningAdapter


@pytest.mark.parametrize("case", CASES)
def test_parse(parser_cls, tokenizer, case):
    text, reasoning, content, cities = CASES[case]
    parsed_reasoning, parsed_content, tool_calls = parser_cls(tokenizer).parse(
        text, _test_request(TOOLS), enable_auto_tools=True
    )
    assert (parsed_reasoning or "") == reasoning
    assert (parsed_content or "") == content
    assert [json.loads(tc.arguments)["city"] for tc in tool_calls or []] == cities
    assert all(tc.name == "get_weather" for tc in tool_calls or [])


@pytest.mark.parametrize("chunk_size", [1, 3])
@pytest.mark.parametrize("case", CASES)
def test_streaming_matches_parse(parser_cls, tokenizer, case, chunk_size):
    text, reasoning, content, cities = CASES[case]
    tokens = [(ord(ch), ch) for ch in text]
    output = collect_output(
        replay_streaming(
            parser_cls(tokenizer),
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=TOOLS,
        )
    )
    assert output.reasoning == reasoning
    assert output.content == content
    assert [json.loads(tc["arguments"])["city"] for tc in output.tool_calls] == cities


@pytest.mark.parametrize("case", ["direct_answer", "reasoning"])
def test_streaming_without_tool_parser(tokenizer, case):
    # Without --enable-auto-tool-choice no tool parser is composed, so the
    # reasoning parser alone must still split out <think> while streaming.
    # The prompt [1, u, 2, 1] already counts as "reasoning ended" (thinking is
    # optional), which is the case the DelegatingParser patch exists for.
    parser_cls = ParserManager.get_parser(reasoning_parser_name="ripple")
    assert parser_cls is not None
    text, reasoning, content, _ = CASES[case]
    output = collect_output(
        replay_streaming(
            parser_cls(tokenizer),
            [(ord(ch), ch) for ch in text],
            chunk_size=1,
            finished_on_last=True,
            prompt_token_ids=[1, ord("u"), 2, 1],
        )
    )
    assert (output.reasoning, output.content) == (reasoning, content)


def test_structured_output_applies_from_first_token(tokenizer):
    # Thinking is optional, so a constrained response must not wait for a
    # "</think>" the model may never emit.
    parser = RippleParser(tokenizer)
    assert parser.is_reasoning_end([1, ord("u"), 2, 1])


# ------------------------------------------------------------------ backport checks
def test_streams_all_patch_installed():
    import inspect

    from vllm.parser import abstract_parser

    src = inspect.getsource(abstract_parser)
    assert "[ripple-65b:parser-streams-all]" in src, "run patches/parser_streams_all.py"
    assert src.count("_reasoning_engine_streams_all") == 4  # 1 assignment + 3 uses


def test_reasoning_adapter_opens_in_content(tokenizer):
    assert RippleParserReasoningAdapter(tokenizer).opens_reasoning_in_content is True


def _write_call(n_chars: int) -> tuple[str, str]:
    body = ('lorem \\"ipsum\\" dolor\\n' * (n_chars // 24 + 1))[:n_chars].rstrip("\\")
    args = '{"path": "a.txt", "content": "' + body + '"}'
    return '\n<tool_call>\n{"name": "write", "arguments": ' + args + "}\n</tool_call>", args


WRITE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}},
        },
    }
]


@pytest.mark.parametrize("n_chars", [4000, 16000])
def test_long_tool_call_streaming_cost(parser_cls, tokenizer, n_chars):
    """One char per delta (worst case). Reports the parser time; fails only when it is far
    out of range. ~4 chars/token, so 16000 chars ~ a 4K-token write."""
    text, args = _write_call(n_chars)
    t0 = time.perf_counter()
    output = collect_output(
        replay_streaming(
            parser_cls(tokenizer),
            [(ord(c) + 1000, c) for c in text],
            chunk_size=1,
            finished_on_last=True,
            tools=WRITE_TOOLS,
        )
    )
    dt = time.perf_counter() - t0
    print(f"\nlong tool call {n_chars} chars, 1 char/delta: {dt:.2f} s parser time")
    assert [json.loads(tc["arguments"]) for tc in output.tool_calls] == [json.loads(args)]
    assert dt < 120.0


# ------------------------------------------------------- auto tool choice grammar (ripple_hpu)
def _req(tool_choice, strict=None, **kw):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    fn = {
        "name": "list",
        "description": "List files.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    }
    if strict is not None:
        fn["strict"] = strict
    return ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": fn}],
        tool_choice=tool_choice,
        **kw,
    )


def _tag(parser_cls, tokenizer, request):
    out = parser_cls(tokenizer, request.tools).adjust_request(request=request)
    so = out.structured_outputs
    return json.loads(so.structural_tag) if so is not None and so.structural_tag else None


def test_auto_non_strict_gets_tool_grammar(parser_cls, tokenizer, monkeypatch):
    monkeypatch.setenv("RIPPLE_CONSTRAIN_AUTO_TOOLS", "1")
    tag = _tag(parser_cls, tokenizer, _req("auto"))
    assert tag is not None and tag["format"]["type"] == "triggered_tags"
    assert tag["format"]["triggers"] == ["<tool_call>"]
    begins = {t["begin"] for t in tag["format"]["tags"]}
    assert '<tool_call>\n{"name": "list", "arguments": ' in begins
    assert tag["format"]["tags"][0]["content"]["json_schema"]["required"] == ["path"]


def test_auto_non_strict_upstream_when_off(parser_cls, tokenizer, monkeypatch):
    monkeypatch.setenv("RIPPLE_CONSTRAIN_AUTO_TOOLS", "0")
    assert _tag(parser_cls, tokenizer, _req("auto")) is None


def test_auto_strict_false_keeps_free_arguments(parser_cls, tokenizer, monkeypatch):
    # strict=False on every tool: the caller opted out, so upstream's "no strict tool" rule holds.
    monkeypatch.setenv("RIPPLE_CONSTRAIN_AUTO_TOOLS", "1")
    assert _tag(parser_cls, tokenizer, _req("auto", strict=False)) is None


def test_auto_keeps_callers_response_format(parser_cls, tokenizer, monkeypatch):
    monkeypatch.setenv("RIPPLE_CONSTRAIN_AUTO_TOOLS", "1")
    fmt = {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}
    out = parser_cls(tokenizer, None).adjust_request(request=_req("auto", response_format=fmt))
    assert out.response_format is not None and out.response_format.type == "json_schema"
    assert out.structured_outputs is None or not out.structured_outputs.structural_tag


def test_required_and_none_unchanged(parser_cls, tokenizer, monkeypatch):
    monkeypatch.setenv("RIPPLE_CONSTRAIN_AUTO_TOOLS", "1")
    assert _tag(parser_cls, tokenizer, _req("required"))["format"]["type"] == "tags_with_separator"
    assert _tag(parser_cls, tokenizer, _req("none")) is None
