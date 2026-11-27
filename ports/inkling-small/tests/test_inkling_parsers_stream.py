# SPDX-License-Identifier: Apache-2.0
"""inkling_py parsers through vLLM's DelegatingParser on CPU: the streamed deltas, joined, must equal the
non-streaming parse. A stand-in tokenizer makes each marker one token and every other character one token, and
the stream feeds one token per delta (every split point), or <|end_message|><|message_model|> as one delta. Needs
vLLM v0.30.0 (skips without it); no model files.
"""

import json

import pytest

pytest.importorskip("vllm")

MSG = "<|message_model|>"
THINK = "<|content_thinking|>"
TEXT = "<|content_text|>"
TOOL = "<|content_invoke_tool_json|>"
TOOL_TEXT = "<|content_invoke_tool_text|>"
END = "<|end_message|>"
END_SAMPLING = "<|content_model_end_sampling|>"
MARKERS = (MSG, THINK, TEXT, TOOL, TOOL_TEXT, END, END_SAMPLING)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "d",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]
WEATHER = '{"name": "get_weather", "args": {"city": "Paris"}}'
CASES = {
    "one-block": THINK + "Plan the answer." + END + MSG + TEXT + "The answer." + END,
    "two-blocks": THINK + "First, the plan." + END + MSG + THINK + "Then a check." + END + MSG + TEXT + "Done." + END,
    "two-blocks/text-between": THINK
    + "First."
    + END
    + MSG
    + TEXT
    + "Partly."
    + END
    + MSG
    + THINK
    + "More."
    + END_SAMPLING
    + MSG
    + TEXT
    + "Done."
    + END,
    "tool-call": THINK + "Look it up." + END + MSG + "get_weather" + TOOL + WEATHER + END,
    "truncated-tool-call": THINK + "Look it up." + END + MSG + "get_weather" + TOOL + WEATHER[:30],
}


class CharTokenizer:
    """Each marker is one token (ids 1..7); every other character is one token."""

    def __init__(self):
        self._ids = {m: i for i, m in enumerate(MARKERS, start=1)}
        self._markers = {i: m for m, i in self._ids.items()}

    def get_vocab(self):
        return dict(self._ids)

    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            marker = next((m for m in MARKERS if text.startswith(m, i)), None)
            ids.append(self._ids[marker] if marker else 1000 + ord(text[i]))
            i += len(marker) if marker else 1
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._markers.get(t) or chr(t - 1000) for t in ids)


@pytest.fixture(scope="module")
def parser_cls():
    from vllm.parser.parser_manager import ParserManager
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    ToolParserManager.register_lazy_module("inkling_py", "inkling_hpu.parsers", "InklingPyToolParser")
    ReasoningParserManager.register_lazy_module("inkling_py", "inkling_hpu.parsers", "InklingPyReasoningParser")
    return ParserManager.get_parser(
        tool_parser_name="inkling_py", reasoning_parser_name="inkling_py", enable_auto_tools=True
    )


def _new(parser_cls, tools):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    tok = CharTokenizer()
    kwargs = {"messages": [{"role": "user", "content": "q"}], "model": "m"}
    if tools is not None:
        kwargs["tools"] = tools
    request = ChatCompletionRequest(**kwargs)
    parser = parser_cls(tok, tools=request.tools)
    return tok, parser, parser.adjust_request(request)


def _streamed(parser_cls, output, tools, batch_end_msg):
    tok, parser, request = _new(parser_cls, tools)
    prompt = tok.encode(MSG + TEXT + "q" + END + MSG)
    ids = tok.encode(output)
    end, msg = tok.encode(END + MSG)
    chunks, i = [], 0
    while i < len(ids):
        step = 2 if batch_end_msg and ids[i : i + 2] == [end, msg] else 1
        chunks.append(ids[i : i + step])
        i += step
    reasoning, content, calls = [], [], {}
    for i, chunk in enumerate(chunks):
        dm = parser.parse_delta(
            tok.decode(chunk),
            chunk,
            request,
            prompt_token_ids=prompt if i == 0 else None,
            finished=i == len(chunks) - 1,
        )
        if dm is None:
            continue
        # the field v0.30.0 declares; an undeclared one (OpenAIBaseModel allows extras) streams under another key
        assert "reasoning_content" not in dm.model_dump(), dm
        reasoning.append(dm.reasoning or "")
        content.append(dm.content or "")
        for tc in dm.tool_calls:
            name, args = calls.get(tc.index, ("", ""))
            calls[tc.index] = (name + (tc.function.name or ""), args + (tc.function.arguments or ""))
    joined = [(name, json.loads(args)) for name, args in map(calls.get, sorted(calls))]
    return "".join(reasoning) or None, "".join(content) or None, joined


def _whole(parser_cls, output, tools):
    _, parser, request = _new(parser_cls, tools)
    reasoning, content, calls = parser.parse(output, request, enable_auto_tools=True)
    return reasoning, content, [(c.name, json.loads(c.arguments)) for c in calls or []]


# batch-end-msg: <|end_message|><|message_model|> in one delta leaves an open header, so the next thinking block
# is still the reasoning parser's
@pytest.mark.parametrize("batch_end_msg", [False, True], ids=["token", "batch-end-msg"])
@pytest.mark.parametrize("tools", [TOOLS, None], ids=["tools", "no-tools"])
@pytest.mark.parametrize("name", CASES)
def test_stream_joins_to_the_non_streaming_parse(parser_cls, name, tools, batch_end_msg):
    output = CASES[name]
    whole = _whole(parser_cls, output, tools)
    assert _streamed(parser_cls, output, tools, batch_end_msg) == whole
    reasoning, content, calls = whole
    assert not [m for m in MARKERS if m in (reasoning or "") or m in (content or "")], whole
    assert calls == ([("get_weather", {"city": "Paris"})] if name == "tool-call" and tools else [])


def test_without_a_reasoning_parser_thinking_streams_as_content():
    """vLLM sets skip_reasoning_parsing on the tool parser when no reasoning parser is configured."""
    from vllm.parser.parser_manager import ParserManager
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    ToolParserManager.register_lazy_module("inkling_py", "inkling_hpu.parsers", "InklingPyToolParser")
    tool_only = ParserManager.get_parser(tool_parser_name="inkling_py", enable_auto_tools=True)
    reasoning, content, calls = _streamed(tool_only, CASES["two-blocks"], TOOLS, False)
    assert (reasoning, content, calls) == (None, "First, the plan.Then a check.Done.", [])


def test_every_thinking_block_is_reasoning(parser_cls):
    assert _whole(parser_cls, CASES["two-blocks"], TOOLS)[:2] == ("First, the plan.Then a check.", "Done.")
    assert _whole(parser_cls, CASES["two-blocks/text-between"], None)[:2] == ("First.More.", "Partly.Done.")
