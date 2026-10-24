# SPDX-License-Identifier: Apache-2.0
"""inkling_py parsers with the real tokenizer and vLLM's DelegatingParser, token by token, as the server
streams. No block marker may reach content or reasoning; a long thinking block (longer than any fixed decode
window) must not delay the switch to content. Needs vLLM and INKLING_MODEL_DIR (for the tokenizer); skips
without them."""

import os

import pytest

pytest.importorskip("vllm")
MODEL_DIR = os.environ.get("INKLING_MODEL_DIR")
pytestmark = pytest.mark.skipif(not MODEL_DIR, reason="INKLING_MODEL_DIR not set")

MSG = "<|message_model|>"
THINK = "<|content_thinking|>"
TEXT = "<|content_text|>"
TOOL = "<|content_invoke_tool_json|>"
END = "<|end_message|>"
MARKERS = (MSG, THINK, TEXT, TOOL, END, "<|content_model_end_sampling|>")
LONG_THINK = (
    "The user asks a general question about supercomputers: what defines them and what they are used for. "
    "No tools are needed; it is a standard topic I can answer directly. Points to cover: definition, massive "
    "parallelism, specialized architecture, purpose, and ranking on the TOP500 list. I will give a clear, "
    "structured answer without invoking tools, since nothing more is needed here."
)
ANSWER = "A supercomputer is defined less by a single spec and more by scale."
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
LONG = THINK + LONG_THINK + END + MSG + TEXT + ANSWER + END
TOOL_OUT = (
    THINK + LONG_THINK + END + MSG + "get_weather" + TOOL + '{"name": "get_weather", "args": {"city": "Paris"}}' + END
)


@pytest.fixture(scope="module")
def tok():
    import inkling_hpu
    from transformers import AutoTokenizer

    inkling_hpu.register()
    return AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)


def _stream(tok, output, tools=None, tool_choice=None, batch_end_msg=False):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser.parser_manager import ParserManager

    parser_cls = ParserManager.get_parser(
        tool_parser_name="inkling_py", reasoning_parser_name="inkling_py", enable_auto_tools=True
    )
    kwargs = {"messages": [{"role": "user", "content": "q"}], "model": "m"}
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    request = ChatCompletionRequest(**kwargs)
    parser = parser_cls(tok, tools=getattr(request, "tools", None))
    request = parser.reasoning_parser.adjust_request(request)

    prompt_ids = tok(MSG + TEXT + "What is a supercomputer?" + END + MSG, add_special_tokens=False).input_ids
    ids = tok(output, add_special_tokens=False).input_ids
    end_id = tok(END, add_special_tokens=False).input_ids[0]
    msg_id = tok(MSG, add_special_tokens=False).input_ids[0]
    chunks, i = [], 0
    while i < len(ids):
        step = 2 if batch_end_msg and i + 1 < len(ids) and ids[i] == end_id and ids[i + 1] == msg_id else 1
        chunks.append(ids[i : i + step])
        i += step
    content, reasoning, calls = [], [], []
    for ci, chunk in enumerate(chunks):
        dm = parser.parse_delta(
            tok.decode(chunk, skip_special_tokens=False),
            list(chunk),
            request,
            prompt_token_ids=prompt_ids if ci == 0 else None,
            finished=ci == len(chunks) - 1,
        )
        if dm is None:
            continue
        content.append(dm.content or "")
        reasoning.append(getattr(dm, "reasoning", None) or getattr(dm, "reasoning_content", None) or "")
        calls.extend(dm.tool_calls or [])
    return "".join(content), "".join(reasoning), calls


@pytest.mark.parametrize(
    "name,output,kwargs,expect_tool",
    [
        ("long-think", LONG, {}, False),
        ("long-think/batched-end-msg", LONG, {"batch_end_msg": True}, False),
        ("short-think", THINK + "Simple." + END + MSG + TEXT + ANSWER + END, {}, False),
        ("no-think", TEXT + ANSWER + END, {}, False),
        ("tool-call", TOOL_OUT, {"tools": TOOLS}, True),
        ("tool-choice-none", LONG, {"tool_choice": "none"}, False),
        ("tools-defined/text-answer", LONG, {"tools": TOOLS}, False),
    ],
)
def test_streaming_has_no_marker_leaks(tok, name, output, kwargs, expect_tool):
    content, reasoning, calls = _stream(tok, output, **kwargs)
    assert not [m for m in MARKERS if m in content or m in reasoning], (content, reasoning)
    if expect_tool:
        assert [c for c in calls if c.function and c.function.name == "get_weather" and c.id]
    else:
        assert not calls
        assert ANSWER in content
    if THINK in output:
        assert "TOP500" in reasoning or "Simple" in reasoning


def test_non_streaming_split(tok):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    from inkling_hpu.parsers import InklingPyReasoningParser, InklingPyToolParser

    req = ChatCompletionRequest(messages=[{"role": "user", "content": "q"}], model="m", tools=TOOLS)
    reasoning, content = InklingPyReasoningParser(tok).extract_reasoning(TOOL_OUT, req)
    assert "TOP500" in reasoning and THINK not in (content or "")
    info = InklingPyToolParser(tok, TOOLS).extract_tool_calls(content, req)
    assert info.tools_called and info.tool_calls[0].function.name == "get_weather"
    assert '"Paris"' in info.tool_calls[0].function.arguments
    assert not info.content
