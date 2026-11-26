# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ripple reasoning and tool-call parser for vLLM v0.30.0 (ripple_hpu).

Taken from vLLM main 03bd3c6 (the PR #65745 merge): vllm/parser/ripple.py unchanged, plus
_JSON_WS, scan_json_object_end and extract_json_member_object from vllm/parser/utils.py
(absent from v0.30.0). v0.30.0 already has the parser engine and make_adapters; the adapters
and the tool parser class (structural tag "hermes", as upstream) are built at the bottom.
Registered under the name "ripple" by ripple_hpu.register(). One serving change from
upstream: tool_choice "auto" also gets the tool grammar when no tool is strict
(RIPPLE_CONSTRAIN_AUTO_TOOLS, RippleEngineToolParser).

Ripple decides per turn whether to think; the generation prompt does not open
``<think>``. Tool calls are JSON objects in Hermes-style tags::

    <think>
    ...reasoning...
    </think>
    ...content...
    <tool_call>
    {"name": "get_weather", "arguments": {"city": "Paris"}}
    </tool_call>

The markers are plain text, not special tokens.
"""

from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

_JSON_WS = " \t\r\n"


def scan_json_object_end(raw: str, start: int) -> int | None:
    """Return the end index (exclusive) of the JSON object starting at
    ``raw[start]`` (which must be ``{``), or ``None`` when it is still
    unterminated. String- and escape-aware so braces inside strings do not
    mislead the depth count."""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def extract_json_member_object(raw: str, key: str) -> str | None:
    """Extract the raw text span of the top-level object-valued member ``key``
    from a (possibly incomplete) ``{...}`` JSON wrapper.

    Returns the verbatim substring (prefix-stable as the input grows, which
    the engine's argument-delta diffing relies on), possibly an unterminated
    object prefix; ``None`` when the value has not started. Raises
    ``ValueError`` when the member is present but its value is not a JSON
    object. A verbatim substring (not re-serialized JSON) is required because
    the engine diffs successive converter outputs and expects each to extend
    the previous one; the scanner is string/escape-aware so a matching literal
    inside another value cannot mislead it, and it recovers a partial span when
    the wrapper is truncated (where ``json.loads`` would fail)."""
    depth = 0
    in_string = False
    escape = False
    string_start = -1
    last_string: str | None = None
    for i, ch in enumerate(raw):
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                if depth == 1:
                    last_string = raw[string_start + 1 : i]
            continue
        if ch == '"':
            in_string = True
            string_start = i
        elif ch == ":" and depth == 1 and last_string == key:
            value_start = i + 1
            while value_start < len(raw) and raw[value_start] in _JSON_WS:
                value_start += 1
            if value_start >= len(raw):
                return None
            if raw[value_start] != "{":
                raise ValueError(f"member {key!r} must be a JSON object")
            value_end = scan_json_object_end(raw, value_start)
            if value_end is None:
                return raw[value_start:]
            return raw[value_start:value_end]
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
    return None


THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"


def _framed(marker: str, before: bool = True, after: bool = True) -> tuple[str, ...]:
    """Spellings of ``marker`` with its optional framing newlines, longest first."""
    heads = ("\n", "") if before else ("",)
    tails = ("\n", "") if after else ("",)
    spellings = {head + marker + tail for head in heads for tail in tails}
    return tuple(sorted(spellings, key=len, reverse=True))


def _ripple_arg_converter(raw_args: str, partial: bool) -> str:
    args = extract_json_member_object(raw_args, "arguments")
    return args if args is not None else ("" if partial else "{}")


@functools.cache
def ripple_config() -> ParserEngineConfig:
    return ParserEngineConfig(
        name="ripple",
        initial_state=ParserState.CONTENT,
        # Reasoning is optional, so structured output applies from the first
        # token and the model answers directly.
        wait_for_reasoning=False,
        terminals={
            # The chat template frames each marker with newlines
            # ("<think>\n...\n</think>\n", "\n<tool_call>\n...\n</tool_call>"),
            # so they belong to the markers rather than the content.
            "THINK_START": _framed(THINK_START, before=False),
            "THINK_END": _framed(THINK_END),
            "TOOL_START": _framed(TOOL_CALL_START),
            "TOOL_END": _framed(TOOL_CALL_END, after=False),
        },
        transitions={
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            (ParserState.REASONING, "THINK_START"): Transition(ParserState.REASONING),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "THINK_END"): Transition(ParserState.CONTENT),
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
        },
        arg_converter=_ripple_arg_converter,
        # Streaming cannot retract text it already sent, so keep the
        # non-streaming content the same.
        strip_content_whitespace_with_tools=False,
    )


class RippleParser(ParserEngine):
    CONFIG_NAME = "ripple"

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        # The chat template has no thinking switch.
        kwargs.pop("chat_template_kwargs", None)
        kwargs.setdefault("parser_engine_config", ripple_config())
        super().__init__(tokenizer, tools, **kwargs)

    @property
    def reasoning_end_str(self) -> str:
        # The thinking budget matches this as token ids. The template's leading
        # newline merges into the previous token, so leave it out.
        return THINK_END + "\n"


# ---------------------------------------------------------------- v0.30.0 adapters (ripple_hpu)
from vllm.parser.engine.adapters import make_adapters  # noqa: E402

RippleParserReasoningAdapter, _RippleParserToolAdapter = make_adapters(RippleParser)


ENV_CONSTRAIN_AUTO_TOOLS = "RIPPLE_CONSTRAIN_AUTO_TOOLS"


def constrain_auto_tools() -> bool:
    return os.getenv(ENV_CONSTRAIN_AUTO_TOOLS, "1").strip().lower() in ("1", "true", "yes")


def _strict_unless_opted_out(tool):
    """A copy of `tool` marked strict, unless the caller set strict=False."""
    function = getattr(tool, "function", None)
    if function is not None:  # chat completions tool
        if function.strict is not None:
            return tool
        return tool.model_copy(update={"function": function.model_copy(update={"strict": True})})
    if getattr(tool, "type", None) == "function" and getattr(tool, "strict", None) is None:
        return tool.model_copy(update={"strict": True})  # Responses API function tool
    return tool


def _has_output_constraint(request) -> bool:
    if getattr(request, "structured_outputs", None) is not None:
        return True
    fmt = getattr(request, "response_format", None)
    if fmt is not None and getattr(fmt, "type", "text") != "text":
        return True
    text_fmt = getattr(getattr(request, "text", None), "format", None)  # Responses API
    return text_fmt is not None and getattr(text_fmt, "type", "text") != "text"


class RippleEngineToolParser(_RippleParserToolAdapter):  # type: ignore[valid-type, misc]
    """Upstream's tool parser plus one serving choice (ripple_hpu).

    Upstream applies the hermes structural tag to tool_choice "auto" only when some tool is
    strict. Agent clients send non-strict tools, and at temperature 1.0 Ripple sometimes calls
    a tool it was not offered (for example a ``glob`` tool its system prompt names): an
    unusable call. With RIPPLE_CONSTRAIN_AUTO_TOOLS=1 (default) an auto request is built as if
    its tools were strict, so a call names an offered tool and its arguments follow that tool's
    schema; text before a call stays free, and the grammar starts after reasoning, as for strict
    tools. Tools the caller marked strict=False keep free-form arguments. Requests that already
    carry an output constraint (response_format, structured outputs) are left as upstream leaves
    them. 0 restores upstream.
    """

    structural_tag_model = "hermes"

    def get_structural_tag(self, request, *, reasoning: bool = False):
        tag = super().get_structural_tag(request, reasoning=reasoning)
        if (
            tag is not None
            or not constrain_auto_tools()
            or request.tool_choice != "auto"
            or not request.tools
            or _has_output_constraint(request)
        ):
            return tag
        from vllm import envs
        from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

        if not envs.VLLM_ENFORCE_STRICT_TOOL_CALLING:
            return None
        return get_model_structural_tag(
            model=self.structural_tag_model,
            tools=[_strict_unless_opted_out(t) for t in request.tools],
            tool_choice="auto",
            reasoning=reasoning,
        )


def _opens_reasoning_in_content(self) -> bool:
    """Whether the grammar can open reasoning after content has started (PR #65745,
    vllm/parser/engine/adapters.py). Read by the patched DelegatingParser
    (patches/parser_streams_all.py)."""
    start = self._parser_engine.parser_engine_config.transitions.get((ParserState.CONTENT, "THINK_START"))
    return start is not None and start.next_state is ParserState.REASONING


RippleParserReasoningAdapter.opens_reasoning_in_content = property(_opens_reasoning_in_content)

__all__ = ["RippleParser", "RippleParserReasoningAdapter", "RippleEngineToolParser", "ripple_config"]
