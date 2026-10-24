# SPDX-License-Identifier: Apache-2.0
"""Reasoning and tool parsers for Inkling's typed-block output, registered as `inkling_py`.

vLLM's own inkling parsers wrap the Rust parser extension (`vllm._rust_tool_parser`), which a Gaudi build of
vLLM (VLLM_TARGET_DEVICE=empty) does not compile: the server starts and the first chat request fails. These
implement the same block format in Python.

Output format (every marker is one special token):

    <|message_model|><|content_thinking|>...reasoning...<|end_message|>
    <|message_model|><|content_text|>...visible text...<|end_message|>
    <|message_model|>get_weather<|content_invoke_tool_json|>{"name": "get_weather", "args": {...}}<|end_message|>

Blocks repeat in any order, and sampling may end a block with <|content_model_end_sampling|>. The prompt
ends with <|message_model|>, so the first generated block starts with a bare content marker; later blocks
carry a `<|message_model|>[recipient]` header.

The reasoning parser takes the thinking blocks and leaves the other markers in place for the tool parser,
which extracts tool blocks and strips every remaining marker. With --enable-auto-tool-choice the tool
parser runs on every chat request, so it is also what cleans the visible text.

The markers are special tokens, so the detokenizer must keep them: adjust_request sets
skip_special_tokens=False on every chat request (--override-generation-config cannot set it).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser
from vllm.tool_parsers.abstract_tool_parser import ToolParser

logger = init_logger(__name__)

MSG = "<|message_model|>"
THINK = "<|content_thinking|>"
TEXT = "<|content_text|>"
TOOL_JSON = "<|content_invoke_tool_json|>"
TOOL_TEXT = "<|content_invoke_tool_text|>"
END = "<|end_message|>"
END_SAMPLING = "<|content_model_end_sampling|>"
MARKERS = (MSG, THINK, TEXT, TOOL_JSON, TOOL_TEXT, END, END_SAMPLING)

# Optional block header: the message marker plus recipient text (never contains '<').
_HDR = r"(?:" + re.escape(MSG) + r"[^<]*)?"
_CLOSE = r"(?:" + re.escape(END) + r"|" + re.escape(END_SAMPLING) + r"|\Z)"
_THINK_BLOCK_RE = re.compile(_HDR + re.escape(THINK) + r"(.*?)" + _CLOSE, re.DOTALL)
_TOOL_BLOCK_RE = re.compile(_HDR + re.escape(TOOL_JSON) + r"(.*?)" + _CLOSE, re.DOTALL)
# Whole tool-block spans, markers included: kept intact for the tool parser or dropped, never half-stripped.
_TOOL_SPAN_RE = re.compile(_HDR + re.escape(TOOL_JSON) + r".*?" + _CLOSE, re.DOTALL)
# The message marker and its header first, so the recipient text goes with it.
_ALL_MARKERS_RE = re.compile(re.escape(MSG) + r"[^<]*|" + "|".join(re.escape(t) for t in MARKERS[1:]))
_CLOSE_RE = re.compile(re.escape(END) + r"|" + re.escape(END_SAMPLING))


def _strip_markers(s: str) -> str:
    return _ALL_MARKERS_RE.sub("", s)


def _clean_content(s: str, keep_tool_blocks: bool) -> str:
    """Strip markers from visible text. keep_tool_blocks leaves tool spans verbatim for the tool parser;
    otherwise they are removed whole, so tool syntax never leaks when no tool parsing runs."""
    if not keep_tool_blocks:
        return _strip_markers(_TOOL_SPAN_RE.sub("", s))
    parts, last = [], 0
    for m in _TOOL_SPAN_RE.finditer(s):
        parts.append(_strip_markers(s[last : m.start()]))
        parts.append(m.group(0))
        last = m.end()
    parts.append(_strip_markers(s[last:]))
    return "".join(parts)


def _inside_block(text: str, opener: str) -> bool:
    """True while the last block of type `opener` is still open."""
    o = text.rfind(opener)
    if o == -1:
        return False
    tail = text[o:]
    return END not in tail and END_SAMPLING not in tail


def _marker_hold(s: str) -> int:
    """Length of a trailing suffix of `s` that could start a marker (held back so partial markers never
    reach the client)."""
    longest = 0
    for tok in MARKERS:
        for k in range(min(len(s), len(tok) - 1), 0, -1):
            if tok.startswith(s[-k:]):
                longest = max(longest, k)
                break
    return longest


class InklingPyReasoningParser(ReasoningParser):
    """Reasoning = the thinking blocks; content = the rest, with other markers kept for the tool parser."""

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._streamed_reasoning_len = 0
        self._streamed_content_len = 0
        # Marker id -> kind, so is_reasoning_end scans raw ids instead of decoding a window of tokens.
        self._marker_kind: dict[int, str] = {}
        kinds = (
            ("msg", MSG),
            ("think", THINK),
            ("text", TEXT),
            ("tool", TOOL_JSON),
            ("tool", TOOL_TEXT),
            ("end", END),
            ("end", END_SAMPLING),
        )
        for kind, tok in kinds:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"inkling_py: marker {tok!r} is not a single token ({ids})")
            self._marker_kind[ids[0]] = kind

    def extract_reasoning(self, model_output: str, request) -> tuple[str | None, str | None]:
        """The server uses this content verbatim when no tool call is extracted, so it must be marker-free;
        tool spans stay only when tool extraction will run on them."""
        keep_tools = bool(getattr(request, "tools", None)) and getattr(request, "tool_choice", None) != "none"
        thinks = _THINK_BLOCK_RE.findall(model_output)
        reasoning = "\n".join(t.strip() for t in thinks if t.strip())
        content = _clean_content(_THINK_BLOCK_RE.sub("", model_output), keep_tool_blocks=keep_tools).strip()
        return (reasoning or None), (content or None)

    def adjust_request(self, request):
        request.skip_special_tokens = False
        # A request without tools defaults to tool_choice "none", which streams raw deltas past the tool
        # parser (markers would leak). tool_choice None routes the stream through it.
        if not getattr(request, "tools", None) and getattr(request, "tool_choice", None) == "none":
            request.tool_choice = None
        return request

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Has reasoning ended for what follows? Called on prompt ids (the prompt ends with an open
        <|message_model|> header: not ended) and on generated ids. The newest marker decides: an open header
        or an open thinking block means not ended; a closed block or a non-thinking block means ended.
        Scanning ids matters: a fixed decode window would miss the opener of a long thinking block and leak
        <|end_message|> into content."""
        for tid in reversed(list(input_ids)):
            kind = self._marker_kind.get(tid)
            if kind is not None:
                return kind not in ("msg", "think")
        return False

    def is_reasoning_end_streaming(self, input_ids: Sequence[int], delta_ids) -> bool:
        return self.is_reasoning_end(list(input_ids) + list(delta_ids))

    def extract_content_ids(self, input_ids: Sequence[int]) -> Sequence[int]:
        return input_ids

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Stream thinking bodies as reasoning; hold everything else.

        Content returned before is_reasoning_end first fires goes to the client, so it must be marker-free.
        Content returned in the delta where it fires becomes the tool parser's initial text, so it must keep
        its markers (the tool parser needs the block openers)."""
        if THINK not in current_text:
            if self.is_reasoning_end(list(current_token_ids)):
                new_c = current_text[self._streamed_content_len :]
                self._streamed_content_len = len(current_text)
                return DeltaMessage(content=new_c) if new_c else None
            if delta_text and not _ALL_MARKERS_RE.search(delta_text):
                self._streamed_content_len = len(current_text)
                return DeltaMessage(content=delta_text)
            return None
        body_start = current_text.rfind(THINK) + len(THINK)
        after = current_text[body_start:]
        close = _CLOSE_RE.search(after)
        if close is None:
            new = after[self._streamed_reasoning_len :]
            hold = _marker_hold(new)
            if hold:
                new = new[:-hold]
            if new:
                self._streamed_reasoning_len += len(new)
                return DeltaMessage(reasoning_content=new)
            return None
        # Thinking block closed: flush its body. Text after the close waits for the transition delta (an
        # END + <|message_model|> pair in one delta leaves an open header, so reasoning has not ended yet).
        body, rest = after[: close.start()], after[close.end() :]
        new_r = body[self._streamed_reasoning_len :]
        self._streamed_reasoning_len = len(body)
        new_c = ""
        if self.is_reasoning_end(list(current_token_ids)):
            new_c = rest[self._streamed_content_len :]
            self._streamed_content_len = len(rest)
        if new_r or new_c:
            return DeltaMessage(reasoning_content=new_r or None, content=new_c or None)
        return None


class InklingPyToolParser(ToolParser):
    """Extracts <|content_invoke_tool_json|> blocks as OpenAI tool calls and strips the remaining markers."""

    def __init__(self, tokenizer, tools=None):
        super().__init__(tokenizer, tools)
        self._emitted_calls = 0
        self._content_len = 0

    @staticmethod
    def _parse_payload(raw: str) -> FunctionCall | None:
        try:
            obj = json.loads(raw.strip())
        except json.JSONDecodeError:
            logger.warning("inkling_py: undecodable tool payload %r", raw[:200])
            return None
        name = obj.get("name")
        if not name:
            return None
        args = obj.get("args", obj.get("arguments", {}))
        return FunctionCall(name=name, arguments=json.dumps(args, ensure_ascii=False))

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        blocks = _TOOL_BLOCK_RE.findall(model_output)
        content = _strip_markers(_TOOL_BLOCK_RE.sub("", model_output)).strip()
        calls = [ToolCall(function=fc) for fc in map(self._parse_payload, blocks) if fc is not None]
        return ExtractedToolCallInformation(tools_called=bool(calls), tool_calls=calls, content=content or None)

    def _stream_visible(self, current_text: str, visible: str) -> DeltaMessage | None:
        safe = len(visible) - _marker_hold(current_text)
        if safe > self._content_len:
            chunk = visible[self._content_len : safe]
            self._content_len = safe
            return DeltaMessage(content=chunk) if chunk else None
        return None

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Stream visible text as it arrives (partial markers held back); emit each tool call once its block
        closes."""
        if not getattr(request, "tools", None):
            # No tools: never emit calls, drop any tool syntax, stream the cleaned text.
            if _inside_block(current_text, TOOL_JSON):
                return None
            return self._stream_visible(current_text, _clean_content(current_text, keep_tool_blocks=False))
        done_blocks = _TOOL_BLOCK_RE.findall(current_text)
        if _inside_block(current_text, TOOL_JSON):
            done_blocks = done_blocks[:-1]
        if len(done_blocks) > self._emitted_calls:
            deltas = []
            for i in range(self._emitted_calls, len(done_blocks)):
                fc = self._parse_payload(done_blocks[i])
                if fc is None:
                    continue
                # The chunk that opens a tool call must carry a string id (DeltaToolCall.id defaults to
                # None); strict OpenAI clients reject the stream otherwise.
                deltas.append(
                    DeltaToolCall(
                        index=i,
                        type="function",
                        id=make_tool_call_id(),
                        function=DeltaFunctionCall(name=fc.name, arguments=fc.arguments),
                    )
                )
            self._emitted_calls = len(done_blocks)
            return DeltaMessage(tool_calls=deltas) if deltas else None
        if _inside_block(current_text, TOOL_JSON):
            return None
        return self._stream_visible(current_text, _strip_markers(_TOOL_BLOCK_RE.sub("", current_text)))
