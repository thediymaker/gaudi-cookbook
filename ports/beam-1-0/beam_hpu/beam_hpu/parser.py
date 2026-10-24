# SPDX-License-Identifier: Apache-2.0
"""The `beam` reasoning and tool parsers and the Beam response grammar, for vLLM v0.30.

Backport of vLLM PR #62052 (merge 153a1716, `vllm/parser/beam.py`). The PR's parsers sit
on the response-template framework (`vllm/parser/response_template.py`,
`vllm/parser/response_template_grammar.py`, `vllm/parser/chat_parsing/`), which v0.30
does not have; those files are vendored verbatim (vendor_pr/, installed in-tree by
patches/install_pr_files.py). This module is the PR's `beam.py` rewritten against
v0.30's request objects and helpers:

Output format (checkpoint `response_template`, tokenizer_config.json), after the
generation prompt `<|start|>assistant`:

    <|channel|>analysis<|message|>REASONING<|end|><|start|>assistant
    then either  <|channel|>final<|message|>ANSWER<|return|>
    or calls     ' to=NAME'<|channel|>analysis <|constrain|>xml<|message|>BODY
                 joined by <|end|><|start|>assistant, the last ending <|call|>
    BODY = <rfl:parameter name="k">\\nVALUE\\n</rfl:parameter> ...

Parsing (reasoning -> `reasoning`, final -> `content`, calls -> `tool_calls` with
arguments typed by the tool's JSON schema) is the vendored ResponseTemplateParser,
streaming and non-streaming. Non-streaming, a tool region is one complete call; streaming,
calls are sent while they are written (_BeamResponseStream, below).

Grammar (`_template_tag`): every Chat Completions turn is constrained by an xgrammar
structural tag derived from the response template, from the first generated token:
non-empty reasoning first, then an answer or calls; call names limited to the request's
tools for tool_choice required/named; each tool's arguments held to its JSON schema where
the template can express it (types, enums, required keys, declared or sorted key order);
for tool_choice "none" no call can be written; response_format json_schema/json_object/
regex/choice/grammar goes inside the final channel. Every closer at the end of the turn is
optional: <|return|> and <|call|> are the request's stop tokens, which xgrammar accepts
only once the grammar is complete and never as grammar text (a mandatory trailing stop
token makes the matcher reject it, FINISHED_ERROR, HTTP 500). The PR's grammar already
does this ("EOS can terminate a call without writing its closer").

v0.30 differences handled here:
  - no `set_prompt_token_ids` call before adjust_request: the prompt of a Chat
    Completions request ends in the generation prompt `<|start|>assistant` unless
    `continue_final_message` is set, so that is the prefix; with
    `continue_final_message` (and Responses partial completion) the turn is left
    unconstrained, and forced tool choice is rejected with HTTP 400;
  - no `--tool-strict-level`: tools are taken as given (`strict` per tool);
  - Responses API: v0.30 does not pass `reasoning_ended` for parser grammars, so the
    engine's reasoning gate holds the grammar until the reasoning closes; the tag for
    Responses requests is the after-reasoning form, and `text.format` moves into it;
  - xgrammar 0.2.3 ignores `excludes` on json_schema formats (0.2.8 honours it), so a
    JSON-typed parameter value may still spell `<|end|>` inside a JSON string;
  - the prefix is decoded from the last `<|start|>assistant` of the prompt, not the whole
    prompt (the template parser truncates to the last start anchor anyway);
  - `\\s` in negated regex classes becomes ASCII whitespace (_ascii_space_classes), or
    xgrammar 0.2.3 logs ~60 "clamping to 127" warnings per compile of an auto-tools tag.

Fixes on top of upstream (all three also apply to upstream's code):
  - local `$ref`s in tool parameter schemas are inlined before parsing and grammar
    derivation (resolve_tool_refs): upstream returned `$defs`-declared object arguments as
    JSON strings and left those tools' arguments unconstrained;
  - count_reasoning_tokens is incremental (_IncrementalReasoningCounter, same counts):
    upstream's counter re-walks the whole output on every call, and v0.30 streaming calls
    it after every delta (quadratic; a 32K-token reasoning stream cost minutes of CPU on
    the API server's event loop);
  - tool calls stream (_BeamResponseStream): upstream sends a call only when its region
    closes, so a client saw nothing while Beam wrote a long call (a `write` of a whole
    file: minutes, past the idle timeouts of agent clients and proxies). A call is announced
    (index, id, name) in the delta where its region opens. When the request's grammar
    holds the tool's arguments to its schema (strict arguments: each key once, so the
    parser's dict order is the written order), each parameter follows as it is written:
    string values as they arrive (JSON-escaped; a possible `\\n</rfl:parameter>` is held
    until it resolves), typed values (number, boolean, null, object, array) whole at their
    closer; the rest of the arguments at the close. Without that grammar the arguments
    come whole at the close. Per call the fragments join to exactly the non-streaming
    arguments, except for a call cut off inside a streamed value (max_tokens): the parser
    drops that parameter at the close, but its text was already sent.

Env: VLLM_BEAM_ALWAYS_CONSTRAIN_OUTPUT (default 1, as upstream) constrains turns that ask
for no tool or output constraint; 0 leaves those free (forced tool choice, tool_choice
none with tools, strict tools and response_format still get the grammar).
VLLM_BEAM_STREAM_TOOL_ARGS (default 1) streams tool calls as above; 0 restores upstream's
streaming (each call whole at its close), read per request.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.parser import response_template as _response_template
from vllm.parser.chat_parsing import ResponseParser
from vllm.parser.chat_parsing.response_parser import _coerce, _schema_types
from vllm.parser.chat_parsing.response_templates import ResponseTemplate, ResponseTemplateField
from vllm.parser.response_template import (
    CONTENT_FIELD,
    THINKING_FIELD,
    TOOL_FIELD,
    ResponseTemplateParser,
    ResponseTemplateReasoningParser,
    ResponseTemplateToolParser,
    _as_tool_dicts,
    _decode,
    _is_named_tool_choice,
    _ResponseStream,
    _to_tool_calls,
    _tool_is_strict,
    _tool_parsing_enabled,
)
from vllm.parser.response_template_grammar import (
    _arguments_format,
    _openers,
    answer_rest_structural_tag,
    message_start_anchors,
    tool_call_structural_tag,
)
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

logger = init_logger("vllm.beam_hpu.parser")

NAME = "beam"
START_TOKEN = "<|start|>"
ANCHOR = "<|start|>assistant"  # checkpoint start_anchor_pattern
ENV_ALWAYS_CONSTRAIN = "VLLM_BEAM_ALWAYS_CONSTRAIN_OUTPUT"
ENV_STREAM_TOOL_ARGS = "VLLM_BEAM_STREAM_TOOL_ARGS"
# Request attribute set by BeamParser._template_tag: the tools whose arguments the request's
# grammar holds to their schema (strict arguments), read by the stream.
STRICT_TOOLS_ATTR = "_beam_strict_tools"


def always_constrain_output() -> bool:
    return os.getenv(ENV_ALWAYS_CONSTRAIN, "1").strip().lower() in ("1", "true", "yes")


def stream_tool_arguments() -> bool:
    return os.getenv(ENV_STREAM_TOOL_ARGS, "1").strip().lower() in ("1", "true", "yes")


def _enforce_strict_tools() -> bool:
    from vllm import envs

    return bool(getattr(envs, "VLLM_ENFORCE_STRICT_TOOL_CALLING", True))


def _named_tool(request: ChatCompletionRequest | ResponsesRequest) -> str:
    choice = request.tool_choice
    if isinstance(choice, ChatCompletionNamedToolChoiceParam):
        return choice.function.name
    return choice.name  # ToolChoiceFunction


def _starts_message(template: ResponseTemplate, prefix: str) -> bool:
    """Whether output after `prefix` starts the assistant message."""
    ends = [match.end() for match in template.start_anchor_re.finditer(prefix)]
    return bool(ends) and ends[-1] == len(prefix)


def _open_field(template: ResponseTemplate, prefix: str) -> str | None:
    """The field that `prefix` leaves open with its explicit opener, if any."""
    field = None
    for event in ResponseParser(template, prefix=prefix).initial_events:
        if event["type"] == "region_open" and event["end"] > event["start"]:
            field = event["field"]
        elif event["type"] in {"region_close", "region_malformed"}:
            field = None
    return field


def _tail_start(tokenizer: Any, vocab: dict[str, int], ids: list[int]) -> int:
    """Index of the last `<|start|>assistant` in `ids` (0 if none): the response-template
    parser truncates its prefix to the last start anchor, so nothing before it matters."""
    start_id = vocab.get(START_TOKEN)
    if start_id is None:
        return 0
    for index in range(len(ids) - 1, -1, -1):
        if ids[index] == start_id and _decode(tokenizer, ids[index : index + 2]).startswith(ANCHOR):
            return index
    return 0


def _message_tail(tokenizer: Any, vocab: dict[str, int], token_ids: Sequence[int]) -> str:
    """Decode `token_ids` from the last `<|start|>assistant` on, so a 100K-token prompt is
    not decoded and re-parsed per request; same parser state as decoding all of it."""
    ids = list(token_ids)
    return _decode(tokenizer, ids[_tail_start(tokenizer, vocab, ids) :])


def _continues_final_message(request: ChatCompletionRequest | ResponsesRequest) -> bool:
    if isinstance(request, ChatCompletionRequest):
        return bool(request.continue_final_message)
    try:
        from vllm.entrypoints.openai.responses.utils import should_continue_final_message

        return bool(should_continue_final_message(request.input))
    except Exception:
        return False


def _xgrammar_supports(structured_outputs: StructuredOutputsParams) -> bool:
    """Whether the constraint passes v0.30's xgrammar validation on its own."""
    from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar

    try:
        # validate_xgrammar_grammar rewrites `choice` into `grammar` in place.
        validate_xgrammar_grammar(SamplingParams(structured_outputs=replace(structured_outputs)))
    except (TypeError, ValueError, VLLMValidationError):
        return False
    return True


def structured_outputs_to_format(params: StructuredOutputsParams) -> dict[str, Any] | None:
    """An xgrammar structural-tag format for a request's own output constraint (the
    PR's abstract_parser.structured_outputs_to_format, as plain dicts)."""
    if params.json_object:
        return {"type": "json_schema", "json_schema": {"type": "object"}}
    if params.json is not None:
        schema = params.json
        if isinstance(schema, str):
            schema = json.loads(schema)
        return {"type": "json_schema", "json_schema": schema}
    if params.regex is not None:
        return {"type": "regex", "pattern": params.regex}
    if params.choice is not None:
        return {"type": "or", "elements": [{"type": "const_string", "value": c} for c in params.choice]}
    if params.grammar is not None:
        from vllm.v1.structured_output.utils import grammar_is_likely_lark

        grammar = params.grammar
        if grammar_is_likely_lark(grammar):
            import xgrammar

            try:
                grammar = str(xgrammar.Grammar.from_lark(grammar))
            except Exception as e:
                raise VLLMValidationError("Invalid grammar specification.") from e
        return {"type": "grammar", "grammar": grammar}
    return None


_REF_PREFIXES = ("#/$defs/", "#/definitions/")
_REF_DEPTH = 6


def _inline_refs(node: Any, defs: dict[str, Any], depth: int = 0) -> Any:
    """`node` with local `$ref`s (#/$defs/X, #/definitions/X) replaced by their target,
    up to _REF_DEPTH levels (deeper or unknown refs stay as they are)."""
    if isinstance(node, list):
        return [_inline_refs(item, defs, depth) for item in node]
    if not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if isinstance(ref, str) and depth < _REF_DEPTH:
        for prefix in _REF_PREFIXES:
            if ref.startswith(prefix) and isinstance(defs.get(ref[len(prefix) :]), dict):
                target = defs[ref[len(prefix) :]]
                merged = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
                return _inline_refs(merged, defs, depth + 1)
    return {k: _inline_refs(v, defs, depth) for k, v in node.items()}


def resolve_tool_refs(function: dict[str, Any]) -> dict[str, Any]:
    """An OpenAI function definition whose parameter schemas have local `$ref`s inlined.

    The template parser types argument values from each property's JSON-schema `type`
    (anyOf/oneOf included) and does not follow `$ref`: an object-valued parameter declared
    through `$defs` (pydantic and most MCP servers do this) came back as a JSON *string*,
    and the grammar left such tools' arguments unconstrained. Inlining fixes both; `$defs`
    stay in place for refs left unresolved."""
    params = function.get("parameters")
    if not isinstance(params, dict):
        return function
    defs = {**(params.get("definitions") or {}), **(params.get("$defs") or {})}
    if not defs or "$ref" not in json.dumps(params.get("properties") or {}):
        return function
    resolved = dict(params)
    resolved["properties"] = _inline_refs(params.get("properties") or {}, defs)
    return {**function, "parameters": resolved}


def _resolve_tool_dict(tool: dict[str, Any]) -> dict[str, Any]:
    if isinstance(tool.get("function"), dict):
        return {**tool, "function": resolve_tool_refs(tool["function"])}
    return resolve_tool_refs(tool)


def _ascii_space_classes(node: Any) -> Any:
    """Rewrite `\\s` inside negated character classes of regex formats to ASCII whitespace.

    The tool-name regex (upstream `_other_names`, from the template's `[^\\s<]+`) negates
    `\\s`; xgrammar 0.2.3 expands it to non-ASCII spaces too and then logs "Negative
    Character class contains byte greater than 127, clamping to 127" about 60 times per
    compile. With ASCII whitespace the grammar is the same up to names containing Unicode
    spaces, and compiles quietly."""
    if isinstance(node, dict):
        out = {k: _ascii_space_classes(v) for k, v in node.items()}
        if out.get("type") == "regex" and isinstance(out.get("pattern"), str):
            out["pattern"] = out["pattern"].replace("[^\\s", "[^ \\t\\n\\r")
        return out
    if isinstance(node, list):
        return [_ascii_space_classes(v) for v in node]
    return node


class BeamReasoningParser(ResponseTemplateReasoningParser):
    """Selects `BeamParser` as the reasoning parser. Also the engine core's structured
    output reasoning gate (is_reasoning_end / is_reasoning_end_streaming)."""

    @classmethod
    def unified_parser_cls(cls) -> type[ResponseTemplateParser]:
        return BeamParser

    @property
    def reasoning_start_str(self) -> str | None:
        if literal := super().reasoning_start_str:
            return literal
        thinking = self.response_template.fields.get(THINKING_FIELD)
        if thinking is None:
            return None
        openers = _openers(thinking, message_start_anchors(self.response_template))
        if not openers:
            return None
        marker = min(openers, key=len)
        return marker if all(opener.endswith(marker) for opener in openers) else None

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # Called on the prompt (serving, engine core): decode only its last message.
        ids = list(input_ids)
        return super().is_reasoning_end(ids[_tail_start(self.model_tokenizer, self.vocab, ids) :])


class BeamToolParser(ResponseTemplateToolParser):
    """Selects `BeamParser` as the tool parser. Extraction is BeamParser's; vLLM's
    JSON-schema path for required/named tool choice must never be used (Beam writes XML)."""

    supports_required_and_named = False

    @classmethod
    def unified_parser_cls(cls) -> type[ResponseTemplateParser]:
        return BeamParser


class _IncrementalReasoningCounter:
    """Upstream `_ReasoningTokenCounter` semantics (a generated token counts when its text
    overlaps the inside of a thinking region; text held back as a possible closer counts
    until the closer arrives) in O(new tokens) per call. Upstream re-walks every token on
    every call; v0.30 streaming calls count_reasoning_tokens after every delta, which made a
    long reasoning stream quadratic (2K tokens: 0.65 s of event-loop CPU, 32K: minutes).

    A token is final once it can no longer change: it overlaps a closed thinking span
    (counted), it ends before the open region or no region is open (not counted), or it
    ends inside the region body the parser has already emitted (counted). Tokens after
    that are pending and counted provisionally, as upstream counts them."""

    def __init__(self, template: ResponseTemplate, prefix: str, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.parser = ResponseParser(template, prefix=prefix)
        self.pos = len(self.parser.input_text)
        self.num_tokens = 0
        self.first: int | None = None
        self.last: int | None = None
        self.final = 0
        self.pending: list[tuple[int, int]] = []  # (start, end) of undecided tokens
        self.closed: list[tuple[int, int]] = []  # closed spans that pending tokens may touch
        self.open: int | None = None  # body start of the open thinking region
        self.emitted = 0  # body characters emitted since `open`
        self._track(self.parser.initial_events)

    def continues(self, token_ids: Sequence[int]) -> bool:
        if len(token_ids) < self.num_tokens:
            return False
        return self.num_tokens == 0 or (token_ids[0] == self.first and token_ids[self.num_tokens - 1] == self.last)

    def count(self, token_ids: Sequence[int]) -> int:
        new_ids = list(token_ids[self.num_tokens :])
        if new_ids:
            texts = [_decode(self.tokenizer, [token_id]) for token_id in new_ids]
            for text in texts:
                self.pending.append((self.pos, self.pos + len(text)))
                self.pos += len(text)
            if self.first is None:
                self.first = new_ids[0]
            self.last = new_ids[-1]
            self.num_tokens += len(new_ids)
            self._track(self.parser.feed("".join(texts)))
        self._settle()
        if self.open is None:
            return self.final
        return self.final + sum(1 for _, end in self.pending if end > self.open)

    def _settle(self) -> None:
        index = 0
        for start, end in self.pending:
            if any(begin < end and start < close for begin, close in self.closed):
                self.final += 1
            elif self.open is None or end <= self.open:
                pass
            elif end <= self.open + self.emitted:
                self.final += 1
            else:
                break
            index += 1
        del self.pending[:index]
        if self.pending:
            first_start = self.pending[0][0]
            self.closed = [span for span in self.closed if span[1] > first_start]
        else:
            self.closed = []

    def _track(self, events) -> None:
        for event in events:
            if event["field"] != THINKING_FIELD:
                continue
            if event["type"] == "region_open":
                self.open, self.emitted = event["end"], 0
            elif event["type"] == "region_chunk":
                self.emitted += len(event["text"])
            elif self.open is not None:
                self.closed.append((self.open, event["start"]))
                self.open = None


# The checkpoint's tool-call body (tokenizer_config.json `response_template`), which
# _ArgumentStream reads incrementally; any other template streams calls whole.
_TAG_PATTERN = '<rfl:parameter name="(?P<key>[^"]+)">\\n(?P<value>.*?)\\n(?:</rfl:parameter>|(?=<rfl:parameter name="))'
_VALUE_PARSER = {"name": "text", "args": {"strip": False}}
_TRANSFORM = {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}}
_PARAM_OPEN = '<rfl:parameter name="'
_KEY_END = '">\n'
_VALUE_END = "\n</rfl:parameter>"
_NEXT_PARAM = "\n" + _PARAM_OPEN  # a value also ends where the next parameter opens
# Schema types _coerce changes a value's text for; every other value stays the raw string.
_COERCED_TYPES = frozenset({"integer", "number", "boolean", "null", "object", "array"})


def _streams_beam_arguments(field: ResponseTemplateField | None) -> bool:
    return (
        field is not None
        and field.content == "xml-inline"
        and field.content_args == {"tag_pattern": _TAG_PATTERN, "value_parser": _VALUE_PARSER}
        and field.transform == _TRANSFORM
        and not field.transform_each
    )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


class _ArgumentStream:
    """The JSON arguments of one Beam call, as text, while its XML body arrives.

    The template parser builds the arguments at the close: `xml-inline` finds each
    `<rfl:parameter name="K">\\nV\\n</rfl:parameter>` in order (V ends at the first
    `\\n</rfl:parameter>`, or at a `\\n` before the next opener), types V by the tool schema
    (_coerce) and serializes the dict with json.dumps: keys in first-written order, a
    repeated key keeping its first place with its last value. So for a call written with
    each key once (the strict grammar's guarantee) parameter i is the i-th `"K": value` of
    the final text, and `feed` returns text that extends a prefix of it: `{` or `, ` and the
    key once a parameter's opener is complete, then a string value as it arrives (JSON
    escaping is per character), a typed value when its closer arrives. Bytes that may start
    a value closer are held until they resolve. Anything not in the grammar's shape (text
    between parameters other than one newline, a malformed opener, a repeated key) stops
    the stream; the caller sends the rest at the close. The closing `}` is never sent here.

    The one way the text can stop being a prefix is a body that ends inside a value (the
    output was cut off): the parser drops that parameter."""

    def __init__(self, properties: dict[str, Any] | None) -> None:
        self.properties = properties or {}  # empty: the parser types nothing
        self.buffer = ""  # unread body text
        self.state = "between"  # between | key | value | stopped
        self.count = 0
        self.keys: set[str] = set()
        self.head = ""  # `, "K": ` of a typed value
        self.raw: list[str] = []  # its text so far
        self.types: tuple[str, ...] = ()  # schema types of the open parameter
        self.string = False

    def feed(self, text: str) -> str:
        if self.state == "stopped":
            return ""
        self.buffer += text
        out: list[str] = []
        while self._step(out):
            pass
        return "".join(out)

    def _stop(self) -> bool:
        self.state, self.buffer, self.raw = "stopped", "", []
        return False

    def _step(self, out: list[str]) -> bool:
        """Read what `buffer` decides; False when it needs more text (or stopped)."""
        if self.state == "between":
            rest = self.buffer[1:] if self.buffer.startswith("\n") else self.buffer
            if not rest.startswith(_PARAM_OPEN):
                return False if _PARAM_OPEN.startswith(rest) else self._stop()
            self.buffer, self.state = rest[len(_PARAM_OPEN) :], "key"
            return True
        if self.state == "key":
            quote = self.buffer.find('"')
            if quote < 0:
                return False
            key, after = self.buffer[:quote], self.buffer[quote:]
            if not after.startswith(_KEY_END):
                return False if _KEY_END.startswith(after) else self._stop()
            if not key or key in self.keys:
                return self._stop()
            self.keys.add(key)
            types = _schema_types(self.properties[key]) if key in self.properties else ()
            self.types = types
            self.string = not _COERCED_TYPES.intersection(types)
            head = ("{" if not self.count else ", ") + _json(key) + ": "
            self.count += 1
            if self.string:
                out.append(head + '"')
            else:
                self.head, self.raw = head, []
            self.buffer, self.state = after[len(_KEY_END) :], "value"
            return True
        if self.state != "value":
            return False
        ends = [
            (i, n)
            for i, n in ((self.buffer.find(_VALUE_END), len(_VALUE_END)), (self.buffer.find(_NEXT_PARAM), 1))
            if i >= 0
        ]
        if ends:
            # The two closers differ at their third character: at most one starts at i.
            end, consumed = min(ends)
            self._value(self.buffer[:end], out, done=True)
            self.buffer, self.state = self.buffer[end + consumed :], "between"
            return True
        # Only the last newline can start a closer (each has one, first).
        cut = self.buffer.rfind("\n")
        tail = self.buffer[cut:] if cut >= 0 else ""
        if not (tail and (_VALUE_END.startswith(tail) or _NEXT_PARAM.startswith(tail))):
            cut = len(self.buffer)
        if cut:
            self._value(self.buffer[:cut], out, done=False)
            self.buffer = self.buffer[cut:]
        return False

    def _value(self, text: str, out: list[str], *, done: bool) -> None:
        if self.string:
            if text:
                out.append(_json(text)[1:-1])
            if done:
                out.append('"')
            return
        self.raw.append(text)
        if done:
            raw = "".join(self.raw)
            out.append(self.head + _json(_coerce(raw, self.types)))


class _StreamedCall:
    """A tool call whose region is open: announced once (index, id), then its arguments."""

    def __init__(self, name: str, arguments: _ArgumentStream | None) -> None:
        self.name = name
        self.arguments = arguments  # None: the arguments come at the close
        self.index: int | None = None
        self.sent: list[str] = []
        self.pending: list[str] = []  # read during the current delta

    def feed(self, text: str) -> None:
        if self.arguments is not None and (piece := self.arguments.feed(text)):
            self.pending.append(piece)


class _BeamResponseStream(_ResponseStream):
    """Upstream's stream with tool calls streamed: announced in the delta where the region
    opens and the arguments in fragments (_ArgumentStream) for tools in `strict_tools`,
    the rest of the close-time arguments at the close. A call that opens and closes in one
    delta (always, non-streaming) is upstream's one complete call, as are the warnings and
    drops of malformed or incomplete regions; an announced call cannot be taken back, so
    such a region only logs."""

    def __init__(self, *args, strict_tools: frozenset[str], **kwargs) -> None:
        self.strict_tools = strict_tools
        self._call: _StreamedCall | None = None
        super().__init__(*args, **kwargs)

    def feed(self, text: str, *, finished: bool) -> DeltaMessage | None:
        message = super().feed(text, finished=finished)
        update = self._call_update()
        if update is None:
            return message
        if message is None:
            return DeltaMessage(tool_calls=[update])
        # Assigned, not appended: serving dumps the delta with exclude_unset.
        message.tool_calls = [*message.tool_calls, update]
        return message

    def _call_update(self) -> DeltaToolCall | None:
        """The open call's announcement, or the arguments it read in this delta."""
        call = self._call
        if call is None:
            return None
        piece = "".join(call.pending)
        call.pending.clear()
        if piece:
            call.sent.append(piece)
        if call.index is None:
            call.index = self._next_tool_index
            self._next_tool_index += 1
            return DeltaToolCall(
                index=call.index,
                id=self.new_tool_call_id(call.name),
                type="function",
                function=DeltaFunctionCall(name=call.name, arguments=piece),
            )
        if not piece:
            return None
        return DeltaToolCall(index=call.index, function=DeltaFunctionCall(arguments=piece))

    def _route_tool_event(self, event: dict[str, Any], delta) -> None:
        if not self.parse_tools:
            return super()._route_tool_event(event, delta)
        kind, call = event["type"], self._call
        if kind == "region_open":
            name = (event.get("captures") or {}).get("name")
            self._call = None
            if isinstance(name, str) and name:
                arguments = None
                if name in self.strict_tools:
                    arguments = _ArgumentStream(self.parser._tool_params.get(name))
                self._call = _StreamedCall(name, arguments)
            return
        if call is None:
            # A region the prompt opened, or one without a name: upstream.
            return super()._route_tool_event(event, delta)
        if kind == "region_chunk":
            call.feed(event["text"])
            return
        self._call = None
        if call.index is None:
            return super()._route_tool_event(event, delta)
        log = _response_template.logger
        if kind == "region_malformed":
            log.warning("response_template: dropping a malformed tool call")
            return
        calls = _to_tool_calls(event.get("value"))
        if not calls:
            log.warning("response_template: dropping an incomplete tool call")
            return
        (_, arguments), more = calls[0], calls[1:]
        sent = "".join(call.sent)
        if arguments.startswith(sent):
            if len(arguments) > len(sent):
                delta.tool_calls.append(
                    DeltaToolCall(index=call.index, function=DeltaFunctionCall(arguments=arguments[len(sent) :]))
                )
        else:
            logger.warning(
                "beam: tool call %d (%s) ended inside a streamed argument value (cut off?); "
                "%d characters were streamed, the parsed arguments are %d characters and "
                "drop that value",
                call.index,
                call.name,
                len(sent),
                len(arguments),
            )
        for name, arguments in more:
            delta.tool_calls.append(
                DeltaToolCall(
                    index=self._next_tool_index,
                    id=self.new_tool_call_id(name),
                    type="function",
                    function=DeltaFunctionCall(name=name, arguments=arguments),
                )
            )
            self._next_tool_index += 1


def _strict_argument_tools(field: ResponseTemplateField | None, functions: list[dict[str, Any]]) -> frozenset[str]:
    """Names of the `functions` whose arguments tool_call_structural_tag(...,
    strict_arguments=True, sorted_arguments=True) holds to their schema: it derives the same
    _arguments_format for each and keeps calls to such a name to it (other names take any
    arguments). A name declared twice is left out (the grammar allows either schema)."""
    if field is None:
        return frozenset()
    names = [function.get("name") for function in functions]
    strict = set()
    for function, name in zip(functions, names, strict=True):
        if not isinstance(name, str) or names.count(name) != 1:
            continue
        parameters = function.get("parameters") or {"type": "object", "properties": {}}
        try:
            if _arguments_format(field, parameters, strict=True, sorted_arguments=True) is not None:
                strict.add(name)
        except ValueError:
            pass
    return frozenset(strict)


class BeamParser(ResponseTemplateParser):
    """Parse Beam output with the checkpoint's `response_template`, and constrain it with a
    grammar derived from that template."""

    reasoning_parser_cls = BeamReasoningParser
    tool_parser_cls = BeamToolParser

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        chat_template_kwargs = kwargs.get("chat_template_kwargs") or {}
        # From the message start the grammar opens the message with non-empty reasoning,
        # unless the request turns thinking off (chat_template_kwargs enable_thinking=false;
        # reasoning_effort "none" is mapped to "minimal" by frontend.py, see there).
        self._requires_reasoning = bool(chat_template_kwargs.get("enable_thinking", True))
        # Until the prompt is known (parse_delta passes it), assume the generation prompt.
        anchors = message_start_anchors(self.response_template)
        self._prefix = anchors[0] if anchors else ""
        self._prefix_known = True
        self._tools = [_resolve_tool_dict(tool) for tool in self._tools]

    @classmethod
    def compose(cls, reasoning_parser_cls, tool_parser_cls, enable_auto_tools: bool):
        return type(
            cls.__name__,
            (cls,),
            {
                "reasoning_parser_cls": reasoning_parser_cls,
                "tool_parser_cls": tool_parser_cls,
                "_enable_auto_tools": enable_auto_tools,
            },
        )

    def set_prompt_token_ids(self, prompt_token_ids: Sequence[int]) -> None:
        prompt_token_ids = list(prompt_token_ids)
        if prompt_token_ids == self._prompt_token_ids:
            return
        self._prompt_token_ids = prompt_token_ids
        self._prefix = _message_tail(self.model_tokenizer, self.vocab, prompt_token_ids)
        self._prefix_known = True
        self._stream = None
        self._reasoning_counter = None

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        if self._reasoning_parser is None or THINKING_FIELD not in self.response_template.fields:
            return 0
        counter = self._reasoning_counter
        if counter is None or not counter.continues(token_ids):
            counter = self._reasoning_counter = _IncrementalReasoningCounter(
                self.response_template, self._prefix, self.model_tokenizer
            )
        return counter.count(token_ids)

    def _new_stream(self, request, *, enable_auto_tools: bool, include_reasoning: bool) -> _ResponseStream:
        if not stream_tool_arguments():
            return super()._new_stream(
                request, enable_auto_tools=enable_auto_tools, include_reasoning=include_reasoning
            )
        # Upstream's _new_stream, with the streaming tool calls.
        strict_tools = getattr(request, STRICT_TOOLS_ATTR, None) or frozenset()
        if not _streams_beam_arguments(self.response_template.fields.get(TOOL_FIELD)):
            strict_tools = frozenset()
        return _BeamResponseStream(
            self.response_template,
            prefix=self._prefix,
            tools=self._tools,
            parse_reasoning=self._reasoning_parser is not None,
            include_reasoning=include_reasoning,
            parse_tools=self._tool_parser is not None
            and _tool_parsing_enabled(request, enable_auto_tools=enable_auto_tools),
            new_tool_call_id=self._new_tool_call_id,
            strict_tools=strict_tools,
        )

    def adjust_request(self, request):
        if self._prompt_token_ids is None and _continues_final_message(request):
            # The prompt ends inside a prefilled assistant message we have not seen.
            self._prefix_known = False
        return super().adjust_request(request)

    def _constrain_request(self, request):
        if not self._prefix_known:
            if self._forced(request):
                raise VLLMValidationError(
                    "tool_choice required or named is not supported with a continued "
                    "(prefilled) final assistant message",
                    parameter="tool_choice",
                )
            return request
        # v0.30 Responses serving does not pass reasoning_ended, so the engine's reasoning
        # gate holds the grammar until the reasoning closes: use the after-reasoning form.
        message_start = isinstance(request, ChatCompletionRequest) and _starts_message(
            self.response_template, self._prefix
        )
        tag = self._template_tag(request, message_start=message_start)
        if tag is not None:
            grammar = StructuredOutputsParams(structural_tag=json.dumps(tag))
            if isinstance(request, ChatCompletionRequest):
                request.structured_outputs = grammar
                request.response_format = None
                # From the message start the tag also covers the reasoning: serving then
                # passes reasoning_ended=True and the grammar applies from token 0.
                request._grammar_from_parser = message_start
            else:
                if request.text is not None and request.text.format is not None:
                    request.text = request.text.model_copy(update={"format": None})
                request.structured_outputs = grammar
        return request

    def _tools_enabled(self, request) -> bool:
        return (
            self._tool_parser is not None
            and bool(request.tools)
            and _tool_parsing_enabled(request, enable_auto_tools=self._enable_auto_tools)
        )

    def _forced(self, request) -> bool:
        return self._tools_enabled(request) and (request.tool_choice == "required" or _is_named_tool_choice(request))

    def _template_tag(self, request, *, message_start: bool) -> dict[str, Any] | None:
        """The template grammar for `request`, or None to leave its output to the
        request's own constraint, if any (PR #62052 BeamParser._template_tag)."""
        fields = self.response_template.fields
        named = _is_named_tool_choice(request)
        tools_enabled = self._tools_enabled(request)
        forced = self._forced(request)
        # The prompt may still list the tools, so only the grammar keeps the model from
        # calling them.
        tools_forbidden = bool(request.tools) and request.tool_choice == "none" and TOOL_FIELD in fields
        strict_tools = tools_enabled and (
            forced or (_enforce_strict_tools() and any(map(_tool_is_strict, request.tools)))
        )
        output = request.extract_structured_outputs()
        if forced and output is not None:
            logger.warning_once("beam: structured outputs are ignored because tool_choice forces a tool call.")
            output = None
        answer = fields.get(CONTENT_FIELD)
        if output is not None and (
            output.structural_tag is not None
            or answer is None
            or answer.open_re is None
            or not _xgrammar_supports(output)
        ):
            if strict_tools:
                logger.warning_once("beam: strict tools cannot be enforced alongside this structured output.")
            return None
        always = always_constrain_output() and not forced
        if not strict_tools and output is None and not tools_forbidden and not always:
            return None

        # Inside an open answer, only the always-on grammar applies, finishing a free
        # answer. Inside open reasoning, the grammar needs the reasoning gate to defer it
        # to the reasoning's end.
        open_field = _open_field(self.response_template, self._prefix)
        reasoning_gated = self._reasoning_parser is not None and request.include_reasoning
        tag = None
        if open_field == CONTENT_FIELD:
            if always and output is None:
                tag = answer_rest_structural_tag(fields[CONTENT_FIELD])
        elif open_field != THINKING_FIELD or reasoning_gated:
            functions = []
            if tools_enabled:
                functions = [resolve_tool_refs(t.get("function", t)) for t in _as_tool_dicts(request.tools)]
            if named:
                name = _named_tool(request)
                functions = [f for f in functions if f.get("name") == name]
            answer_format = None if output is None else structured_outputs_to_format(output)
            try:
                tag = tool_call_structural_tag(
                    fields.get(TOOL_FIELD),
                    functions,
                    parallel_tool_calls=(not named and request.parallel_tool_calls is not False),
                    thinking=fields.get(THINKING_FIELD),
                    answer=None if forced else answer,
                    answer_format=answer_format,
                    message_start=(message_start_anchors(self.response_template) if message_start else None),
                    requires_reasoning=self._requires_reasoning,
                    # Beam drifts from schemas the grammar does not enforce, and also
                    # writes arguments sorted by name.
                    sorted_arguments=True,
                    strict_arguments=True,
                )
            except ValueError as exc:
                raise VLLMValidationError(str(exc), parameter="tools") from exc
            if tag is not None and functions:
                # For the stream: these tools' calls are written with each key once.
                setattr(request, STRICT_TOOLS_ATTR, _strict_argument_tools(fields.get(TOOL_FIELD), functions))
        if tag is not None:
            tag = _ascii_space_classes(tag)
        if tag is None and forced:
            kind = "named" if named else "required"
            raise VLLMValidationError(
                f"beam: the response grammar cannot guarantee {kind} tool choice for this request",
                parameter="tool_choice",
            )
        if tag is None and strict_tools:
            logger.warning_once("beam: strict tools cannot be enforced for this request.")
        return tag
