# SPDX-License-Identifier: Apache-2.0
"""MuseSparkHPUToolParser: the PR's Muse ATEM tool parser plus binding forced tool_choice.

The PR parser (vllm/tool_parsers/muse_tool_parser.py) opts out of vLLM's JSON-schema
path for tool_choice "required" / named function, because that path forces JSON while
Muse writes ATEM XML, and installs nothing in its place: the request reaches the engine
with no structured-output constraint, so "required" decodes exactly like "auto" and the
model may answer in prose. Even a grammar installed by the usual route would not apply:
the Muse reasoning parser only reports the end of reasoning once an ATEM tool channel
exists, and the structured-output manager waits for that.

For Chat Completions requests with tool_choice "required" or a named function this
subclass installs an xgrammar structural tag that describes the whole assistant turn in
Muse's own format, and sets `request._grammar_from_parser` (v0.30's hook, also used by
Mistral's grammar factory) so serving passes reasoning_ended=True and the grammar
applies from the first generated token:

    ( " to=self" <|message|> text <|eom|> <|start|> "assistant" )?        # reasoning
    ( " to=commentary" <|message|> text <|eom|> <|start|> "assistant" )?  # user-visible note
    " to=<tool>" <|message|> "<atem:function_calls>\n<atem:invoke name=\"<tool>\">\n"
        <atem:parameter name="p">value</atem:parameter>\n ...   # declared order,
                                                                 # required ones mandatory,
                                                                 # enums enforced
    "</atem:invoke>\n</atem:function_calls>"
    ( <|eom|> <|start|> "assistant" [reasoning] [commentary] <next call> )*   # "required"
                                                                            # + parallel
    ( <|eot|> )?                              # then the request's stop token ends the turn

The turn end is optional on purpose. <|eot|> is an eos id in the checkpoint's
generation_config, so vLLM puts it in the request's stop tokens and hands those to
xgrammar (override_stop_tokens). xgrammar accepts a stop token only once the grammar is
complete, and never as a grammar element: with a mandatory trailing <|eot|> the bitmask
allowed <|eot|> after the call but accept_token rejected it, the scheduler terminated the
request ("grammar rejected tokens [200008]") and the client got HTTP 500. With the
optional element the grammar is complete after the last call, so <|eot|> is accepted as
the stop token (or, if a deployment drops it from the stop set, as the element, after
which the tokenizer eos ends the request).

Message bodies (reasoning, commentary, parameter values) exclude every special token:
the chat template escapes "<|" in all text it renders, so the model never sees a special
token inside a body, and a stray marker there would desynchronise the parser.

tool_choice "none" on a request that carries tools gets a structural tag too. The tool
schemas stay in the system prompt (and tool calls may sit in the history), so the model
calls a tool anyway, and for "none" vLLM runs no tool parser: the call came back as raw
ATEM markup in `content` (5 of 5 on the real model before this). The "none" grammar allows
only the channels that are not tool calls:

    ( " to=self" <|message|> text <|eom|> <|start|> "assistant" )?
    ( " to=commentary" <|message|> text <|eom|> <|start|> "assistant" )?
    " to=user" <|message|> text ( <|eot|> )?

It is skipped when the request brings its own structured output (response_format /
structured_outputs), and requests without tools (whose tool_choice defaults to "none")
are not touched.

The framing markers are matched as single special tokens (xgrammar TokenFormat), never
spelled out, so the model emits exactly the token sequence its chat template renders
for a tool call. The parser's extraction (non-streaming and streaming) is unchanged.
tool_choice "auto" and the Responses API are untouched.

"""

from __future__ import annotations

import json

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.logger import init_logger
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.muse_tool_parser import MuseToolParser

logger = init_logger("vllm.muse_spark_hpu.tool_parser")  # under "vllm": its handlers print

MARKERS = ("<|message|>", "<|eom|>", "<|eot|>", "<|start|>")


def marker_token_ids(tokenizer) -> dict[str, int]:
    ids = {}
    for marker in MARKERS:
        tid = tokenizer.convert_tokens_to_ids(marker)
        if tid is None or tid < 0 or tid == getattr(tokenizer, "unk_token_id", None):
            raise ValueError(f"muse forced-tool grammar: no single token for {marker}")
        ids[marker] = int(tid)
    return ids


def special_token_ids(tokenizer, ids: dict[str, int]) -> list[int]:
    """Every special token id (the framing markers included)."""
    out = set(ids.values())
    decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if isinstance(decoder, dict):
        out.update(int(i) for i, tok in decoder.items() if getattr(tok, "special", False))
    out.update(int(i) for i in (getattr(tokenizer, "all_special_ids", None) or []))
    return sorted(out)


def _value_format(prop) -> dict:
    """Grammar for one ATEM parameter value (scalars as-is, lists/objects as JSON)."""
    enum = prop.get("enum") if isinstance(prop, dict) else None
    if enum and all(isinstance(v, str) for v in enum):
        return {"type": "or", "elements": [{"type": "const_string", "value": v} for v in enum]}
    if isinstance(prop, dict) and prop.get("type") == "boolean":
        return {
            "type": "or",
            "elements": [{"type": "const_string", "value": "true"}, {"type": "const_string", "value": "false"}],
        }
    # "<|" begins every special token; the template renders it as "< |" in values
    return {"type": "any_text", "excludes": ["</atem:parameter>", "<|"]}


def _params_format(parameters) -> dict | None:
    """Parameters in declared order; required ones mandatory, others optional."""
    if not isinstance(parameters, dict):
        return None
    props = parameters.get("properties") or {}
    if not isinstance(props, dict) or not props:
        return None
    required = set(parameters.get("required") or [])
    elements = []
    for name, prop in props.items():
        tag = {
            "type": "tag",
            "begin": '<atem:parameter name="' + name + '">',
            "content": _value_format(prop),
            "end": "</atem:parameter>\n",
        }
        elements.append(tag if name in required else {"type": "optional", "content": tag})
    return {"type": "sequence", "elements": elements}


def build_structural_tag(
    tools, forced_name: str | None, parallel: bool, ids: dict[str, int], specials: list[int] | None = None
) -> dict | None:
    msg = {"type": "token", "token": ids["<|message|>"]}
    eom = {"type": "token", "token": ids["<|eom|>"]}
    eot = {"type": "token", "token": ids["<|eot|>"]}
    start = {"type": "token", "token": ids["<|start|>"]}
    next_message = [eom, start, {"type": "const_string", "value": "assistant"}]
    body_excludes = sorted(set(specials or []) | {ids[m] for m in MARKERS})

    def side_channel(recipient):
        return {
            "type": "optional",
            "content": {
                "type": "sequence",
                "elements": [
                    {"type": "const_string", "value": " to=" + recipient},
                    msg,
                    {"type": "any_tokens", "exclude_tokens": body_excludes},
                ]
                + next_message,
            },
        }

    reasoning = side_channel("self")
    commentary = side_channel("commentary")

    calls = []
    for tool in tools or []:
        fn = getattr(tool, "function", None)
        name = getattr(fn, "name", None)
        if not name or (forced_name is not None and name != forced_name):
            continue
        elements = [
            {"type": "const_string", "value": " to=" + name},
            msg,
            {"type": "const_string", "value": '<atem:function_calls>\n<atem:invoke name="' + name + '">\n'},
        ]
        params = _params_format(getattr(fn, "parameters", None))
        if params is not None:
            elements.append(params)
        elements.append({"type": "const_string", "value": "</atem:invoke>\n</atem:function_calls>"})
        calls.append({"type": "sequence", "elements": elements})
    if not calls:
        return None
    one_call = {
        "type": "sequence",
        "elements": [reasoning, commentary, calls[0] if len(calls) == 1 else {"type": "or", "elements": calls}],
    }
    turn = [one_call]
    if forced_name is None and parallel:
        turn.append({"type": "star", "content": {"type": "sequence", "elements": next_message + [one_call]}})
    # Optional: <|eot|> is a stop token, which xgrammar accepts only on a complete grammar
    # (see the module docstring); a mandatory element here can never be consumed.
    turn.append({"type": "optional", "content": eot})
    return {"type": "structural_tag", "format": {"type": "sequence", "elements": turn}}


def build_no_tool_tag(ids: dict[str, int], specials: list[int] | None = None) -> dict:
    """tool_choice "none": reasoning and commentary optional, then the user answer; no tool channel."""
    msg = {"type": "token", "token": ids["<|message|>"]}
    body = {"type": "any_tokens", "exclude_tokens": sorted(set(specials or []) | {ids[m] for m in MARKERS})}
    next_message = [
        {"type": "token", "token": ids["<|eom|>"]},
        {"type": "token", "token": ids["<|start|>"]},
        {"type": "const_string", "value": "assistant"},
    ]

    def side_channel(recipient):
        return {
            "type": "optional",
            "content": {
                "type": "sequence",
                "elements": [{"type": "const_string", "value": " to=" + recipient}, msg, body] + next_message,
            },
        }

    return {
        "type": "structural_tag",
        "format": {
            "type": "sequence",
            "elements": [
                side_channel("self"),
                side_channel("commentary"),
                {"type": "const_string", "value": " to=user"},
                msg,
                body,
                {"type": "optional", "content": {"type": "token", "token": ids["<|eot|>"]}},
            ],
        },
    }


def _has_own_constraint(request) -> bool:
    so = getattr(request, "structured_outputs", None)
    if so is not None and not so.all_constraints_none():
        return True
    rf = getattr(request, "response_format", None)
    return rf is not None and getattr(rf, "type", "text") != "text"


_SPECIALS: dict[int, tuple[dict[str, int], list[int]]] = {}


def _token_tables(tokenizer) -> tuple[dict[str, int], list[int]]:
    """(marker ids, all special ids), once per tokenizer object (one per server process)."""
    key = id(tokenizer)
    if key not in _SPECIALS:
        ids = marker_token_ids(tokenizer)
        _SPECIALS[key] = (ids, special_token_ids(tokenizer, ids))
    return _SPECIALS[key]


class MuseSparkHPUToolParser(MuseToolParser):
    _logged: set = set()

    def adjust_request(self, request):
        request = super().adjust_request(request)
        if not isinstance(request, ChatCompletionRequest) or not request.tools:
            return request
        tool_choice = getattr(request, "tool_choice", None)
        ids, specials = _token_tables(self.model_tokenizer)
        if tool_choice == "required":
            kind, forced_name = "required", None
        elif isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):
            kind, forced_name = "named", tool_choice.function.name
        elif tool_choice == "none":
            if _has_own_constraint(request):
                return request
            kind, forced_name = "none", None
        else:
            return request
        if kind == "none":
            tag = build_no_tool_tag(ids, specials)
        else:
            tag = build_structural_tag(
                request.tools, forced_name, getattr(request, "parallel_tool_calls", True) is not False, ids, specials
            )
        if tag is None:
            return request
        request.structured_outputs = StructuredOutputsParams(structural_tag=json.dumps(tag))
        request.response_format = None
        # The grammar covers the optional reasoning channel itself: apply it from token 0
        # instead of waiting for the reasoning parser (which only fires on ATEM).
        request._grammar_from_parser = True
        if kind not in MuseSparkHPUToolParser._logged:
            MuseSparkHPUToolParser._logged.add(kind)
            # warning: this logger is outside vLLM's "vllm" tree, so INFO is not printed
            logger.warning(
                "muse_spark_hpu: tool_choice=%s constrained by the ATEM structural tag "
                "(optional <|eot|> end, %d special tokens excluded from bodies)",
                kind,
                len(specials),
            )
        return request
