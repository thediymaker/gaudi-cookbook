# SPDX-License-Identifier: Apache-2.0
"""Serving frontend for Beam in vLLM v0.30 (installed by register() in every process).

1. Parsers named `beam` (beam_hpu/parser.py): `--reasoning-parser beam` and
   `--tool-call-parser beam` (with --enable-auto-tool-choice) select BeamParser, one
   parser for reasoning, answer and tool calls, plus the response grammar. v0.30's
   ParserManager knows nothing about unified response-template parsers, so get_parser is
   wrapped: when either name is `beam` it returns BeamParser composed with the selected
   classes (and refuses to combine `beam` with another parser, as upstream does).

2. reasoning_effort. v0.30 passes the top-level `reasoning_effort` into the chat template
   as `reasoning_effort` (overriding chat_template_kwargs) and adds
   enable_thinking = (effort != "none"). The Beam template maps minimal/low/medium/high/
   xhigh/max to 8/32/64/128/256/512, takes ints (chat_template_kwargs only: the API field
   is a string enum), defaults to 512 when unset, and raises on anything else. It has no
   off switch: Beam always writes an analysis channel, and the template card says a budget
   below 8 is not respected. Unpatched, "none" is an HTTP 400 from the template.
   Decision: "none" is served as "minimal" (8), the smallest trained budget, both as the
   top-level field (Chat Completions and Responses `reasoning.effort`) and as
   chat_template_kwargs.reasoning_effort. Reason: clients and gateways send "none" to
   mean "as little thinking as possible" across models; a 400 breaks them for no gain,
   and forcing an empty analysis channel through the grammar would be off-distribution
   (the model was never trained without reasoning). The response still carries a short
   `reasoning`. enable_thinking is not set to false by the mapping, so the grammar still
   requires non-empty reasoning.

The template's `allow_missing_thinking` is left at its default (true): with false, any
request whose history has an assistant turn without reasoning raises (HTTP 400), which is
every multi-turn request from clients that drop `reasoning` (most OpenAI-compatible chat
UIs and agent loops). v0.30 renders `reasoning` (and `reasoning_content`, renamed to
`reasoning` by the request model) from assistant history when a client sends it back.
"""

from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger("vllm.beam_hpu.frontend")

NAME = "beam"
_PATCHED = "_beam_hpu_frontend"
NONE_EFFORT_AS = "minimal"


def _register_parsers() -> None:
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    ReasoningParserManager.register_lazy_module(NAME, "beam_hpu.parser", "BeamReasoningParser")
    ToolParserManager.register_lazy_module(NAME, "beam_hpu.parser", "BeamToolParser")


def _install_parser_selection() -> None:
    from vllm.parser.parser_manager import ParserManager

    if getattr(ParserManager, _PATCHED, False):
        return
    orig = ParserManager.get_parser.__func__

    def get_parser(
        cls,
        tool_parser_name=None,
        reasoning_parser_name=None,
        enable_auto_tools=False,
        model_name=None,
        is_harmony=False,
        **kwargs,
    ):
        if NAME not in (tool_parser_name, reasoning_parser_name):
            return orig(
                cls,
                tool_parser_name=tool_parser_name,
                reasoning_parser_name=reasoning_parser_name,
                enable_auto_tools=enable_auto_tools,
                model_name=model_name,
                is_harmony=is_harmony,
                **kwargs,
            )
        if {tool_parser_name, reasoning_parser_name} - {NAME, None, ""}:
            raise TypeError(
                f"--reasoning-parser {reasoning_parser_name} and "
                f"--tool-call-parser {tool_parser_name} cannot be combined"
            )
        reasoning_cls = cls.get_reasoning_parser(reasoning_parser_name)
        tool_cls = cls.get_tool_parser(tool_parser_name, enable_auto_tools, model_name)
        if reasoning_cls is None and tool_cls is None:
            return None
        tokenizer = kwargs.get("tokenizer")
        if tokenizer is not None:
            # Upstream's startup check (v0.30 callers pass no tokenizer; BeamParser then
            # fails on its first request if the checkpoint lacks `response_template`).
            from vllm.parser.response_template import validate_tokenizer_response_template

            validate_tokenizer_response_template(
                tokenizer, reasoning_parser=reasoning_parser_name, tool_parser=tool_parser_name if tool_cls else None
            )
        from .parser import BeamParser

        return BeamParser.compose(reasoning_cls, tool_cls, enable_auto_tools)

    ParserManager.get_parser = classmethod(get_parser)
    setattr(ParserManager, _PATCHED, True)


def map_none_effort(request) -> bool:
    """Serve reasoning_effort "none" as "minimal" (module docstring). True if changed."""
    changed = False
    if getattr(request, "reasoning_effort", None) == "none":  # Chat Completions
        request.reasoning_effort = NONE_EFFORT_AS
        changed = True
    reasoning = getattr(request, "reasoning", None)  # Responses
    if reasoning is not None and getattr(reasoning, "effort", None) == "none":
        request.reasoning = reasoning.model_copy(update={"effort": NONE_EFFORT_AS})
        changed = True
    kwargs = getattr(request, "chat_template_kwargs", None)
    if isinstance(kwargs, dict) and kwargs.get("reasoning_effort") == "none":
        request.chat_template_kwargs = {**kwargs, "reasoning_effort": NONE_EFFORT_AS}
        changed = True
    if changed:
        logger.info_once(
            "beam: reasoning_effort 'none' is served as '%s' (8): the model always reasons and has no off switch",
            NONE_EFFORT_AS,
        )
    return changed


def _install_effort_mapping() -> None:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    for cls in (ChatCompletionRequest, ResponsesRequest):
        if getattr(cls, _PATCHED, False):
            continue
        orig = cls.build_chat_params

        def build_chat_params(self, *args, _orig=orig, **kwargs):
            map_none_effort(self)
            return _orig(self, *args, **kwargs)

        cls.build_chat_params = build_chat_params
        setattr(cls, _PATCHED, True)


HISTORY_REASONING_KEYS = ("reasoning_content", "thinking")


def history_reasoning(message):
    """`message` with `reasoning` filled from `reasoning_content` or `thinking` when it has
    none (a shallow copy; the input is not modified), else `message` itself."""
    if not isinstance(message, dict) or message.get("role") != "assistant" or message.get("reasoning"):
        return message
    for key in HISTORY_REASONING_KEYS:
        value = message.get(key)
        if isinstance(value, str) and value:
            return {**message, "reasoning": value}
    return message


TEMPLATE_ROLES = ("system", "developer", "user", "assistant", "tool")


def check_role(message) -> None:
    """HTTP 400 for a message role the Beam chat template does not render.

    v0.30 accepts any role string (CustomChatCompletionMessageParam) and the checkpoint template
    renders only system/developer (leading), user, assistant and tool: any other message is
    dropped without a trace, so a typo or a legacy `function` role silently changes the
    conversation the model sees (beam-verifier: messages_role_invalid must be refused)."""
    role = message.get("role") if isinstance(message, dict) else None
    if role not in TEMPLATE_ROLES:
        from vllm.exceptions import VLLMValidationError

        raise VLLMValidationError(
            f"beam: message role {role!r} is not supported; use one of {', '.join(TEMPLATE_ROLES)}",
            parameter="messages",
        )


def _install_history_reasoning() -> None:
    """Prior-turn reasoning sent back as `reasoning_content` (or `thinking`, the template's
    own name) reaches the chat template on every entry point; roles the template would drop
    are refused (check_role).

    v0.30 renders assistant history reasoning only from `reasoning`
    (chat_utils._parse_chat_message_content). ChatCompletionRequest renames the deprecated
    `reasoning_content` to `reasoning` in a model validator, but TokenizeChatRequest
    (/tokenize with messages) and the other chat-message entry points do not, so there an
    assistant turn's reasoning_content was silently dropped and rendered as an empty analysis
    channel (beam-verifier formatter suite: 6 of 24 render cases). The parse step is the
    single place every path goes through."""
    from vllm.entrypoints import chat_utils

    if getattr(chat_utils, _PATCHED, False):
        return
    orig = chat_utils._parse_chat_message_content

    def _parse_chat_message_content(message, *args, **kwargs):
        check_role(message)
        return orig(history_reasoning(message), *args, **kwargs)

    _parse_chat_message_content.__wrapped__ = orig
    chat_utils._parse_chat_message_content = _parse_chat_message_content
    setattr(chat_utils, _PATCHED, True)


def install() -> None:
    _register_parsers()
    _install_parser_selection()
    _install_effort_mapping()
    _install_history_reasoning()
