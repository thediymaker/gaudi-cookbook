"""Serving-frontend gate for Mistral Large 4 in vLLM v0.30 (every process that loads the plugin).

The tokenizer is tekken v15 (`--tokenizer-mode mistral`): the prompt is mistral-common's own rendering, and the
reasoning effort travels into it as a request field. vLLM v0.30's MistralTokenizer validates
`reasoning_effort` against mistral-common's ReasoningEffort enum before any prompt is built. The image ships
mistral-common 1.11.7, whose enum is {none, high}: `low` and `medium`, two of the model's three efforts
(FP8 README: low / medium / high, default high), are rejected with a 400. 1.13.0 adds them, and the PR's
parser builds its grammar through 1.13.0's `get_grammar_for_chat_request`.

So 1.13.0 (installed --no-deps by the port's install.sh: its pins are all met by the image) is a hard requirement of this
plugin, and install() fails closed (RuntimeError at plugin load) on anything older, instead of a server that
starts and then rejects two efforts or fails on the first JSON-mode request.

No parser is registered here: v0.30's `mistral` reasoning and tool parsers are thin adapters over
vllm/parser/mistral.py, which the vendored PR file replaces. Serve with
`--reasoning-parser mistral --tool-call-parser mistral --enable-auto-tool-choice`.

Request corrections (install(), API server):
- Effort aliases. OpenAI clients send `minimal`, `none`, `xhigh` and `max`; tekken v15 takes only low / medium /
  high, so they failed with a 400 at rendering. minimal and none become low (the model has no thinking-off
  switch), xhigh and max become high. Unset stays unset: the model's own default (high).
- JSON nesting guard. A deeply nested response_format json_schema or structured_outputs.json hit Python's
  recursion limit in mistral-common's schema check (a 500 from depth ~300). v0.30 has no check_json_nesting, so
  validate_request_params refuses a constraint nested deeper than MAX_JSON_DEPTH with a 400.
- Plain chat without a grammar. v0.30 defaults tool_choice to "none" when a request has no tools, and the PR
  parser then attaches mistral-common's "none" grammar (an optional [THINK] block, then free text) to every chat
  request. It only bans other control tokens, but a grammar row never drafts, so DSpark never ran on chat
  traffic. A request with no tools, no response_format (or text) and no structured_outputs is now treated as
  mistral-common's own default (tool_choice auto without tools: no grammar). ML4_PLAIN_CHAT_GRAMMAR=1 restores
  the grammar.
- Forced calls without leading text. For tool_choice required / named, mistral-common's grammar allows free text
  before the call, so a named choice could end at max_tokens with no call. The optional text is dropped
  (ML4_STRICT_FORCED_TOOLS=0 keeps it).
"""

from __future__ import annotations

import os

MIN_MISTRAL_COMMON = (1, 13, 0)
EFFORTS = ("low", "medium", "high")  # the model's efforts (FP8 README); default high
EFFORT_ALIASES = {"minimal": "low", "none": "low", "xhigh": "high", "max": "high"}
MAX_JSON_DEPTH = int(os.environ.get("ML4_MAX_JSON_DEPTH", "128"))  # depth 150 passed, 900 hit the recursion limit
PLAIN_CHAT_GRAMMAR = os.environ.get("ML4_PLAIN_CHAT_GRAMMAR", "0") == "1"
STRICT_FORCED_TOOLS = os.environ.get("ML4_STRICT_FORCED_TOOLS", "1") == "1"


def _version_tuple(v: str) -> tuple[int, ...]:
    out = []
    for part in v.split(".")[:3]:
        digits = "".join(c for c in part if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def problems() -> list[str]:
    """Why this environment cannot serve Mistral Large 4's frontend ([] when it can)."""
    try:
        import mistral_common
        from mistral_common.protocol.instruct.request import ReasoningEffort
    except ImportError as e:
        return [f"mistral_common not importable: {e}"]
    out = []
    version = getattr(mistral_common, "__version__", "0")
    if _version_tuple(version) < MIN_MISTRAL_COMMON:
        out.append(f"mistral_common {version} < {'.'.join(map(str, MIN_MISTRAL_COMMON))}")
    values = {e.value for e in ReasoningEffort}
    missing = [e for e in EFFORTS if e not in values]
    if missing:
        out.append(f"ReasoningEffort {sorted(values)} lacks {missing}")
    return out


def install() -> None:
    found = problems()
    if found:
        raise RuntimeError(
            "mistral_large4_hpu: " + "; ".join(found) + ". Install "
            "mistral-common 1.13.0 with --no-deps (ports/mistral-large-4/install.sh)."
        )
    _install_request_fixes()


def json_depth(obj) -> int:
    """Nesting depth of a JSON value (dict / list, or JSON text), without recursion."""
    if isinstance(obj, str | bytes):
        text = obj.decode(errors="replace") if isinstance(obj, bytes) else obj
        depth = best = 0
        in_str = esc = False
        for ch in text:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch in "[{":
                depth += 1
                best = max(best, depth)
            elif ch in "]}":
                depth -= 1
        return best
    best, stack = 0, [(obj, 1)]
    while stack:
        o, d = stack.pop()
        if isinstance(o, dict):
            best = max(best, d)
            stack.extend((v, d + 1) for v in o.values())
        elif isinstance(o, list | tuple):
            best = max(best, d)
            stack.extend((v, d + 1) for v in o)
    return best


def _constraints(request) -> list:
    out = []
    rf = getattr(request, "response_format", None)
    js = getattr(rf, "json_schema", None) if rf is not None else None
    if js is not None:
        out.append(getattr(js, "json_schema", None) or getattr(js, "schema_", None) or getattr(js, "schema", None))
    so = getattr(request, "structured_outputs", None)
    if so is not None:
        out.append(getattr(so, "json", None))
    for t in getattr(request, "tools", None) or []:
        out.append(getattr(getattr(t, "function", None), "parameters", None))
    return [c for c in out if c is not None and not isinstance(c, bool)]


def _strict_forced_call(request):
    """tool_choice required / named: mistral-common's grammar lets free text come before the forced call
    (`fcalls: content? (...)+`), and the model can talk until max_tokens without ever calling (a named choice on
    "Hello there!" wrote 1024 tokens of text). Drop that optional text: after the reasoning, the call comes first."""
    so = getattr(request, "structured_outputs", None)
    g = getattr(so, "grammar", None) if so is not None else None
    tc = getattr(request, "tool_choice", None)
    forced = tc == "required" or (tc is not None and not isinstance(tc, str))
    if forced and isinstance(g, str) and g.count("fcalls: content? (") == 1:
        import dataclasses

        request.structured_outputs = dataclasses.replace(so, grammar=g.replace("fcalls: content? (", "fcalls: ("))
    return request


def _install_request_fixes() -> None:
    try:
        import vllm.tokenizers.mistral as mt
        from vllm.exceptions import VLLMValidationError
    except ImportError:
        return
    if getattr(mt.validate_request_params, "_ml4", False):
        return
    orig_validate = mt.validate_request_params

    def validate_request_params(request):
        effort = getattr(request, "reasoning_effort", None)
        if effort in EFFORT_ALIASES:
            request.reasoning_effort = EFFORT_ALIASES[effort]
        for c in _constraints(request):
            if json_depth(c) > MAX_JSON_DEPTH:
                raise VLLMValidationError(
                    f"JSON schema nested deeper than {MAX_JSON_DEPTH} levels is not supported.",
                    parameter="response_format",
                )
        return orig_validate(request)

    validate_request_params._ml4 = True
    mt.validate_request_params = validate_request_params

    try:
        from vllm.parser.mistral import MistralParser
    except ImportError:
        return
    orig_adjust = MistralParser.adjust_request

    def adjust_request(self, request):
        rf = getattr(request, "response_format", None)
        if (
            not PLAIN_CHAT_GRAMMAR
            and hasattr(request, "messages")
            and not getattr(request, "tools", None)
            and getattr(request, "structured_outputs", None) is None
            and (rf is None or getattr(rf, "type", None) == "text")
            and not getattr(request, "_grammar_from_parser", False)
        ):
            request.response_format = None
            return request
        request = orig_adjust(self, request)
        return _strict_forced_call(request) if STRICT_FORCED_TOOLS else request

    MistralParser.adjust_request = adjust_request
