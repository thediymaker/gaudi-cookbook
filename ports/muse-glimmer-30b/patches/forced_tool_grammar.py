#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""forced-tool-grammar: `tool_choice` "required" and named functions are enforced on Muse Glimmer.

The `muse_glimmer` tool parser opts out of vLLM's JSON-schema path for "required" and named
`tool_choice` (the model writes ATEM XML, not JSON) and installs nothing in its place. The request
then reaches the engine with no constraint and decodes exactly like "auto": the model often answers in
prose and the response has no tool call. A grammar alone would not help either, because the
structured-output manager waits for the reasoning parser's end signal, which this model's parser only
gives once a tool channel is open.

The patch makes the parser install an xgrammar structural tag for those requests that describes the
whole assistant turn in the model's own format, and marks the request so the grammar applies from the
first generated token:

    ( " to=self" <|message|> reasoning <|eom|> <|start|> "assistant" )?
    " to=<tool>" <|message|> <atem:function_calls> <atem:invoke name="<tool>"> parameters ...
    </atem:invoke> </atem:function_calls>
    ( <|eom|> <|start|> "assistant" [reasoning] <next call> )*      only "required" with parallel calls

The turn then ends with the model's own end token: <|eot|> is a request stop token, which the
grammar matcher accepts once the grammar is complete (a grammar that spells it out is rejected).

Parameters follow the declared order, required ones are mandatory, string enums and booleans are
enforced. The framing markers are matched as single special tokens. The parser's own extraction reads
the result unchanged, streaming or not. "auto", "none" and the Responses API are untouched.

Usage: forced_tool_grammar.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

PATCH_ID = "muse-glimmer-forced-tool-grammar"
MARKER = f"[gaudi-port:{PATCH_ID}]"
TARGET = ("vllm", "tool_parsers/muse_glimmer_tool_parser.py")

# The early return that leaves required/named choices unconstrained.
OLD = """        if request.tools and (
            tool_choice == "required"
            or isinstance(
                tool_choice, (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction)
            )
        ):
            return request
        return super().adjust_request(request)
"""
NEW = f"""        if request.tools and (
            tool_choice == "required"
            or isinstance(
                tool_choice, (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction)
            )
        ):
            return _forced_tool_grammar(self, request)  # {MARKER}
        return super().adjust_request(request)
"""
REQUIRED = (
    "\nclass MuseGlimmerToolParser(ToolParser):\n",
    "    supports_required_and_named = False\n",
    "\nlogger = init_logger(__name__)\n",
)

HELPERS = f'''

# --- {MARKER} ---------------------------------------------------------------
import json as _json  # noqa: E402

from vllm.sampling_params import StructuredOutputsParams as _SOParams  # noqa: E402

_FRAMING = ("<|message|>", "<|eom|>", "<|eot|>", "<|start|>")


def _framing_token_ids(tokenizer):
    ids = {{}}
    for marker in _FRAMING:
        tid = tokenizer.convert_tokens_to_ids(marker)
        if tid is None or tid < 0 or tid == getattr(tokenizer, "unk_token_id", None):
            raise ValueError(f"forced tool grammar: {{marker}} is not a single token")
        ids[marker] = int(tid)
    return ids


def _value_format(prop):
    """One ATEM parameter value: string enums and booleans exact, anything else free text."""
    enum = prop.get("enum") if isinstance(prop, dict) else None
    if enum and all(isinstance(v, str) for v in enum):
        return {{"type": "or", "elements": [{{"type": "const_string", "value": v}} for v in enum]}}
    if isinstance(prop, dict) and prop.get("type") == "boolean":
        return {{"type": "or", "elements": [
            {{"type": "const_string", "value": "true"}}, {{"type": "const_string", "value": "false"}}]}}
    return {{"type": "any_text", "excludes": ["</atem:parameter>"]}}


def _params_format(parameters):
    """Parameters in declared order; required ones mandatory, the others optional."""
    if not isinstance(parameters, dict):
        return None
    props = parameters.get("properties") or {{}}
    if not isinstance(props, dict) or not props:
        return None
    required = set(parameters.get("required") or [])
    elements = []
    for name, prop in props.items():
        tag = {{
            "type": "tag",
            "begin": '<atem:parameter name="' + name + '">',
            "content": _value_format(prop),
            "end": "</atem:parameter>\\n",
        }}
        elements.append(tag if name in required else {{"type": "optional", "content": tag}})
    return {{"type": "sequence", "elements": elements}}


def build_structural_tag(tools, forced_name, parallel, ids):
    """Structural tag for one forced assistant turn, or None when no tool matches."""
    msg = {{"type": "token", "token": ids["<|message|>"]}}
    next_message = [
        {{"type": "token", "token": ids["<|eom|>"]}},
        {{"type": "token", "token": ids["<|start|>"]}},
        {{"type": "const_string", "value": "assistant"}},
    ]
    reasoning = {{"type": "optional", "content": {{"type": "sequence", "elements": [
        {{"type": "const_string", "value": " to=self"}},
        msg,
        {{"type": "any_tokens", "exclude_tokens": [ids[m] for m in _FRAMING]}},
    ] + next_message}}}}
    calls = []
    for tool in tools:
        fn = getattr(tool, "function", None)
        name = getattr(fn, "name", None)
        if not name or (forced_name is not None and name != forced_name):
            continue
        elements = [
            {{"type": "const_string", "value": " to=" + name}},
            msg,
            {{"type": "const_string", "value": '<atem:function_calls>\\n<atem:invoke name="' + name + '">\\n'}},
        ]
        params = _params_format(getattr(fn, "parameters", None))
        if params is not None:
            elements.append(params)
        elements.append({{"type": "const_string", "value": "</atem:invoke>\\n</atem:function_calls>"}})
        calls.append({{"type": "sequence", "elements": elements}})
    if not calls:
        return None
    call = calls[0] if len(calls) == 1 else {{"type": "or", "elements": calls}}
    one_call = {{"type": "sequence", "elements": [reasoning, call]}}
    turn = [one_call]
    if forced_name is None and parallel:
        turn.append({{"type": "star", "content": {{"type": "sequence", "elements": next_message + [one_call]}}}})
    # No <|eot|> element: it is a stop token, accepted by the matcher only where the grammar is complete.
    return {{"type": "structural_tag", "format": {{"type": "sequence", "elements": turn}}}}


def _forced_tool_grammar(parser, request):
    if not isinstance(request, ChatCompletionRequest):
        return request
    choice = request.tool_choice
    forced_name = choice.function.name if isinstance(choice, ChatCompletionNamedToolChoiceParam) else None
    tag = build_structural_tag(
        request.tools,
        forced_name,
        request.parallel_tool_calls is not False,
        _framing_token_ids(parser.model_tokenizer),
    )
    if tag is None:
        return request
    request.structured_outputs = _SOParams(structural_tag=_json.dumps(tag))
    request.response_format = None
    # The grammar covers the optional reasoning channel itself, so it must apply from the first
    # token instead of waiting for the reasoning parser (which only ends on a tool channel).
    request._grammar_from_parser = True
    return request
'''

UPSTREAM_PRIVATE_ATTR = "_grammar_from_parser: bool = PrivateAttr"


class Drift(Exception):
    """The installed source does not match what this patch expects."""


def patch(src: str, protocol: str) -> str:
    for needle in REQUIRED + (OLD,):
        n = src.count(needle)
        if n != 1:
            raise Drift(f"{needle.strip().splitlines()[0]!r} found {n} times, expected 1")
    if protocol.count(UPSTREAM_PRIVATE_ATTR) != 1:
        raise Drift("ChatCompletionRequest has no _grammar_from_parser attribute")
    return src.replace(OLD, NEW, 1).rstrip("\n") + "\n" + HELPERS


def locate(rel: str, root: str | None) -> Path | None:
    if root:
        p = Path(root) / "vllm" / rel
        return p if p.is_file() else None
    spec = importlib.util.find_spec("vllm")
    for loc in (spec.submodule_search_locations or []) if spec else []:
        p = Path(loc) / rel
        if p.is_file():
            return p
    return None


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".gaudi-port.tmp")
    tmp.write_text(text)
    os.chmod(tmp, path.stat().st_mode & 0o7777)
    os.replace(tmp, path)
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(f"{path.stem}.*.pyc"):
            pyc.unlink(missing_ok=True)


def run(check: bool, root: str | None) -> int:
    path = locate(TARGET[1], root)
    proto = locate("entrypoints/openai/chat_completion/protocol.py", root)
    if path is None or proto is None:
        print(f"{MARKER} tool parser or chat protocol not found", file=sys.stderr)
        return 2
    src = path.read_text()
    if MARKER in src:
        print(f"{MARKER} applied")
        return 0
    try:
        new = patch(src, proto.read_text())
        compile(new, str(path), "exec", dont_inherit=True)
    except (Drift, SyntaxError) as e:
        print(f"{MARKER} {e}; source drifted, nothing written", file=sys.stderr)
        return 2
    if check:
        print(f"{MARKER} not applied (anchors match)")
        return 1
    write_atomic(path, new)
    print(f"{MARKER} patched {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report the state without writing (0 applied, 1 not applied)")
    ap.add_argument("--root", help="directory holding vllm/ (default: the installed package)")
    args = ap.parse_args()
    return run(args.check, args.root)


if __name__ == "__main__":
    sys.exit(main())
