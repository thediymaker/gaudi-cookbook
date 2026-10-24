# SPDX-License-Identifier: Apache-2.0
r"""Llama 4 tool-call parser that reads every call syntax Llama 4 writes (``llama4_lenient``).

Load it with ``--tool-parser-plugin <this file>`` and select it with
``--tool-call-parser llama4_lenient``. Pair it with ``llama4-tools.jinja`` from the same directory.
The pure-Python half (``parse_output``) has no vLLM dependency.

Accepted forms (those Meta's reference decoder accepts):
  <function=NAME>{json}</function>          Meta's documented custom-tool syntax; one per line,
                                            closing tag optional, <|python_start|> optional
  [a(x=1), b(y="z")]  /  a(x=1)             Python list, or bare calls to known tools
  <|python_start|> ... <|python_end|>       a list, bare calls, tags or JSON inside
  {"name": .., "parameters": ..}            JSON object(s) at the start of a line
A Python list is looked for at the start of the reply and at the start of any later line that
begins "[<tool>(", because the model sometimes writes a bracketed line that is not a call first.

Argument repair, guided by the request's tool schemas (nothing is evaluated):
  * "true"/"false" -> bool, numeric strings -> int/float, where the schema wants those types
  * null for an optional parameter -> omitted
  * a string holding a JSON or Python-literal list or object, where the schema wants one

Call policy, env LLAMA4_TOOL_CALL_POLICY (request.parallel_tool_calls=false forces "first"):
  independent (default) keep only calls that cannot depend on an earlier call's result in the
         same turn: a leading run of read-only calls (read/glob/grep/list/webfetch...), or a
         leading run of edits/writes to distinct files. A mutation after a read, a second edit of
         one file, or anything after a command run is cut. Llama 4 tends to plan a whole task as
         one parallel list (read, then an edit of text it has not seen, then a test run); the
         blind steps fail or apply a wrong edit. Dropping them makes the model take the next step
         with the result in view. LLAMA4_READONLY_TOOLS / LLAMA4_EDIT_TOOLS (comma-separated tool
         names) override the built-in lists.
  reads  a leading run of read-only calls, else only the first call
  first  only the first call
  all    every call (upstream behaviour)

Streaming: prose streams as it arrives. Once a call region starts, nothing more is sent until each
call parses; then the calls the policy allows go out whole (no truncated JSON prefixes).
Call-shaped text that never parses is returned as content at the end.
"""

from __future__ import annotations

import ast
import contextlib
import json
import os
import re
import uuid
from collections.abc import Sequence

PY_START = "<|python_start|>"
PY_END = "<|python_end|>"
_TRAILERS = ("<|eom|>", "<|eot|>", "<|end_of_text|>")
_STOP_TOKENS = ("<|eom|>", "<|eot|>", "<|end_of_text|>")
_NAME_CONSTS = {"true": True, "false": False, "null": None, "True": True, "False": False, "None": None}
_BARE_START = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\(")
_TAG = re.compile(r"<function=([A-Za-z_][\w.\-]*)>")
_JSON_START = re.compile(r'(^|\n)[ \t]*\{\s*"(name|type|function)"')
_DEFAULT_EDIT = (
    "edit",
    "write",
    "patch",
    "multiedit",
    "write_file",
    "replace_in_file",
    "write_to_file",
    "str_replace",
    "apply_patch",
)
_DEFAULT_READONLY = (
    "read",
    "glob",
    "grep",
    "list",
    "webfetch",
    "ls",
    "view",
    "search",
    "read_file",
    "list_files",
    "search_files",
    "codesearch",
)


# --------------------------------------------------------------------------- python literals
class _NotACall(ValueError):
    pass


def _value(node: ast.AST):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str | int | float | bool) or node.value is None:
            return node.value
        raise _NotACall(type(node.value).__name__)
    if isinstance(node, ast.Name) and node.id in _NAME_CONSTS:
        return _NAME_CONSTS[node.id]
    if isinstance(node, ast.List | ast.Tuple):
        return [_value(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        out = {}
        for k, v in zip(node.keys, node.values, strict=False):
            if k is None:
                raise _NotACall("dict unpacking")
            key = _value(k)
            if not isinstance(key, str):
                raise _NotACall("non-string key")
            out[key] = _value(v)
        return out
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        v = _value(node.operand)
        if isinstance(v, bool) or not isinstance(v, int | float):
            raise _NotACall("unary on non-number")
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _value(node.left), _value(node.right)
        if isinstance(a, str) and isinstance(b, str):
            return a + b
        raise _NotACall("+ on non-strings")
    raise _NotACall(type(node).__name__)


def _parse_calls(src: str, tool_names, allow_bare: bool):
    """[(name, args)] if ``src`` is entirely python tool calls, else None."""
    src = src.strip()
    if not src:
        return None
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return None
    nodes = []
    for st in tree.body:
        if not isinstance(st, ast.Expr):
            return None
        v = st.value
        if isinstance(v, ast.List):
            if not v.elts or not all(isinstance(e, ast.Call) for e in v.elts):
                return None
            nodes.extend(v.elts)
        elif isinstance(v, ast.Call) and allow_bare:
            nodes.append(v)
        elif isinstance(v, ast.Tuple) and allow_bare and v.elts and all(isinstance(e, ast.Call) for e in v.elts):
            nodes.extend(v.elts)
        else:
            return None
    calls = []
    for c in nodes:
        if not isinstance(c.func, ast.Name) or c.args:
            return None
        name = c.func.id
        if tool_names and name not in tool_names:
            return None
        try:
            args = {}
            for kw in c.keywords:
                if kw.arg is None:
                    return None
                args[kw.arg] = _value(kw.value)
        except _NotACall:
            return None
        calls.append((name, args))
    return calls or None


def _longest_calls(region: str, tool_names, allow_bare: bool):
    calls = _parse_calls(region, tool_names, allow_bare)
    if calls:
        return calls
    for i in range(len(region) - 1, 0, -1):
        if region[i] in ")]":
            calls = _parse_calls(region[: i + 1], tool_names, allow_bare)
            if calls:
                return calls
    return None


def strip_markers(text: str) -> str:
    text = text.replace(PY_START, "").replace(PY_END, "")
    for t in _TRAILERS:
        text = text.replace(t, "")
    return text


def _loads_lenient(s: str):
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        pass
    try:
        return _value(ast.parse(s.strip(), mode="eval").body)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- other syntaxes
def _parse_tags(body: str, tool_names):
    """<function=NAME>{json}</function> calls. Returns (prose_before, calls, closed) or None."""
    m = _TAG.search(body)
    if not m:
        return None
    pre, pos, calls, dec = body[: m.start()], m.start(), [], json.JSONDecoder()
    closed = True
    while True:
        m = _TAG.match(body, pos)
        if not m:
            break
        rest = body[m.end() :]
        lead = len(rest) - len(rest.lstrip())
        try:
            args, n = dec.raw_decode(rest.lstrip())
        except Exception:  # noqa: BLE001
            end = rest.find("</function>")
            if end == -1:
                closed = False
                break
            args, n = _loads_lenient(rest[:end]), end - lead
        if not isinstance(args, dict):
            closed = False
            break
        if tool_names and m.group(1) not in tool_names:
            return None
        calls.append((m.group(1), args))
        off = m.end() + lead + n
        tail = body[off:]
        if tail.lstrip().startswith("</function>"):
            off += tail.find("</function>") + len("</function>")
        nxt = _TAG.search(body, off)
        if not nxt or body[off : nxt.start()].strip(" \n\t;,"):
            break
        pos = nxt.start()
    return (pre, calls, closed) if calls else None


def _parse_json(body: str, tool_names):
    m = _JSON_START.search(body)
    if not m:
        return None
    start = m.start() + (1 if m.group(1) else 0)
    pre, s = body[:start], body[start:]
    dec, calls, pos = json.JSONDecoder(), [], 0
    while pos < len(s):
        while pos < len(s) and s[pos] in " \n\t;,":
            pos += 1
        if pos >= len(s):
            break
        try:
            obj, n = dec.raw_decode(s[pos:])
        except Exception:  # noqa: BLE001
            break
        pos += n
        if not isinstance(obj, dict):
            break
        fn = obj.get("function") if isinstance(obj.get("function"), dict) else {}
        name = obj.get("name") or fn.get("name")
        args = obj.get("parameters", obj.get("arguments", fn.get("arguments", fn.get("parameters", {}))))
        if isinstance(args, str):
            args = _loads_lenient(args)
        if not name or not isinstance(args, dict) or (tool_names and name not in tool_names):
            break
        calls.append((name, args))
    return (pre, calls) if calls else None


def _pythonic_regions(body: str, tool_names):
    """Candidate starts of a pythonic call region: (index, kind)."""
    out = []
    i = body.find(PY_START)
    if i != -1:
        out.append((i, "marker"))
    s = body.lstrip()
    lead = len(body) - len(s)
    if s.startswith("["):
        out.append((lead, "list"))
    elif tool_names:
        m = _BARE_START.match(s)
        if m and m.group(0)[:-1] in tool_names:
            out.append((lead, "bare"))
    if tool_names:
        for m in re.finditer(r"\n[ \t]*\[([A-Za-z_][A-Za-z0-9_]*)\(", body):
            if m.group(1) in tool_names:
                out.append((m.start() + 1 + (len(m.group(0)) - 1 - len(m.group(0).lstrip())), "list"))
    return sorted(set(out))


def _parse_pythonic(body: str, tool_names):
    for idx, kind in _pythonic_regions(body, tool_names):
        region = body[idx:]
        if kind == "marker":
            inner = region[len(PY_START) :].split(PY_END, 1)[0]
            calls = _longest_calls(inner, None, allow_bare=True)
        elif kind == "list":
            calls = _longest_calls(region, None, allow_bare=False)
        else:
            calls = _longest_calls(region, tool_names, allow_bare=True)
        if calls:
            return body[:idx], calls
    return None


# --------------------------------------------------------------------------- schema repair
def _schemas(tools):
    out = {}
    for t in tools or []:
        fn = getattr(t, "function", None)
        if fn is None and isinstance(t, dict):
            fn = t.get("function", t)
        if fn is None:
            continue
        name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
        params = fn.get("parameters") if isinstance(fn, dict) else getattr(fn, "parameters", None)
        if name:
            out[name] = params or {}
    return out


def _want(schema: dict):
    t = (schema or {}).get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if t is None:
        for k in ("anyOf", "oneOf"):
            for alt in (schema or {}).get(k) or []:
                if isinstance(alt, dict) and alt.get("type") not in (None, "null"):
                    return alt.get("type")
    return t


def _coerce(args: dict, params: dict) -> dict:
    props = (params or {}).get("properties") or {}
    required = set((params or {}).get("required") or [])
    out = dict(args)
    for k, v in args.items():
        if v is None and k not in required and props:
            del out[k]  # null for an optional parameter means "not given"
            continue
        want = _want(props.get(k) or {})
        if not isinstance(v, str) or want in (None, "string"):
            continue
        s = v.strip()
        if want == "boolean" and s.lower() in ("true", "false"):
            out[k] = s.lower() == "true"
        elif want == "integer" and re.fullmatch(r"-?\d+", s):
            out[k] = int(s)
        elif want == "number" and re.fullmatch(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", s):
            out[k] = float(s)
        elif want in ("array", "object") and s[:1] in "[{":
            parsed = _loads_lenient(s)
            if isinstance(parsed, list if want == "array" else dict):
                out[k] = parsed
    return out


def _readonly():
    env = os.environ.get("LLAMA4_READONLY_TOOLS")
    return {x.strip() for x in env.split(",")} if env else set(_DEFAULT_READONLY)


def _edit_tools():
    env = os.environ.get("LLAMA4_EDIT_TOOLS")
    return {x.strip() for x in env.split(",")} if env else set(_DEFAULT_EDIT)


def _target(args: dict):
    for k in ("filePath", "file_path", "path", "file", "filename"):
        if isinstance(args.get(k), str):
            return args[k]
    return None


def apply_policy(calls, policy: str):
    if len(calls) <= 1 or policy == "all":
        return calls
    if policy == "first":
        return calls[:1]
    ro = _readonly()
    if policy == "independent":
        # Keep only calls that cannot depend on the result of an earlier call in the same turn:
        # a run of read-only calls, or a run of edits/writes to distinct files. A mutation after
        # a read, a second edit of the same file, or anything after a command run is cut.
        if calls[0][0] in ro:
            n = 1
            while n < len(calls) and calls[n][0] in ro:
                n += 1
            return calls[:n]
        ed = _edit_tools()
        if calls[0][0] not in ed or _target(calls[0][1]) is None:
            return calls[:1]
        seen, n = {_target(calls[0][1])}, 1
        while n < len(calls) and calls[n][0] in ed:
            t = _target(calls[n][1])
            if t is None or t in seen:
                break
            seen.add(t)
            n += 1
        return calls[:n]
    if calls[0][0] not in ro:
        return calls[:1]
    n = 1
    while n < len(calls) and calls[n][0] in ro:
        n += 1
    return calls[:n]


def _policy(request=None) -> str:
    if request is not None and getattr(request, "parallel_tool_calls", True) is False:
        return "first"
    p = os.environ.get("LLAMA4_TOOL_CALL_POLICY", "independent").strip().lower()
    return p if p in ("all", "independent", "reads", "first") else "independent"


# --------------------------------------------------------------------------- entry point
def parse_output(text: str, tools, policy: str | None = None):
    """Non-streaming: (content, [(name, args)]). ``tools`` = request tools (dicts/objects) or names."""
    schemas = {} if tools is None or (isinstance(tools, set | frozenset)) else _schemas(tools)
    names = set(schemas) or (set(tools) if isinstance(tools, set | frozenset) else None) or None
    body = text
    for t in _TRAILERS:
        body = body.replace(t, "")
    flat = body.replace(PY_START, "").replace(PY_END, "")
    found = None
    r = _parse_tags(flat, names)
    if r:
        found = (r[0], r[1])
    if not found:
        found = _parse_pythonic(body, names)
    if not found:
        found = _parse_json(flat, names)
    if not found:
        return strip_markers(text), []
    pre, calls = found
    calls = [(n, _coerce(a, schemas.get(n))) for n, a in calls]
    calls = apply_policy(calls, policy or _policy())
    return strip_markers(pre), calls


def _maybe_region_start(text: str) -> int:
    """Index where a call region may start, or -1. Conservative: may hold back prose briefly."""
    cands = [i for i in (text.find(PY_START), text.find("<function=")) if i != -1]
    s = text.lstrip()
    if s.startswith(("[", "{")):
        cands.append(len(text) - len(s))
    m = re.search(r"\n[ \t]*(\[[A-Za-z_]|\{\s*\"|<function=)", text)
    if m:
        cands.append(m.start() + 1)
    return min(cands) if cands else -1


def _safe_prose_end(text: str) -> int:
    """How much prose can be sent now: hold back a trailing line that may still turn into a
    call (a partial <|python_start|> / <function= marker, or a line opening with [ or {)."""
    nl = text.rfind("\n")
    line = text[nl + 1 :]
    s = line.lstrip()
    for p in (PY_START, "<function="):
        for k in range(len(p) - 1, 0, -1):
            if line.endswith(p[:k]):
                return len(text) - k
    if s == "" or s[:1] in "[{<":
        return nl + 1 if nl != -1 else 0
    return len(text)


def _region_settled(region: str, n: int) -> bool:
    """True once the region text shows its n parsed calls are complete (the closing ] / } /
    </function> / ) / <|python_end|> was written, or another tag has started)."""
    t = region.rstrip()
    return n > 0 and (t.endswith(("</function>", "]", "}", ")", PY_END)) or region.count("<function=") > n)


# --------------------------------------------------------------------------- vLLM integration
try:
    from vllm.logger import init_logger
    from vllm.tool_parsers import ToolParserManager
    from vllm.tool_parsers.llama4_pythonic_tool_parser import Llama4PythonicToolParser

    try:  # vLLM 0.30 and later
        from vllm.entrypoints.generate.base.protocol import (
            DeltaFunctionCall,
            DeltaMessage,
            DeltaToolCall,
            ExtractedToolCallInformation,
            FunctionCall,
            ToolCall,
        )
    except ImportError:
        try:  # vLLM 0.26
            from vllm.entrypoints.openai.engine.protocol import (
                DeltaFunctionCall,
                DeltaMessage,
                DeltaToolCall,
                ExtractedToolCallInformation,
                FunctionCall,
                ToolCall,
            )
        except ImportError:  # older vLLM
            from vllm.entrypoints.openai.protocol import (  # type: ignore
                DeltaFunctionCall,
                DeltaMessage,
                DeltaToolCall,
                ExtractedToolCallInformation,
                FunctionCall,
                ToolCall,
            )
    _HAVE_VLLM = True
except Exception:  # noqa: BLE001
    _HAVE_VLLM = False


if _HAVE_VLLM:
    logger = init_logger(__name__)
    _PARTIALS = tuple(p[:k] for p in (PY_START, "<function=") for k in range(1, len(p)))

    @ToolParserManager.register_module("llama4_lenient")
    class Llama4LenientToolParser(Llama4PythonicToolParser):
        def __init__(self, tokenizer, *args, **kwargs):
            super().__init__(tokenizer, *args, **kwargs)
            vocab = {}
            with contextlib.suppress(Exception):
                vocab = self.vocab
            self._stop_ids = {vocab[t] for t in _STOP_TOKENS if t in vocab}
            self._sent = 0  # chars of the cleaned text already sent as content
            self._region = None  # start of the call region once seen
            self._emitted = 0  # tool calls already sent

        def adjust_request(self, request):
            request = super().adjust_request(request)
            if request.tools and request.tool_choice != "none":
                request.skip_special_tokens = False  # keep <|python_start|> visible
            return request

        def extract_tool_calls(self, model_output, request):
            content, calls = parse_output(model_output, request.tools, _policy(request))
            if not calls:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=content)
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=[
                    ToolCall(
                        type="function", function=FunctionCall(name=n, arguments=json.dumps(a, ensure_ascii=False))
                    )
                    for n, a in calls
                ],
                content=content.strip() or None,
            )

        def _delta_calls(self, calls):
            deltas = []
            for i in range(self._emitted, len(calls)):
                name, args = calls[i]
                arg_json = json.dumps(args, ensure_ascii=False)
                deltas.append(
                    DeltaToolCall(
                        index=i,
                        type="function",
                        id=f"chatcmpl-tool-{uuid.uuid4().hex[:24]}",
                        function=DeltaFunctionCall(name=name, arguments=arg_json),
                    )
                )
                while len(self.prev_tool_call_arr) <= i:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= i:
                    self.streamed_args_for_tool.append("")
                self.prev_tool_call_arr[i] = {"name": name, "arguments": arg_json}
                self.streamed_args_for_tool[i] = arg_json
            self._emitted = len(calls)
            self.current_tool_id = self._emitted
            return deltas

        def extract_tool_calls_streaming(
            self,
            previous_text: str,
            current_text: str,
            delta_text: str,
            previous_token_ids: Sequence[int],
            current_token_ids: Sequence[int],
            delta_token_ids: Sequence[int],
            request,
        ):
            final = bool(self._stop_ids & set(delta_token_ids or ()))
            text = current_text
            for t in _TRAILERS:
                text = text.replace(t, "")
            if self._region is None:
                idx = _maybe_region_start(text)
                if idx == -1:
                    end = len(text) if final else _safe_prose_end(text)
                    chunk = strip_markers(text[self._sent : end]) if end > self._sent else ""
                    self._sent = max(self._sent, end)
                    return DeltaMessage(content=chunk) if chunk else None
                self._region = idx
                if idx > self._sent:
                    chunk = strip_markers(text[self._sent : idx])
                    self._sent = idx
                    if chunk.strip() and not final:
                        return DeltaMessage(content=chunk)
            # Buffered call region: send each call once it parses and the policy allows it.
            content, calls = parse_output(text[self._region :], request.tools, _policy(request))
            ready = final or _region_settled(text[self._region :], len(calls))
            if calls and len(calls) > self._emitted and ready:
                extra = ""
                if self._emitted == 0 and content.strip():
                    extra = content  # prose inside the region before the first call
                    self._sent = len(text)
                msg = DeltaMessage(tool_calls=self._delta_calls(calls))
                if extra:
                    msg = DeltaMessage(content=extra, tool_calls=msg.tool_calls)
                return msg
            if final and self._emitted == 0:
                chunk = strip_markers(text[self._sent :])
                self._sent = len(text)
                logger.info("llama4_lenient: call-shaped text did not parse; returned as content (%.120s)", chunk)
                return DeltaMessage(content=chunk) if chunk else None
            return None
