#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""gemma4-required-tool-grammar: enforce tool_choice "required" and named tools for the gemma4 parser.

The engine-based gemma4 tool parser skips structured outputs for required and named tool choice,
so the model is free to write its native tool-call syntax, or prose. Nothing forces a call: a
request with tool_choice "required" or a named function can come back with no tool call at all.

The fix restores the constraint the generic tool parsers use:

* adjust_request falls through to ToolParser.adjust_request for required and named tool choice,
  so the output is constrained to the tools' JSON schema;
* the parser declares supports_required_and_named = True, so the serving layer parses that JSON
  into tool calls (streaming and non-streaming) instead of handing it to the native-syntax parser;
* extract_required_tool_call_streaming treats an IndexError from partial_json_parser like
  MalformedJSON (not parseable yet), instead of letting it end the stream;
* engine-based streams hand the tool extractor only the newest delta, so for this parser the
  required/named JSON path accumulates the streamed text on the stream state and passes the full
  text to the JSON extractors. Other parsers are not affected.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "gemma4-required-tool-grammar"
MARKER = f"[gaudi-fix:{FIX_ID}]"

ADJUST_OLD = """        if request.tools:
            tc = request.tool_choice
            if tc == "required" or isinstance(
                tc, (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction)
            ):
                request.skip_special_tokens = False
                return request
        return super().adjust_request(request)"""
ADJUST_NEW = f"""        # {MARKER} required and named tool choice keep the base structured-output
        # constraint (the tools' JSON schema), so a call is always produced.
        return super().adjust_request(request)"""

SUPPORTS_OLD = "    supports_required_and_named = False\n"
SUPPORTS_NEW = f"""    supports_required_and_named = True  # {MARKER} JSON path for required and named
    gaudi_fix_accumulate_required_json = True  # {MARKER} see parser/abstract_parser.py
"""

STREAMING_OLD = """    except (
        partial_json_parser.core.exceptions.MalformedJSON,
        json.JSONDecodeError,
    ):
        obj = None

    # check if the current text is a valid array"""
STREAMING_NEW = f"""    except (
        partial_json_parser.core.exceptions.MalformedJSON,
        json.JSONDecodeError,
        IndexError,  # {MARKER} raised by partial_json_parser on some partial input
    ):
        obj = None

    # check if the current text is a valid array"""

ABSTRACT_OLD = """        supports_required_and_named = self._tool_parser.supports_required_and_named

        if request.tool_choice == "none":"""
ABSTRACT_NEW = f"""        supports_required_and_named = self._tool_parser.supports_required_and_named
        # {MARKER} engine-based streams pass only the newest delta; the JSON extractors for
        # required/named need the accumulated text, so keep it on the stream state.
        if (
            self._engine_based
            and supports_required_and_named
            and getattr(self._tool_parser, "gaudi_fix_accumulate_required_json", False)
            and request.tool_choice
            and (
                request.tool_choice == "required"
                or isinstance(request.tool_choice, (ToolChoiceFunction, ChatCompletionNamedToolChoiceParam))
            )
        ):
            previous_text = getattr(self._stream_state, "gaudi_fix_json_text", "")
            current_text = previous_text + (delta_text or "")
            self._stream_state.gaudi_fix_json_text = current_text

        if request.tool_choice == "none":"""


class Drift(Exception):
    """The installed source does not match what this fix expects."""


def replace_once(src: str, old: str, new: str, label: str) -> str:
    n = src.count(old)
    if n != 1:
        raise Drift(f"anchor {label!r} found {n} times, expected 1")
    return src.replace(old, new, 1)


def require_once(src: str, needle: str, label: str) -> None:
    n = src.count(needle)
    if n != 1:
        raise Drift(f"{label}: {needle.strip()!r} found {n} times, expected 1")


def patch_tool_parser(src: str, ctx: dict[str, str]) -> str:
    require_once(src, "class Gemma4EngineToolParser(", "gemma4_engine_tool_parser.py")
    src = replace_once(src, ADJUST_OLD, ADJUST_NEW, "adjust_request required/named skip")
    return replace_once(src, SUPPORTS_OLD, SUPPORTS_NEW, "supports_required_and_named")


def patch_streaming(src: str, ctx: dict[str, str]) -> str:
    require_once(src, "def extract_required_tool_call_streaming(", "streaming.py")
    return replace_once(src, STREAMING_OLD, STREAMING_NEW, "partial JSON exceptions")


def patch_abstract(src: str, ctx: dict[str, str]) -> str:
    require_once(src, "    def _extract_tool_calls_streaming(\n", "abstract_parser.py")
    require_once(src, "        self._stream_state = StreamState(", "abstract_parser.py")
    for name in ("from openai.types.responses import ToolChoiceFunction", "    ChatCompletionNamedToolChoiceParam,\n"):
        if name not in src.split("\nclass ", 1)[0]:
            raise Drift(f"abstract_parser.py does not import {name.strip()!r}")
    return replace_once(src, ABSTRACT_OLD, ABSTRACT_NEW, "required/named streaming branch")


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm", "tool_parsers/gemma4_engine_tool_parser.py", patch_tool_parser, "tool_parser"),
    ("vllm", "tool_parsers/streaming.py", patch_streaming, "streaming"),
    ("vllm", "parser/abstract_parser.py", patch_abstract, "abstract_parser"),
)


# ---- common driver (identical in every fix) --------------------------------------------------


def locate(pkg: str, rel: str, root: str | None) -> Path | None:
    """Find <pkg>/<rel> under --root, or in the installed package without importing it."""
    if root:
        p = Path(root) / pkg / rel
        return p if p.is_file() else None
    spec = importlib.util.find_spec(pkg)
    for loc in (spec.submodule_search_locations or []) if spec else []:
        p = Path(loc) / rel
        if p.is_file():
            return p
    return None


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".gaudi-fix.tmp")
    tmp.write_text(text)
    os.chmod(tmp, path.stat().st_mode & 0o7777)
    os.replace(tmp, path)
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(f"{path.stem}.*.pyc"):
            pyc.unlink(missing_ok=True)


def run(check: bool, root: str | None) -> int:
    ctx: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for pkg, rel, _, key in TARGETS:
        path = locate(pkg, rel, root)
        if path is None:
            print(f"{MARKER} {pkg}/{rel} not found", file=sys.stderr)
            return 2
        paths[key] = path
        ctx[key] = path.read_text()
    if UPSTREAM_FIXED and all(MARKER not in ctx[k] and needle in ctx[k] for k, needle in UPSTREAM_FIXED):
        print(f"{MARKER} not needed: the upstream fix is present")
        return 0
    done, todo = [], []
    try:
        for _pkg, _rel, fn, key in TARGETS:
            if fn is None:
                continue
            src = ctx[key]
            if MARKER in src:
                done.append(paths[key])
                continue
            new = fn(src, ctx)
            if MARKER not in new:
                raise Drift("patched text lacks the marker")
            compile(new, str(paths[key]), "exec", dont_inherit=True)
            todo.append((paths[key], new))
    except (Drift, SyntaxError) as e:
        print(f"{MARKER} {e}; source drifted, nothing written", file=sys.stderr)
        return 2
    if done and todo:
        print(f"{MARKER} partly applied ({', '.join(map(str, done))}); restore the files and re-run", file=sys.stderr)
        return 2
    if not todo:
        print(f"{MARKER} applied")
        return 0
    if check:
        print(f"{MARKER} not applied (anchors match)")
        return 1
    for path, new in todo:
        write_atomic(path, new)
        print(f"{MARKER} patched {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report the state without writing (0 applied, 1 not applied)")
    ap.add_argument("--root", help="directory holding vllm/ and vllm_gaudi/ (default: the installed packages)")
    args = ap.parse_args()
    return run(args.check, args.root)


if __name__ == "__main__":
    sys.exit(main())
