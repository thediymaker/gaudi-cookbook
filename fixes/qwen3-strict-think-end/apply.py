#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""qwen3-strict-think-end: end Qwen3 reasoning only at </think>, not at a <tool_call> inside it.

vLLM's engine-based Qwen3 parser (vllm/parser/qwen3.py) treats a <tool_call> that appears while
the model is still reasoning as an implicit end of reasoning: the grammar has a
(REASONING, TOOL_START) -> TOOL_PREAMBLE transition, and on the vLLM commit v0.26.0 tracks
Qwen3Parser.is_reasoning_end also returns True on an unpaired <tool_call> id. This helps models
that call tools without closing </think>, but a model that quotes the chat template's tool-format
example while thinking then has that example returned as a real tool call, before its actual
call.

Opt-in. With the fix only </think> ends the reasoning: the implicit-end transition is removed,
and where is_reasoning_end has its own unpaired-<tool_call> check, it is skipped. Apply it only
for models that always close </think> before calling a tool.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "qwen3-strict-think-end"
MARKER = f"[gaudi-fix:{FIX_ID}]"

TRANSITION_OLD = """            # Tool call directly from reasoning (implicit end)
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
"""
TRANSITION_NEW = f"""            # {MARKER} no implicit end of reasoning at a <tool_call>: only </think>
            # ends the reasoning, so a tool-call example quoted while thinking stays reasoning.
"""

# Only on the vLLM commit v0.26.0 tracks: a separate unpaired-<tool_call> check.
REASONING_END_OLD = """    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        if super().is_reasoning_end(input_ids):
            return True
"""
REASONING_END_NEW = f"""    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        if super().is_reasoning_end(input_ids):
            return True
        return False  # {MARKER} only </think> ends the reasoning
"""


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


def patch_parser(src: str, ctx: dict[str, str]) -> str:
    require_once(src, "class Qwen3Parser(", "qwen3.py")
    src = replace_once(src, TRANSITION_OLD, TRANSITION_NEW, "implicit-end transition")
    n = src.count(REASONING_END_OLD)
    if n == 1:
        src = src.replace(REASONING_END_OLD, REASONING_END_NEW, 1)
    elif n != 0 or "    def is_reasoning_end(" in src.split("class Qwen3Parser(", 1)[1]:
        raise Drift("Qwen3Parser.is_reasoning_end has an unknown form")
    return src


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (("vllm", "parser/qwen3.py", patch_parser, "parser"),)


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
