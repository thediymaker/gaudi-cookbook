#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""glm45v-box-tokens: the glm45 reasoning parser drops GLM-4.5V's answer box markers.

GLM-4.5V wraps short final answers in <|begin_of_box|> ... <|end_of_box|>. vLLM maps
--reasoning-parser glm45 onto the unified GLM-4.7 parser (vllm/parser/glm47_moe.py), which knows
nothing about those markers, so they reach the client in "content", for example
"<|begin_of_box|>Paris<|end_of_box|>". skip_special_tokens does not help: with a reasoning parser
active the text keeps special tokens so the parser can match <think> and </think>.

The fix registers the two markers as parser terminals with transitions that emit nothing, in the
content and reasoning states, the same way the parser already swallows a stray </think>. Models
that never emit the markers (GLM-4.7) are unaffected.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "glm45v-box-tokens"
MARKER = f"[gaudi-fix:{FIX_ID}]"

TERMINALS = """        terminals={
            **reasoning_terminals,
            "TOOL_START": TOOL_CALL_START,
"""
NEW_TERMINALS = f"""        terminals={{
            **reasoning_terminals,
            "BOX_START": "<|begin_of_box|>",  # {MARKER}
            "BOX_END": "<|end_of_box|>",
            "TOOL_START": TOOL_CALL_START,
"""
TRANSITIONS = """        transitions={
            **reasoning_transitions,
"""
NEW_TRANSITIONS = f"""        transitions={{
            **reasoning_transitions,
            # {MARKER} GLM-4.5V answer box markers are consumed, never emitted
            (ParserState.CONTENT, "BOX_START"): Transition(ParserState.CONTENT, ()),
            (ParserState.CONTENT, "BOX_END"): Transition(ParserState.CONTENT, ()),
            (ParserState.REASONING, "BOX_START"): Transition(ParserState.REASONING, ()),
            (ParserState.REASONING, "BOX_END"): Transition(ParserState.REASONING, ()),
"""


class Drift(Exception):
    """The installed source does not match what this fix expects."""


def replace_once(src: str, old: str, new: str, label: str) -> str:
    n = src.count(old)
    if n != 1:
        raise Drift(f"anchor {label!r} found {n} times, expected 1")
    return src.replace(old, new, 1)


def patch_parser(src: str, ctx: dict[str, str]) -> str:
    if "def glm47_moe_config(" not in src:
        raise Drift("glm47_moe_config not found")
    src = replace_once(src, TERMINALS, NEW_TERMINALS, "terminals")
    return replace_once(src, TRANSITIONS, NEW_TRANSITIONS, "transitions")


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = (("parser", "<|begin_of_box|>"),)

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (("vllm", "parser/glm47_moe.py", patch_parser, "parser"),)


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
