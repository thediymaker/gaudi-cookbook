#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The DelegatingParser change of vLLM PR #65745 (Ripple support) in the vLLM of vllm-gaudi v0.30.0.

Ripple's reasoning grammar can open <think> after content has started, and the generation prompt does not open
it. With no tool parser configured nothing else consumes reasoning markers, so the reasoning engine has to parse
the whole stream, as the non-streaming path does, even when the prompt (an earlier turn's "</think>") already
closed reasoning. Without this a later turn's "<think>...</think>" streams out as content.

vllm/parser/abstract_parser.py, four edits taken from the PR:

1. __init__: self._reasoning_engine_streams_all = no tool parser and the reasoning parser's
   opens_reasoning_in_content (ripple_hpu.parser sets it on its reasoning adapter; other parsers lack it, so
   getattr(..., False) leaves them unchanged).
2. _in_reasoning_phase: also True while streams_all.
3. parse_delta: never mark reasoning ended while streams_all.
4. _flush_engine_parsers: flush the reasoning engine after reasoning ended while streams_all.

Usage: parser_streams_all.py [--check] [--root DIR_HOLDING_vllm]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 source drifted (nothing written).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

MARKER = "[ripple-65b:parser-streams-all]"
TARGET = ("vllm", "parser/abstract_parser.py")

# (label, old, new); every anchor must match exactly once
EDITS = [
    (
        "init",
        "        ) and (self._tool_parser is None or self._tool_parser.engine_based_streaming)\n",
        "        ) and (self._tool_parser is None or self._tool_parser.engine_based_streaming)\n"
        f"        # {MARKER} Without a tool parser nothing else consumes reasoning markers, so a\n"
        "        # grammar that can open reasoning from content parses the whole stream,\n"
        "        # as the non-streaming path does, even when the prompt closed reasoning.\n"
        "        self._reasoning_engine_streams_all = self._tool_parser is None and getattr(\n"
        '            self._reasoning_parser, "opens_reasoning_in_content", False\n'
        "        )\n",
    ),
    (
        "in_reasoning_phase",
        "        if self._reasoning_parser is None:\n"
        "            return False\n"
        "        return not state.reasoning_ended\n",
        "        if self._reasoning_parser is None:\n"
        "            return False\n"
        f"        return not state.reasoning_ended or self._reasoning_engine_streams_all  # {MARKER}\n",
    ),
    (
        "parse_delta",
        "            if should_transition:\n                state.reasoning_ended = True\n",
        f"            if should_transition and not self._reasoning_engine_streams_all:  # {MARKER}\n"
        "                state.reasoning_ended = True\n",
    ),
    (
        "flush",
        "            if parser is self._reasoning_parser and reasoning_ended:\n                continue\n",
        "            if (\n"
        "                parser is self._reasoning_parser\n"
        "                and reasoning_ended\n"
        f"                and not self._reasoning_engine_streams_all  # {MARKER}\n"
        "            ):\n"
        "                continue\n",
    ),
]


class Drift(Exception):
    pass


def patch(src: str) -> str:
    for label, old, new in EDITS:
        n = src.count(old)
        if n != 1:
            raise Drift(f"anchor {label!r} found {n} times, expected 1")
        src = src.replace(old, new)
    return src


def locate(root: str | None) -> Path | None:
    pkg, rel = TARGET
    if root:
        p = Path(root) / pkg / rel
        return p if p.is_file() else None
    spec = importlib.util.find_spec(pkg)
    for loc in (spec.submodule_search_locations or []) if spec else []:
        if (Path(loc) / rel).is_file():
            return Path(loc) / rel
    return None


def run(check: bool, root: str | None) -> int:
    path = locate(root)
    if path is None:
        print(f"{MARKER} vllm/parser/abstract_parser.py not found", file=sys.stderr)
        return 2
    src = path.read_text()
    if MARKER in src:
        print(f"{MARKER} applied")
        return 0
    try:
        new = patch(src)
        compile(new, str(path), "exec", dont_inherit=True)
    except (Drift, SyntaxError) as e:
        print(f"{MARKER} {e}; source drifted, nothing written", file=sys.stderr)
        return 2
    if check:
        print(f"{MARKER} not applied (anchors match)")
        return 1
    tmp = path.with_name(path.name + ".ripple.tmp")
    tmp.write_text(new)
    os.replace(tmp, path)
    for pyc in (path.parent / "__pycache__").glob(f"{path.stem}.*.pyc"):
        pyc.unlink(missing_ok=True)
    print(f"{MARKER} patched {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report the state without writing")
    ap.add_argument("--root", help="directory holding vllm/ (default: the installed package)")
    args = ap.parse_args()
    return run(args.check, args.root)


if __name__ == "__main__":
    sys.exit(main())
