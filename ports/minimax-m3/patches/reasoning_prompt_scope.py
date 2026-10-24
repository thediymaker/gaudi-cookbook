#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Scope the minimax_m3 reasoning parser's prompt check to the current assistant turn.

Symptom: with "stream": true, MiniMax-M3's reasoning trace arrives in `content` with the raw <mm:think> marker
and `reasoning` stays empty. The same request without streaming returns a clean `reasoning` field.

Cause: M3's chat template injects a system block that describes the thinking markers in prose ("wrap your
reasoning in <mm:think></mm:think> tags ..."). Both markers are single tokens, so every prompt already holds an
opener followed by a closer before the model generates anything. vLLM asks the reasoning parser whether
reasoning has already ended in the prompt; MiniMaxM3ReasoningParser.is_reasoning_end scans the whole id list,
finds that closer and answers True. The streaming path then never extracts reasoning, and the serving layer
also tells structured outputs that reasoning is over. Non-streaming parses only the generated text, which is
why it is unaffected. thinking_mode "enabled" hides the bug because that template variant ends the prompt with
an opener.

Fix: every M3 turn starts with the role token "]~b]", so the scan starts after the last one. The default
prompt then has an empty scope (reasoning not ended), the "enabled" prompt has just the prefilled opener, and
mid-generation the scope is the generation, as before. If "]~b]" is not exactly one round-trippable token on
the loaded tokenizer, scoping turns itself off and the parser behaves as stock.

The bug is in vLLM's parser (present on v0.26.0, v0.30.0 and main) and not specific to Gaudi.

Usage: reasoning_prompt_scope.py [--check] [--root DIR_HOLDING_vllm]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 source drifted (nothing written).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

MARKER = "[minimax-m3:reasoning-scope]"
TARGET = ("vllm", "reasoning/minimax_m3_reasoning_parser.py")

INIT_OLD = "        self._end_token_ids = self._encode_marker(self.end_token)\n"
INIT_NEW = INIT_OLD + f"        self._role_open_token_ids = self._resolve_role_open_ids()  # {MARKER}\n"

END_OLD = """    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        start_index = self._rfind_token_sequence(input_ids, self._start_token_ids)"""

END_NEW = f'''    # {MARKER} M3 opens every turn with "]~b]<role>", and the system block the template injects quotes
    # both thinking markers, so marker scans over a prompt must start at the last role opener.
    _ROLE_OPEN_MARKER = "]~b]"

    def _resolve_role_open_ids(self) -> tuple[int, ...]:
        """Role-opener ids, or () to turn scoping off on an unexpected tokenizer."""
        try:
            ids = self._encode_marker(self._ROLE_OPEN_MARKER)
            if len(ids) != 1 or self._decode_text(ids) != self._ROLE_OPEN_MARKER:
                return ()
        except Exception:
            return ()
        return ids

    def _current_turn_token_ids(self, input_ids: Sequence[int]) -> Sequence[int]:
        role_ids = getattr(self, "_role_open_token_ids", ())
        if not role_ids:
            return input_ids
        index = self._rfind_token_sequence(input_ids, role_ids)
        if index < 0:
            return input_ids
        return list(input_ids)[index + len(role_ids) :]

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        input_ids = self._current_turn_token_ids(input_ids)
        start_index = self._rfind_token_sequence(input_ids, self._start_token_ids)'''

EDITS = [("end token ids in __init__", INIT_OLD, INIT_NEW), ("is_reasoning_end", END_OLD, END_NEW)]


class Drift(Exception):
    pass


def patch(src: str) -> str:
    for label, old, new in EDITS:
        n = src.count(old)
        if n != 1:
            raise Drift(f"anchor {label!r} found {n} times, expected 1")
        src = src.replace(old, new, 1)
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
        print(f"{MARKER} {'/'.join(TARGET)} not found", file=sys.stderr)
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
    tmp = path.with_name(path.name + ".port.tmp")
    tmp.write_text(new)
    os.chmod(tmp, path.stat().st_mode & 0o7777)
    os.replace(tmp, path)
    for pyc in (path.parent / "__pycache__").glob(f"{path.stem}.*.pyc"):
        pyc.unlink(missing_ok=True)
    print(f"{MARKER} patched {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report the state without writing (0 applied, 1 not)")
    ap.add_argument("--root", help="directory holding vllm/ (default: the installed package)")
    args = ap.parse_args()
    return run(args.check, args.root)


if __name__ == "__main__":
    sys.exit(main())
