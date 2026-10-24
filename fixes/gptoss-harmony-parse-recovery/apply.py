#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""gptoss-harmony-parse-recovery: a malformed gpt-oss message header must not fail the request.

gpt-oss sometimes starts a tool-call header, abandons it and starts a new one. openai_harmony's
StreamableParser is still inside the first header when the restart arrives and raises HarmonyError
at the second header's <|message|>. HarmonyParser.process_chunk (vllm/parser/harmony.py) feeds
tokens to the parser without error handling, so the whole completion fails with HTTP 500.

The fix feeds each token through a guarded helper. On HarmonyError it first splices the message:
the recipient (to=...) from the abandoned header plus the payload of the restarted one, and
re-parses that with a fresh parser, so the tool call survives. Resyncing at the restart instead
would return a recipient-less commentary message, i.e. prose where a tool call was meant. If the
splice does not parse, the token is dropped and parsing continues; flush() already returns the raw
text for whatever cannot be parsed. A rolling window of the last 512 tokens is kept across message
boundaries, because the abandoned header ends with <|end|>, which completes a message and clears
the per-message token buffer before the error is raised.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "gptoss-harmony-parse-recovery"
MARKER = f"[gaudi-fix:{FIX_ID}]"

_LOOP_HEAD = """        segments: list[Segment] = []
        reasoning_token_count = 0
        for token_id in token_ids:
"""
_LOOP_TAIL = """            channel = self._harmony_parser.current_channel
"""
# The token loop of process_chunk: stock (v0.26.0, v0.30.0), or with the upstream drop-token guard.
LOOP_VARIANTS = (
    (
        "stock",
        _LOOP_HEAD + "            self._harmony_parser.process(token_id)\n" + _LOOP_TAIL,
    ),
    (
        "upstream drop-token guard",
        _LOOP_HEAD
        + """            try:
                self._harmony_parser.process(token_id)
            except HarmonyError:
                continue
"""
        + _LOOP_TAIL,
    ),
)
LOOP_NEW = (
    _LOOP_HEAD
    + f"""            # {MARKER} keep a rolling window across message boundaries and
            # recover from a malformed header instead of failing the request.
            _w = getattr(self, "_gaudi_fix_harmony_window", None)
            if _w is None:
                _w = self._gaudi_fix_harmony_window = []
            _w.append(token_id)
            if len(_w) > 512:
                del _w[:-512]
            if not self._gaudi_fix_process(token_id):
                continue
"""
    + _LOOP_TAIL
)

HELPER_ANCHOR = "    def _poll_completed_message(self) -> Message | None:\n"
HELPER_NEW = (
    f'''    def _gaudi_fix_splice_recipient(self, toks):
        """{MARKER} Recipient from the abandoned header plus the payload of the restarted one."""
        try:
            from vllm.entrypoints.openai.parser.harmony_utils import get_encoding

            enc = get_encoding()
            start = enc.encode("<|start|>", allowed_special="all")[0]
            chan = enc.encode("<|channel|>", allowed_special="all")[0]
            msg = enc.encode("<|message|>", allowed_special="all")[0]
            cons = enc.encode("<|constrain|>", allowed_special="all")[0]
        except Exception:
            return None
        starts = [k for k in range(len(toks)) if toks[k] == start]
        if len(starts) < 2:
            return None
        s1, s2 = starts[-2], starts[-1]
        c1 = next((k for k in range(s1, s2) if toks[k] == chan), None)
        if c1 is None:
            return None
        constrain = [k for k in range(c1, s2) if toks[k] == cons]
        head_end = (constrain[-1] + 1) if constrain else s2
        m2 = next((k for k in range(s2, len(toks)) if toks[k] == msg), None)
        if m2 is None or s1 + 1 >= len(toks):
            return None
        return toks[:s1] + [start, toks[s1 + 1]] + toks[c1:head_end] + toks[m2:]

    def _gaudi_fix_process(self, token_id) -> bool:
        """{MARKER} Feed one token; on a malformed header splice or drop, never raise.

        Returns True when the caller may read parser state for this token.
        """
        try:
            self._harmony_parser.process(token_id)
            return True
        except HarmonyError as exc:
            toks = list(
                getattr(self, "_gaudi_fix_harmony_window", None)
                or (list(self._current_message_tokens) + [token_id])
            )
            repaired = self._gaudi_fix_splice_recipient(toks)
            if repaired is not None:
                fresh = get_streamable_parser_for_assistant()
                try:
                    for t in repaired:
                        fresh.process(t)
                except HarmonyError:
                    pass
                else:
                    # The repaired tokens cover only the current message; earlier messages were
                    # already emitted, so the fresh parser re-emits nothing.
                    self._parser = fresh
                    self._num_processed_messages = 0
                    self._current_message_tokens = list(repaired)
                    self._gaudi_fix_harmony_window = list(repaired)
                    logger.warning("malformed harmony header (%s); repaired by splicing the recipient", exc)
                    return True
            logger.warning("malformed harmony header (%s); dropping one token", exc)
            return False

'''
    + HELPER_ANCHOR
)

# Names the inserted code uses; each must be present in the parser module.
REQUIRED = (
    "from openai_harmony import HarmonyError, Message, Role\n",
    "    get_streamable_parser_for_assistant,\n",
    "logger = init_logger(__name__)\n",
    "        self._current_message_tokens: list[int] = []\n",
    "    def _harmony_parser(self) -> StreamableParser:\n",
    "            self._parser = get_streamable_parser_for_assistant()\n",
)


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


def patch_harmony(src: str, ctx: dict[str, str]) -> str:
    for needle in REQUIRED:
        require_once(src, needle, "parser/harmony.py")
    matches = [(name, old) for name, old in LOOP_VARIANTS if src.count(old) == 1]
    if len(matches) != 1:
        raise Drift("process_chunk token loop has none of the known forms")
    src = replace_once(src, matches[0][1], LOOP_NEW, f"process_chunk token loop ({matches[0][0]})")
    return replace_once(src, HELPER_ANCHOR, HELPER_NEW, "_poll_completed_message")


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (("vllm", "parser/harmony.py", patch_harmony, "harmony"),)


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
