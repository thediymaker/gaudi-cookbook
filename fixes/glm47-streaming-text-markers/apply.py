#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""glm47-streaming-text-markers: the glm47 parser recognises plain-text tool-call markers when streaming.

GLM-4.7 and the GLM-5 family have added tokens for <tool_call> and </tool_call>, but the model can
also write them as ordinary BPE pieces, and a byte-level structural tag (tool_choice "required" or
a named tool) accepts both spellings. vLLM's streaming parser engine treats a terminal that also
has a token-id form as content once the stream has carried token ids, so in streaming a plain-text
call comes back as raw text in delta.content with no tool_calls and finish_reason "stop", while the
same tokens parse correctly with stream=false.

The fix changes the glm47 parser only, per parser instance, in streaming:
  * a plain-text <tool_call> opens a call after reasoning has ended when the request carries a
    structural tag (the grammar forces calls); free-form chat and reasoning text are unchanged;
  * a plain-text </tool_call> closes an open call;
  * <think> and </think> stay id-only;
  * when the model closes a call whose last value is unterminated, the streamed arguments are
    closed as valid JSON. A call cut off by max_tokens is left as stock streams it.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "glm47-streaming-text-markers"
MARKER = f"[gaudi-fix:{FIX_ID}]"

ANCHOR = """        kwargs.setdefault(
            "parser_engine_config",
            glm47_moe_config(thinking=self.thinking_enabled),
        )
        super().__init__(tokenizer, tools, **kwargs)
"""
REPLACEMENT = (
    ANCHOR
    + f"""        _gaudi_fix_glm47_text_markers(self)  # {MARKER}
"""
)

# {content} and {terminal} are the engine calls for the release line (v0.30 passes token_count).
HELPER = """

# {marker} streaming only: plain-text <tool_call> opens a call after reasoning when the
# structural tag forces calls, plain-text </tool_call> closes an open call, and a call closed
# with an unterminated value gets its streamed arguments closed as valid JSON.
def _gaudi_fix_glm47_text_ok(engine, name):
    if name == "TOOL_START":
        return engine.state == ParserState.CONTENT and engine._gaudi_fix_forced_calls
    if name == "TOOL_END":
        return engine.state in (ParserState.TOOL_NAME, ParserState.TOOL_ARGS)
    return False


def _gaudi_fix_glm47_text_markers(parser) -> None:
    from vllm.parser.engine.incremental_lexer import CONTENT_TERMINAL

    engine = parser._engine
    strict_names = frozenset(engine._token_id_terminal_names)
    engine._gaudi_fix_forced_calls = False

    check_skip = parser._check_skip_tool_parsing

    def _check_skip_tool_parsing(request):
        so = getattr(request, "structured_outputs", None)
        engine._gaudi_fix_forced_calls = getattr(so, "structural_tag", None) is not None
        return check_skip(request)

    def _process_lex_tokens(tokens):
        events = []
        id_mode = engine._ever_had_token_ids
        for tok in tokens:
            name = tok.terminal
            if name == CONTENT_TERMINAL or (
                id_mode and name in strict_names and not _gaudi_fix_glm47_text_ok(engine, name)
            ):
                events.extend({content})
            else:
                events.extend({terminal})
        return events

    flush = parser._flush_arg_converter
    handle_end = parser._handle_tool_end
    closing = [False]

    def _handle_tool_end(event, deltas):
        # event.value holds the "</tool_call>" text when the model closed the call; it is empty
        # for the end that finish() synthesises at max_tokens, which is left unchanged.
        closing[0] = bool(event.value)
        try:
            return handle_end(event, deltas)
        finally:
            closing[0] = False

    def _flush_arg_converter(idx):
        out = flush(idx)
        slot = parser._tool_slots[idx]
        prev = slot.streamed_json
        if out is not None or not closing[0] or not prev or parser._arg_converter is None:
            return out
        try:
            json.loads(prev)
            return out
        except ValueError:
            pass
        try:
            closed = parser._fix_arg_types(parser._arg_converter(slot.args, True), slot.name)
        except (ValueError, TypeError):
            return out
        if closed.startswith(prev) and len(closed) > len(prev):
            slot.streamed_json = closed
            return closed[len(prev):]
        return out

    parser._check_skip_tool_parsing = _check_skip_tool_parsing
    engine._process_lex_tokens = _process_lex_tokens
    parser._handle_tool_end = _handle_tool_end
    parser._flush_arg_converter = _flush_arg_converter
"""

# finish() synthesises the end of an open call without a value; the JSON-close guard relies on it.
_FINISH_END = """                events.append(
                    SemanticEvent(
                        EventType.TOOL_CALL_END,
                        tool_index=self.tool_index,
                    )
                )
"""
_PARSER_ENGINE = (
    "self._engine = StreamingParserEngine(",
    "    def _check_skip_tool_parsing(\n",
    "    def _flush_arg_converter(self, idx: int) -> str | None:",
    "        remaining = self._flush_arg_converter(idx)\n",
    "                    self._handle_tool_end(event, tool_call_deltas)\n",
)

# The override reproduces the engine's _process_lex_tokens, so each release line's exact body and
# hooks must be present. (name, engine checks, lexer checks, content call, terminal call)
VARIANTS = (
    (
        "v0.26",
        (
            "self._token_id_terminal_names: frozenset[str] = frozenset(",
            "    def _process_lex_tokens(self, tokens: list[LexToken]) -> list[SemanticEvent]:\n"
            "        events: list[SemanticEvent] = []\n"
            "        strict = self._token_id_terminal_names if self._ever_had_token_ids else None\n"
            "        for tok in tokens:\n"
            "            if tok.terminal == CONTENT_TERMINAL or (strict and tok.terminal in strict):\n"
            "                events.extend(self._on_content(tok.value))\n"
            "            else:\n"
            "                events.extend(self._on_terminal(tok.terminal, tok.value))\n"
            "        return events\n",
            "    def _on_terminal(self, terminal: str, value: str) -> list[SemanticEvent]:",
            "    def _on_content(self, text: str) -> list[SemanticEvent]:",
            _FINISH_END,
        ),
        ('CONTENT_TERMINAL = "__CONTENT__"',),
        "engine._on_content(tok.value)",
        "engine._on_terminal(name, tok.value)",
    ),
    (
        "v0.30",
        (
            "self._token_id_terminal_names: frozenset[str] = frozenset(",
            "    def _process_lex_tokens(self, tokens: list[LexToken]) -> list[SemanticEvent]:\n"
            "        events: list[SemanticEvent] = []\n"
            "        strict = self._token_id_terminal_names if self._ever_had_token_ids else None\n"
            "        for tok in tokens:\n"
            "            if tok.terminal == CONTENT_TERMINAL or (strict and tok.terminal in strict):\n"
            "                events.extend(self._on_content(tok.value, tok.token_count))\n"
            "            else:\n"
            "                events.extend(\n"
            "                    self._on_terminal(tok.terminal, tok.value, tok.token_count)\n"
            "                )\n"
            "        return events\n",
            "    def _on_terminal(\n"
            "        self, terminal: str, value: str, token_count: int = 0\n"
            "    ) -> list[SemanticEvent]:",
            "    def _on_content(self, text: str, token_count: int = 0) -> list[SemanticEvent]:",
            _FINISH_END,
        ),
        ('CONTENT_TERMINAL = "__CONTENT__"', "    token_count: int = 0\n"),
        "engine._on_content(tok.value, tok.token_count)",
        "engine._on_terminal(name, tok.value, tok.token_count)",
    ),
)


class Drift(Exception):
    """The installed source does not match what this fix expects."""


def replace_once(src: str, old: str, new: str, label: str) -> str:
    n = src.count(old)
    if n != 1:
        raise Drift(f"anchor {label!r} found {n} times, expected 1")
    return src.replace(old, new, 1)


def _variant_matches(ctx: dict[str, str], engine_checks, lexer_checks) -> bool:
    return (
        all(ctx["engine"].count(n) == 1 for n in engine_checks)
        and all(ctx["lexer"].count(n) == 1 for n in lexer_checks)
        and all(ctx["parser_engine"].count(n) == 1 for n in _PARSER_ENGINE)
    )


def patch_parser(src: str, ctx: dict[str, str]) -> str:
    hits = [v for v in VARIANTS if _variant_matches(ctx, v[1], v[2])]
    if len(hits) != 1:
        raise Drift(f"parser engine matches {len(hits)} known shapes, expected 1")
    _name, _e, _l, content, terminal = hits[0]
    for needle in ("class Glm47MoeParser(ParserEngine):", "    ParserState,\n", "\nimport json\n"):
        if src.count(needle) != 1:
            raise Drift(f"glm47_moe.py: {needle.strip()!r} not found exactly once")
    src = replace_once(src, ANCHOR, REPLACEMENT, "Glm47MoeParser.__init__")
    return src.rstrip("\n") + "\n" + HELPER.format(marker=MARKER, content=content, terminal=terminal)


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm", "parser/engine/streaming_parser_engine.py", None, "engine"),
    ("vllm", "parser/engine/parser_engine.py", None, "parser_engine"),
    ("vllm", "parser/engine/incremental_lexer.py", None, "lexer"),
    ("vllm", "parser/glm47_moe.py", patch_parser, "parser"),
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
