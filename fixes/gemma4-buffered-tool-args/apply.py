#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""gemma4-buffered-tool-args: emit gemma4 tool-call arguments once, complete and escape-repaired.

Gemma 4 writes tool-call string arguments between its own <|"|> delimiters, and sometimes also
JSON-escapes the quotes and newlines inside them. The engine-based gemma4 parser passes those
escapes through, so a code argument such as '\\"\\"\\"Docstring.\\"\\"\\"' reaches the client
with literal backslashes and the written file is broken.

Every argument JSON the parser produces goes through _gemma4_arg_converter(raw_args, partial):
partial=True for streaming argument diffs, partial=False for the flush at the end of a call and
for non-streaming extraction. The fix edits that function:

* partial=True returns "", so no argument diff streams mid-call; the flush at the end of the call
  sends the whole argument JSON as one delta. Name and id still stream when the call opens.
* partial=False repairs string values whose escapes are evidently spurious (every quote escaped,
  JSON-encoded multi-line text, or Python that only parses after unescaping) before json.dumps.

The repair is not prefix-stable, which is why arguments are buffered. A call cut off in the middle
of its arguments arrives without arguments instead of with invalid JSON.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "gemma4-buffered-tool-args"
MARKER = f"[gaudi-fix:{FIX_ID}]"

REPAIR = (
    f"""
# ---- {MARKER} escape repair for complete argument values ----
"""
    + r"""import ast as _g4_ast
import re as _g4_re

_G4_ESCAPED_QUOTE = _g4_re.compile(r'\\"')
_G4_ESCAPED_BACKSLASH = _g4_re.compile(r"\\\\")
_G4_UNESCAPE = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}


def _g4_unescape_tolerant(value: str) -> str:
    out: list[str] = []
    i, n = 0, len(value)
    while i < n:
        c = value[i]
        if c == "\\" and i + 1 < n and value[i + 1] in _G4_UNESCAPE:
            out.append(_G4_UNESCAPE[value[i + 1]])
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _g4_all_quotes_escaped(value: str) -> bool:
    if '\\"' not in value:
        return False
    stripped = _G4_ESCAPED_BACKSLASH.sub("", value)
    stripped = _G4_ESCAPED_QUOTE.sub("", stripped)
    return '"' not in stripped


def _g4_json_encoded_multiline(value: str) -> bool:
    return "\\n" in value and "\n" not in value and '\\"' in value


def _g4_broken_python_that_unescapes_clean(value: str, repaired: str) -> bool:
    try:
        _g4_ast.parse(value)
        return False
    except SyntaxError:
        pass
    except Exception:
        return False
    try:
        _g4_ast.parse(repaired)
    except Exception:
        return False
    return True


def _g4_repair_value(value):
    if not isinstance(value, str) or '\\"' not in value:
        return value
    repaired = _g4_unescape_tolerant(value)
    if repaired == value:
        return value
    if not (
        _g4_all_quotes_escaped(value)
        or _g4_json_encoded_multiline(value)
        or _g4_broken_python_that_unescapes_clean(value, repaired)
    ):
        return value
    return repaired


def _g4_repair(obj):
    if isinstance(obj, str):
        return _g4_repair_value(obj)
    if isinstance(obj, dict):
        return {k: _g4_repair(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_g4_repair(v) for v in obj]
    return obj


"""
)

DEF_ANCHOR = '''def _gemma4_arg_converter(raw_args: str, partial: bool) -> str:
    """Convert Gemma4 custom arg format to a JSON string."""
'''
DEF_NEW = (
    REPAIR
    + DEF_ANCHOR
    + f"""    if partial:
        # {MARKER} no mid-call argument diffs: the escape repair is not prefix-stable.
        # The flush at the end of the call sends the whole argument JSON as one delta.
        return ""
"""
)
PARSE_OLD = "    parsed = _parse_gemma4_args(text, partial=partial)\n"
PARSE_NEW = f"    parsed = _g4_repair(_parse_gemma4_args(text, partial=partial))  # {MARKER}\n"


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


def patch_gemma4(src: str, ctx: dict[str, str]) -> str:
    # The engine calls the converter with partial=True for streaming diffs and flushes with
    # partial=False at the end of a call; both must still exist for the buffering to be safe.
    for needle in ("converter(slot.args, True)", "converter(slot.args, False)", "def _flush_arg_converter("):
        require_once(ctx["engine"], needle, "parser/engine/parser_engine.py")
    require_once(src, DEF_ANCHOR, "parser/gemma4.py")
    require_once(src, "\nimport json\n", "parser/gemma4.py")
    start = src.index(DEF_ANCHOR)
    nxt = src.find("\ndef ", start + len(DEF_ANCHOR))
    end = len(src) if nxt < 0 else nxt
    body = src[start:end]
    body = replace_once(body, PARSE_OLD, PARSE_NEW, "_gemma4_arg_converter parse call")
    if "return json.dumps(parsed, ensure_ascii=False)\n" not in body:
        raise Drift("_gemma4_arg_converter no longer returns json.dumps(parsed)")
    body = replace_once(body, DEF_ANCHOR, DEF_NEW, "_gemma4_arg_converter definition")
    return src[:start] + body + src[end:]


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm", "parser/engine/parser_engine.py", None, "engine"),
    ("vllm", "parser/gemma4.py", patch_gemma4, "gemma4"),
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
