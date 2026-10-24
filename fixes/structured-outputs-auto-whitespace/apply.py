#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""structured-outputs-auto-whitespace: allow disable_any_whitespace with the default "auto" backend.

--structured-outputs-config '{"disable_any_whitespace": true}' is refused at startup unless the
backend is named as "xgrammar" or "guidance", because the config validator checks the backend name
and the default is "auto". Naming xgrammar works around the error but loses auto's fallback, so
schemas xgrammar cannot compile then fail with HTTP 400.

xgrammar and guidance already read the flag from the engine config. outlines ignored it; with the
fix it passes an empty whitespace pattern to outlines_core when the flag is set. Every backend that
"auto" can pick then honours the flag, so the validator only rejects lm-format-enforcer.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "structured-outputs-auto-whitespace"
MARKER = f"[gaudi-fix:{FIX_ID}]"

VALIDATOR_OLD = """        if self.disable_any_whitespace and self.backend not in ("xgrammar", "guidance"):
            raise ValueError(
                "disable_any_whitespace is only supported for "
                "xgrammar and guidance backends."
            )
"""
VALIDATOR_NEW = f"""        # {MARKER} "auto" only resolves to xgrammar, guidance or outlines,
        # which all honour this flag.
        if self.disable_any_whitespace and self.backend == "lm-format-enforcer":
            raise ValueError(
                "disable_any_whitespace is not supported for the "
                "lm-format-enforcer backend."
            )
"""

OUTLINES_INIT_OLD = """    def __post_init__(self):
        self.vocabulary = get_outlines_vocabulary(self.tokenizer)
        self.cache = get_outlines_cache()
"""
OUTLINES_INIT_NEW = f"""    def __post_init__(self):
        self.vocabulary = get_outlines_vocabulary(self.tokenizer)
        self.cache = get_outlines_cache()
        # {MARKER} None keeps outlines_core's default whitespace pattern.
        self.whitespace_pattern = (
            ""
            if self.vllm_config.structured_outputs_config.disable_any_whitespace
            else None
        )
"""
OUTLINES_JSON_OLD = """        if request_type == StructuredOutputOptions.JSON:
            regex = json_schema.build_regex_from_schema(grammar_spec)
"""
OUTLINES_JSON_NEW = """        if request_type == StructuredOutputOptions.JSON:
            regex = json_schema.build_regex_from_schema(
                grammar_spec, whitespace_pattern=self.whitespace_pattern
            )
"""


class Drift(Exception):
    """The installed source does not match what this fix expects."""


def replace_once(src: str, old: str, new: str, label: str) -> str:
    n = src.count(old)
    if n != 1:
        raise Drift(f"anchor {label!r} found {n} times, expected 1")
    return src.replace(old, new, 1)


def patch_config(src: str, ctx: dict[str, str]) -> str:
    return replace_once(src, VALIDATOR_OLD, VALIDATOR_NEW, "disable_any_whitespace validator")


def patch_outlines(src: str, ctx: dict[str, str]) -> str:
    if "class OutlinesBackend(StructuredOutputBackend):" not in src:
        raise Drift("backend_outlines.py has no OutlinesBackend")
    src = replace_once(src, OUTLINES_INIT_OLD, OUTLINES_INIT_NEW, "OutlinesBackend.__post_init__")
    return replace_once(src, OUTLINES_JSON_OLD, OUTLINES_JSON_NEW, "outlines JSON schema regex")


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = (
    ("config", 'if self.disable_any_whitespace and self.backend == "lm-format-enforcer":'),
    ("outlines", "whitespace_pattern=self.whitespace_pattern"),
)

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm", "config/structured_outputs.py", patch_config, "config"),
    ("vllm", "v1/structured_output/backend_outlines.py", patch_outlines, "outlines"),
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
