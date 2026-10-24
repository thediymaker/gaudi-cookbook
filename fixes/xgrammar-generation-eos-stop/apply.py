#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""xgrammar-generation-eos-stop: let a finished grammar end the turn with the model's real end tokens.

The xgrammar backend builds its TokenizerInfo with TokenizerInfo.from_huggingface(tokenizer,
vocab_size=...), so the grammar's only stop token is tokenizer.eos_token_id. Many chat models end
a turn with a different token that is listed in generation_config.json's eos_token_id (the engine
already stops on those). Once a grammar is complete, for example a structural tag for
tool_choice="required", xgrammar allows only another call or the tokenizer EOS; the real
end-of-turn token is masked, so the model keeps opening new tool calls until max_tokens.

The fix passes stop_token_ids = tokenizer EOS + generation_config eos_token_id to xgrammar. A
grammar can still stop only where it is complete; this only widens which end token is legal there
to the set the engine already treats as end of turn. vLLM later fixed the same problem by passing
each request's stop tokens to the grammar matcher (vLLM #49227); where that is present this fix
reports "not needed".

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "xgrammar-generation-eos-stop"
MARKER = f"[gaudi-fix:{FIX_ID}]"

ANCHOR = """            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                self.tokenizer,
                vocab_size=self.vocab_size,
            )
"""
REPLACEMENT = f"""            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                self.tokenizer,
                vocab_size=self.vocab_size,
                stop_token_ids=_gaudi_fix_generation_stop_ids(self),  # {MARKER}
            )
"""

HELPER = f'''

def _gaudi_fix_generation_stop_ids(backend):
    """{MARKER} Tokenizer EOS plus generation_config eos_token_id, or None for the default."""
    ids = set()
    eos = getattr(backend.tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    try:
        gen_eos = backend.vllm_config.model_config.try_get_generation_config().get("eos_token_id")
    except Exception:
        gen_eos = None
    if isinstance(gen_eos, int):
        gen_eos = [gen_eos]
    for tid in gen_eos or []:
        if isinstance(tid, int):
            ids.add(tid)
    ids = sorted(t for t in ids if 0 <= t < backend.vocab_size)
    if len(ids) > 1:
        logger.info("xgrammar stop tokens: %s", ids)
    return ids or None
'''

REQUIRED = ("\nlogger = init_logger(__name__)\n", "class XgrammarBackend(StructuredOutputBackend):")


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


def patch_backend(src: str, ctx: dict[str, str]) -> str:
    for needle in REQUIRED:
        require_once(src, needle, "backend_xgrammar.py")
    src = replace_once(src, ANCHOR, REPLACEMENT, "TokenizerInfo.from_huggingface in XgrammarBackend")
    return src.rstrip("\n") + "\n" + HELPER


# Text that shows the upstream fix is present (checked only where the marker is absent):
# vLLM #49227 hands each request's stop tokens to the grammar matcher.
UPSTREAM_FIXED = (("backend", "override_stop_tokens=list(stop_token_ids)"),)

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (("vllm", "v1/structured_output/backend_xgrammar.py", patch_backend, "backend"),)


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
