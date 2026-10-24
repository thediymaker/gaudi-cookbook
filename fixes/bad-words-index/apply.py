#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""bad-words-index: key bad_words to the rows of the selective sampling metadata.

InputBatch.make_selective_sampling_metadata builds SamplingMetadata for only the rows being
sampled (a prefill's logits, or the decode batch) and re-indexes temperature, top-p/top-k,
penalties and generators to those rows. bad_words_token_ids was passed through as the whole
batch's dict, keyed by persistent batch index, and vLLM's apply_bad_words indexes the selected
logits with that key. A bad_words request that is not at batch index 0 then either indexes past
the end of its logits (IndexError, which stops the engine) or has its words banned on another
request's row while its own row is left unfiltered.

The fix re-keys the map to the selected rows, the same way the generators are re-keyed.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

FIX_ID = "bad-words-index"
MARKER = f"[gaudi-fix:{FIX_ID}]"

OLD = """            bad_words_token_ids=self.bad_words_token_ids,
            logitsprocs="""
NEW = f"""            # {MARKER} keyed by the selected row, not the persistent batch index
            bad_words_token_ids={{
                i: self.bad_words_token_ids[req_idx]
                for i, req_idx in enumerate(req_indices) if req_idx in self.bad_words_token_ids
            }},
            logitsprocs="""

SIG = re.compile(r"^    def make_selective_sampling_metadata\(\n", re.M)
NEXT_MEMBER = re.compile(r"^    (?:def |@)", re.M)


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


def method_span(src: str) -> tuple[int, int]:
    """Start and end offsets of make_selective_sampling_metadata's body."""
    sigs = list(SIG.finditer(src))
    if len(sigs) != 1:
        raise Drift(f"make_selective_sampling_metadata found {len(sigs)} times, expected 1")
    start = sigs[0].end()
    nxt = NEXT_MEMBER.search(src, start)
    return start, nxt.start() if nxt else len(src)


def patch_input_batch(src: str, ctx: dict[str, str]) -> str:
    start, end = method_span(src)
    body = src[start:end]
    if "for i, req_idx in enumerate(req_indices)" not in body:
        raise Drift("make_selective_sampling_metadata does not re-key rows by req_indices")
    body = replace_once(body, OLD, NEW, "bad_words_token_ids in make_selective_sampling_metadata")
    return src[:start] + body + src[end:]


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = (("input_batch", "                i: self.bad_words_token_ids[req_idx]\n"),)

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (("vllm_gaudi", "v1/worker/hpu_input_batch.py", patch_input_batch, "input_batch"),)


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
