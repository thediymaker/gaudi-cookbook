#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""mm-merge-idxcopy: merge multimodal embeddings with nonzero + index_copy_ instead of masked_scatter_.

vllm-gaudi replaces vLLM's _merge_multimodal_embeddings with its own HPU version
(vllm_gaudi/models/utils.py). For a boolean placeholder mask it calls
inputs_embeds.masked_scatter_(is_multimodal.unsqueeze(-1), ...), a broadcast (stride-0) mask that
the Gaudi graph compiler fails to lower at the sizes concurrent multi-image steps reach
("Graph compile failed ... masked_scatter ... synStatus 26"), which stops the engine.
inputs_embeds holds the whole step, so the size grows with images per step, not per request.

The fix turns the mask into row indices with nonzero and copies with index_copy_, the op the
function's integer-index branch already uses. Row placement is identical, so the result is
bit-identical. nonzero has a data-dependent shape, so a new graph is compiled per distinct
number of multimodal rows.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "mm-merge-idxcopy"
MARKER = f"[gaudi-fix:{FIX_ID}]"

DEF_ANCHOR = "\ndef _merge_multimodal_embeddings(\n"
REBIND_ANCHOR = "\nutils._merge_multimodal_embeddings = _merge_multimodal_embeddings\n"

# 3-D inputs_embeds branch
SCATTER_3D = (
    "\n            inputs_embeds.masked_scatter_(is_multimodal.unsqueeze(-1), mm_embeds_flat.to(dtype=input_dtype))\n"
)
INDEX_3D = f"""
            # {MARKER} broadcast masked_scatter_ fails to compile on HPU; copy by row index
            _mm_rows = is_multimodal.reshape(-1).nonzero(as_tuple=False).reshape(-1)
            inputs_embeds.index_copy_(0, _mm_rows, mm_embeds_flat.to(dtype=input_dtype))
"""
# 2-D branch, inside the try whose RuntimeError handler reports a placeholder count mismatch
SCATTER_2D = (
    "\n        inputs_embeds.masked_scatter_(is_multimodal.unsqueeze(-1), mm_embeds_flat.to(dtype=input_dtype))\n"
)
INDEX_2D = f"""
        # {MARKER} broadcast masked_scatter_ fails to compile on HPU; copy by row index
        _mm_rows = is_multimodal.reshape(-1).nonzero(as_tuple=False).reshape(-1)
        if _mm_rows.numel() != len(mm_embeds_flat):
            raise RuntimeError("multimodal placeholder count mismatch")
        inputs_embeds.index_copy_(0, _mm_rows, mm_embeds_flat.to(dtype=input_dtype))
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


def patch_utils(src: str, ctx: dict[str, str]) -> str:
    require_once(src, DEF_ANCHOR, "models/utils.py")
    require_once(src, REBIND_ANCHOR, "models/utils.py")
    start, end = src.index(DEF_ANCHOR), src.index(REBIND_ANCHOR)
    body = src[start:end]
    for old, label in ((SCATTER_3D, "3-D masked_scatter_"), (SCATTER_2D, "2-D masked_scatter_")):
        if body.count(old) != 1:
            raise Drift(f"{label} is not inside _merge_multimodal_embeddings")
    body = replace_once(body, SCATTER_3D, INDEX_3D, "3-D masked_scatter_")
    body = replace_once(body, SCATTER_2D, INDEX_2D, "2-D masked_scatter_")
    if "masked_scatter_(" in body.replace("# inputs_embeds[is_multimodal]", ""):
        raise Drift("another masked_scatter_ call is left in _merge_multimodal_embeddings")
    return src[:start] + body + src[end:]


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (("vllm_gaudi", "models/utils.py", patch_utils, "utils"),)


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
