#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""swa-window-context: sliding-window layers read only their window of context in chunked prefill, for every model.

A prefill chunk with cached context (the later chunks of a long prompt, or a prefix-cache hit) gathers the
context's KV blocks and runs fused SDPA over context plus chunk. For sliding-window layers the runner can
pass only the last sliding_window // block_size + 1 context blocks (window_block_list), and the mask builder
re-anchors the window for that list. The runner builds the list only when model_type is "gemma4". Every other
interleaved sliding-window model gathers the whole context on its sliding layers and masks it with a window
as wide as the context, on every chunk.

The fix drops the gemma4 condition at the two places that build the list: where it is sliced from each
request's context blocks, and where it is padded and copied to the device. The attention backend and the
mask builder already handle the list for any model; decode already uses a window block table for every
interleaved model. If the gates were already widened to an explicit list of model types, that form is
replaced the same way.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "swa-window-context"
MARKER = f"[gaudi-fix:{FIX_ID}]"

# The gate as the releases ship it, and as a model port may have widened it.
GATES = (
    'self.interleaved_sliding_window and self._get_model_type() == "gemma4"',
    'self.interleaved_sliding_window and self._get_model_type() in ("gemma4", "muse_glimmer")',
)
SITES = (
    (
        "window block list",
        "        if {gate}:\n            # Keep one extra block",
        f"        if self.interleaved_sliding_window:  # {MARKER} every interleaved model\n"
        "            # Keep one extra block",
    ),
    (
        "window block copy",
        "        if {gate} and target_blocks > 0:\n            window_context_blocks = align_and_pad(",
        f"        if self.interleaved_sliding_window and target_blocks > 0:  # {MARKER}\n"
        "            window_context_blocks = align_and_pad(",
    ),
)


class Drift(Exception):
    """The installed source does not match what this fix expects."""


def patch_runner(src: str, ctx: dict[str, str]) -> str:
    for label, old_tmpl, new in SITES:
        olds = [old_tmpl.format(gate=g) for g in GATES]
        counts = [src.count(o) for o in olds]
        if sum(counts) != 1:
            raise Drift(f"anchor {label!r} found {sum(counts)} times, expected 1")
        src = src.replace(olds[counts.index(1)], new, 1)
    return src


# Not fixed upstream.
UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (("vllm_gaudi", "v1/worker/hpu_model_runner.py", patch_runner, "runner"),)


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
