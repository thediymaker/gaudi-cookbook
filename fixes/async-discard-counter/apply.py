#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""async-discard-counter: use vLLM's async_tokens_to_discard counter in the HPU async scheduler.

The vLLM commit that vllm-gaudi v0.26.0 tracks replaced the Request.discard_latest_async_tokens
boolean with async_tokens_to_discard, a counter of stale in-flight output frames to drop after a
forced preemption. HPUAsyncScheduler._update_request_with_output still reads the old attribute in
the branch that drops tokens produced while a request has no output placeholders (a partial
prefill chunk, or a request resumed after preemption). The first request that reaches that branch
raises AttributeError and the engine stops.

The fix matches vLLM's AsyncScheduler: the dropped frame consumes one pending discard. Releases
that already read the attribute defensively report "not needed".

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "async-discard-counter"
MARKER = f"[gaudi-fix:{FIX_ID}]"

OLD = """            if request.discard_latest_async_tokens:
                request.discard_latest_async_tokens = False
            return [], False
"""
NEW = f"""            # {MARKER} vLLM counts stale frames in async_tokens_to_discard
            # (discard_latest_async_tokens no longer exists); this dropped frame consumes one.
            if getattr(request, "async_tokens_to_discard", 0) > 0:
                request.async_tokens_to_discard -= 1
            return [], False
"""
GUARD = "    def _update_request_with_output(self, request: Request, new_token_ids: list[int])"


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


def patch_scheduler(src: str, ctx: dict[str, str]) -> str:
    require_once(src, GUARD, "hpu_async_scheduler.py")
    require_once(ctx["request"], "self.async_tokens_to_discard = 0", "vllm/v1/request.py")
    return replace_once(src, OLD, NEW, "partial-chunk discard branch")


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = (("scheduler", 'if getattr(request, "discard_latest_async_tokens", False):'),)

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm", "v1/request.py", None, "request"),
    ("vllm_gaudi", "v1/core/sched/hpu_async_scheduler.py", patch_scheduler, "scheduler"),
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
