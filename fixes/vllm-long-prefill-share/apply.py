#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""vllm-long-prefill-share: let long prompts take at most a set share of wall time while others decode.

With chunked prefill, every step that carries a chunk of a long prompt runs that chunk's attention
over all of its context so far. At several hundred thousand tokens of context one chunk can take
seconds, and every other running request gets one decode token per step, so a short chat crawls
while one long prompt prefills. --long-prefill-token-threshold shortens the chunk but cannot help
decode, because a long chunk is still in every step.

This time-shares the scheduler. vLLM already defers in-progress prefill chunks on some steps while
decodes run (defer_prefills, used by DP prefill balancing). The fix adds a second reason to take
that branch: after a step that carried a long chunk and took T seconds, the next
T * (1 - share) / share seconds are decode-only for long prompts, as long as some other request
is decoding or a short request is waiting. New and short requests are never deferred. Off unless
VLLM_LONG_PREFILL_SHARE is set, so applying the fix alone changes nothing.

  VLLM_LONG_PREFILL_SHARE        fraction of wall time long prefills may take while others decode
                                 (0 or unset: off, the stock scheduler)
  VLLM_LONG_PREFILL_MIN_TOKENS   prompt length that counts as long (default 65536)
  VLLM_LONG_PREFILL_SHARE_ASYNC  1: also defer long chunks while one is still in flight, so the
                                 share holds under async scheduling (default 0)

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "vllm-long-prefill-share"
MARKER = f"[gaudi-fix:{FIX_ID}]"

HELPERS = f"""
# {MARKER} Long prompts may take at most VLLM_LONG_PREFILL_SHARE of wall time while other
# requests decode. Off when unset. Step durations are taken where steps complete
# (update_from_output): with async scheduling the scheduler runs a step ahead, so the gap between
# schedule() calls would measure the wrong step.
import os as _lps_os
import time as _lps_time

_LPS_SHARE = float(_lps_os.environ.get("VLLM_LONG_PREFILL_SHARE", "0") or 0)
_LPS_MIN_TOKENS = int(_lps_os.environ.get("VLLM_LONG_PREFILL_MIN_TOKENS", "65536") or 65536)
_LPS_ON = 0.0 < _LPS_SHARE < 1.0
_LPS_ASYNC = (_lps_os.environ.get("VLLM_LONG_PREFILL_SHARE_ASYNC", "0") or "0").strip().lower() in (
    "1", "true", "yes", "on")


def _lps_long_in_flight(sched) -> bool:
    # A long request whose in-flight tokens include prompt tokens: a long chunk was scheduled and
    # has not completed yet (async scheduling runs one step ahead).
    for r in sched.running:
        n = getattr(r, "num_in_flight_tokens", 0)
        if (n > 0 and r.num_prompt_tokens >= _LPS_MIN_TOKENS
                and r.num_computed_tokens - n < r.num_prompt_tokens):
            return True
    return False


def _lps_defer_long(sched) -> bool:
    if not _LPS_ON:
        return False
    st = sched.__dict__.get("_lps_state")
    if (st is None or _lps_time.monotonic() >= st["until"]) and not (_LPS_ASYNC and _lps_long_in_flight(sched)):
        return False
    # Yield to anyone decoding, and to short requests waiting to start: with several long prompts
    # mid-prefill their chunks would otherwise fill every step's token budget.
    return (any(not r.is_prefill_chunk for r in sched.running)
            or any(r.num_prompt_tokens < _LPS_MIN_TOKENS for r in sched.waiting))


def _lps_step_done(sched, scheduler_output) -> None:
    if not _LPS_ON:
        return
    now = _lps_time.monotonic()
    st = sched.__dict__.get("_lps_state")
    if st is None:
        st = sched._lps_state = {{"done": now, "until": 0.0}}
    took = min(now - st["done"], 10.0)  # an idle gap is not a step
    st["done"] = now
    reqs = sched.requests
    if any(n > 1 and rid in reqs and reqs[rid].num_prompt_tokens >= _LPS_MIN_TOKENS
           for rid, n in scheduler_output.num_scheduled_tokens.items()):
        st["until"] = now + took * (1.0 - _LPS_SHARE) / _LPS_SHARE
"""

EDITS = (
    ("module helpers", "logger = init_logger(__name__)\n", "logger = init_logger(__name__)\n" + HELPERS),
    (
        "defer_prefills",
        """        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound
        ) and any(not r.is_prefill_chunk for r in self.running)
""",
        f"""        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound
        ) and any(not r.is_prefill_chunk for r in self.running)
        defer_long = _lps_defer_long(self)  # {MARKER}
""",
    ),
    (
        "running-request deferral",
        """            if defer_prefills and request.is_prefill_chunk:
""",
        f"""            if (defer_prefills and request.is_prefill_chunk) or (
                defer_long and request.is_prefill_chunk
                and request.num_prompt_tokens >= _LPS_MIN_TOKENS
            ):  # {MARKER}
""",
    ),
    (
        "step completion",
        """        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
""",
        f"""        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        _lps_step_done(self, scheduler_output)  # {MARKER}
""",
    ),
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


def patch_scheduler(src: str, ctx: dict[str, str]) -> str:
    require_once(ctx["request"], "self.is_prefill_chunk = False", "vllm/v1/request.py")
    for label, old, new in EDITS:
        src = replace_once(src, old, new, label)
    return src


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm", "v1/request.py", None, "request"),
    ("vllm", "v1/core/sched/scheduler.py", patch_scheduler, "scheduler"),
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
