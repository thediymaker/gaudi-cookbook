#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""sampling-rows-refresh: re-copy the device sampling rows after the input batch reorders itself.

The HPU input batch keeps each row's sampling parameters twice, a CPU copy and a device copy.
make_selective_sampling_metadata reads the device copy (temperature, top_p, top_k, min_p and the
three penalties) and refreshes it from the CPU copy only when skip_copy is False, which the runner
passes only on steps where a request joined or left. swap_states, which the runner calls after
that refresh (decodes first, multi-token decodes last), swaps only the CPU copies. A row that
changes owner on a step where nobody joined or left keeps the previous owner's parameters on the
device until the next join or leave: a greedy request is sampled at its neighbour's temperature,
a sampled request turns greedy or loses its top_k/top_p, and penalties use another request's
values and a prompt-token tensor in the old row order.

The fix marks the batch when swap_states moves a row. The next make_selective_sampling_metadata
call first re-copies every row's temperature, top_p, top_k and min_p (and the penalties when any
request uses one) as whole tensors (fixed shapes, so no new graphs), drops the cached prompt-token
tensor, and then runs the stock code. refresh_sampling_metadata clears the mark. Steps without a
swap cost nothing extra.

The fix is appended to the module and wraps the InputBatch methods, so it composes with source
edits to the same file.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "sampling-rows-refresh"
MARKER = f"[gaudi-fix:{FIX_ID}]"

# Each must appear exactly once: the methods this fix wraps, the stock skip_copy branch it relies
# on, and the names the resync uses.
ANCHORS = (
    "class InputBatch:",
    "    def swap_states(self, i1: int, i2: int) -> None:",
    "    def refresh_sampling_metadata(self):",
    "    def make_selective_sampling_metadata(",
    "        if not skip_copy:\n"
    "            async_h2d_update(self.temperature_cpu_tensor, self.temperature, req_indices)",
    "    def _invalidate_prompt_token_ids_cache(self):",
    "from vllm_gaudi.utils import async_h2d_copy, async_h2d_update",
)
# Swaps that must stay CPU-only (if swap_states ever moves the device rows too, this fix is no
# longer needed and must be re-evaluated rather than stacked).
CPU_ONLY_SWAPS = (
    "        self.temperature_cpu[i1], self.temperature_cpu[i2] =\\\n",
    "        self.top_k_cpu[i1], self.top_k_cpu[i2] =\\\n",
)
DEVICE_PAIRS = (
    ("temperature_cpu_tensor", "temperature"),
    ("top_p_cpu_tensor", "top_p"),
    ("top_k_cpu_tensor", "top_k"),
    ("min_p_cpu_tensor", "min_p"),
)
PENALTY_PAIRS = (
    ("frequency_penalties_cpu_tensor", "frequency_penalties"),
    ("presence_penalties_cpu_tensor", "presence_penalties"),
    ("repetition_penalties_cpu_tensor", "repetition_penalties"),
)

BLOCK = f"""

# --- {MARKER} ---------------------------------------------------------------
# swap_states moves only the CPU copies of the per-row sampling parameters, so after a swap the
# device rows are re-copied before the next sampling call reads them.
_SAMPLING_ROWS_DEVICE = {DEVICE_PAIRS!r}
_SAMPLING_ROWS_PENALTIES = {PENALTY_PAIRS!r}


def _sampling_rows_resync(self):
    for src, dst in _SAMPLING_ROWS_DEVICE:
        async_h2d_copy(getattr(self, src), getattr(self, dst))
    if not self.no_penalties:
        for src, dst in _SAMPLING_ROWS_PENALTIES:
            async_h2d_copy(getattr(self, src), getattr(self, dst))
    self._invalidate_prompt_token_ids_cache()
    self._sampling_rows_moved = False


def _sampling_rows_install(cls):
    orig = {{
        name: getattr(cls, name)
        for name in ("swap_states", "refresh_sampling_metadata", "make_selective_sampling_metadata")
    }}

    def swap_states(self, i1, i2):
        orig["swap_states"](self, i1, i2)
        if i1 != i2:
            self._sampling_rows_moved = True

    def refresh_sampling_metadata(self):
        orig["refresh_sampling_metadata"](self)
        self._sampling_rows_moved = False

    def make_selective_sampling_metadata(self, *args, **kwargs):
        if getattr(self, "_sampling_rows_moved", False):
            _sampling_rows_resync(self)
        return orig["make_selective_sampling_metadata"](self, *args, **kwargs)

    for fn in (swap_states, refresh_sampling_metadata, make_selective_sampling_metadata):
        fn.__qualname__ = "InputBatch." + fn.__name__
        setattr(cls, fn.__name__, fn)
    return orig


_SAMPLING_ROWS_ORIG = _sampling_rows_install(InputBatch)
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


def patch_input_batch(src: str, ctx: dict[str, str]) -> str:
    for needle in ANCHORS + CPU_ONLY_SWAPS:
        require_once(src, needle, "hpu_input_batch.py")
    for cpu, dev in DEVICE_PAIRS + PENALTY_PAIRS:
        if f"self.{cpu} = " not in src or f"self.{dev} = torch." not in src:
            raise Drift(f"tensor pair {cpu} -> {dev} not found")
    start = src.index("    def swap_states(self, i1: int, i2: int) -> None:")
    end = src.find("\n    def ", start + 1)
    swap_body = src[start : end if end > 0 else len(src)]
    if "self.temperature[" in swap_body or "self.top_k[" in swap_body:
        raise Drift("swap_states already swaps the device rows")
    return src + BLOCK


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = ()

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
