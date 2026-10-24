#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""rope-cpu-cache: build rotary-embedding cos/sin tables on the CPU.

vLLM builds each RoPE table (_compute_cos_sin_cache) on the device the model is created on. On
Gaudi, fp32 cos/sin of large arguments are inaccurate: the fastest rotary frequency has inverse
frequency 1, so its angle is the position itself in radians, and past roughly 1e5 the device
result is effectively random. Models served with contexts beyond about 128K positions then lose
fine positional resolution in that frequency, and exact copying from long contexts (codes,
identifiers) fails.

The fix wraps _compute_cos_sin_cache of RotaryEmbeddingBase and of every subclass, including
those defined later (vLLM's scaling variants, vllm-gaudi's HPU classes), so the table is computed
under a CPU default device and then moved to the device that was active. Only table construction
changes. vllm-gaudi v0.30.0 already builds the tables on the CPU, so there the fix reports
"not needed".

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "rope-cpu-cache"
MARKER = f"[gaudi-fix:{FIX_ID}]"

REQUIRED = (
    "\nimport torch\n",
    "\nclass RotaryEmbeddingBase(",
    "    def _compute_cos_sin_cache(self) -> torch.Tensor:\n",
)
PATCH = f"""

# --- {MARKER} ---------------------------------------------------------------
# On Gaudi, device fp32 cos/sin of large arguments are inaccurate, so RoPE tables are built on
# the CPU and moved to the active device afterwards.
import functools as _rope_cpu_functools


def _rope_cpu_cache(fn):
    if getattr(fn, "_rope_cpu_cache", False):
        return fn

    @_rope_cpu_functools.wraps(fn)
    def _compute_cos_sin_cache(self, *args, **kwargs):
        dev = torch.get_default_device()
        with torch.device("cpu"):
            out = fn(self, *args, **kwargs)
        if isinstance(out, torch.Tensor) and out.device != dev:
            out = out.to(dev)
        return out

    _compute_cos_sin_cache._rope_cpu_cache = True
    return _compute_cos_sin_cache


def _rope_cpu_wrap(cls):
    fn = cls.__dict__.get("_compute_cos_sin_cache")
    if fn is not None:
        cls._compute_cos_sin_cache = _rope_cpu_cache(fn)
    for sub in cls.__subclasses__():
        _rope_cpu_wrap(sub)


def _rope_cpu_init_subclass(cls, **kwargs):
    super(RotaryEmbeddingBase, cls).__init_subclass__(**kwargs)
    _rope_cpu_wrap(cls)


_rope_cpu_wrap(RotaryEmbeddingBase)
RotaryEmbeddingBase.__init_subclass__ = classmethod(_rope_cpu_init_subclass)
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


def patch_base(src: str, ctx: dict[str, str]) -> str:
    for needle in REQUIRED:
        require_once(src, needle, "rotary_embedding/base.py")
    if "__init_subclass__" in src:
        raise Drift("RotaryEmbeddingBase already customizes __init_subclass__")
    return src.rstrip("\n") + "\n" + PATCH


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = (("hpu_rope", "RotaryEmbeddingBase.__init__ = _rotary_base_init_on_cpu"),)

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm_gaudi", "ops/hpu_rotary_embedding.py", None, "hpu_rope"),
    ("vllm", "model_executor/layers/rotary_embedding/base.py", patch_base, "base"),
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
