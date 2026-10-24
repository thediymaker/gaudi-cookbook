#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""mxk-hooks: wire the Kimi MXFP4 decode kernels (kimi_hpu.mxk) into vllm-gaudi behind KIMI_MXK.

Two hooks, both inert unless KIMI_MXK=1:

* top of vllm_gaudi/__init__.py: kimi_hpu.mxk.setup_env(). vllm_gaudi is the platform plugin that every vLLM
  process (API server, engine core, workers) imports before it opens a device, and the graph compiler reads
  GC_KERNEL_PATH when the device opens.
* end of vllm_gaudi/ops/hpu_mxfp4.py: kimi_hpu.mxk.install(), which wraps HPUGptOssMxfp4MoEMethod (to keep a
  reference to each layer's packed weights) and VllmMixtureOfExpertsOpMXFP4.forward (decode-sized calls go to
  the kernels, everything else to the stock op).

Usage: mxk_hooks.py [--check] [--root SITE_PACKAGES]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 source drifted (nothing written).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "kimi-k2-7:mxk-hooks"
MARKER = f"[{FIX_ID}]"

INIT_HOOK = f"""# {MARKER} KIMI_MXK=1: register the Kimi MoE kernel library before the first HPU op
try:
    from kimi_hpu.mxk import setup_env as _kimi_mxk_setup_env

    _kimi_mxk_setup_env()
except ImportError as _kimi_mxk_e:  # the platform plugin must still import; install() then keeps the stock op
    print(f"[kimi-mxk] kimi_hpu is not importable: {{_kimi_mxk_e}}")
"""

OPS_ANCHOR = "mxfp4_module.GptOssMxfp4MoEMethod = HPUGptOssMxfp4MoEMethod\n"
OPS_HOOK = f"""
# {MARKER} KIMI_MXK=1: decode-sized MoE calls run on the Kimi MXFP4 TPC kernels (ports/kimi-k2-7)
from kimi_hpu import mxk as _kimi_mxk  # noqa: E402

_kimi_mxk.install()
"""


class Drift(Exception):
    """The installed source does not match what this patch expects."""


def require_once(src: str, needle: str, label: str) -> None:
    n = src.count(needle)
    if n != 1:
        raise Drift(f"{label}: {needle.strip()!r} found {n} times, expected 1")


def patch_init(src: str, ctx: dict[str, str]) -> str:
    require_once(src, "\ndef register():\n", "vllm_gaudi/__init__.py")
    return INIT_HOOK + src


def patch_ops(src: str, ctx: dict[str, str]) -> str:
    require_once(src, OPS_ANCHOR, "vllm_gaudi/ops/hpu_mxfp4.py")
    if not src.endswith(OPS_ANCHOR):
        raise Drift("vllm_gaudi/ops/hpu_mxfp4.py: the method swap is not the last statement")
    return src + OPS_HOOK


UPSTREAM_FIXED = ()

TARGETS = (
    ("vllm_gaudi", "__init__.py", patch_init, "init"),
    ("vllm_gaudi", "ops/hpu_mxfp4.py", patch_ops, "ops"),
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
