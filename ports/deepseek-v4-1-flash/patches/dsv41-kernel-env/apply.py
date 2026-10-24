#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""dsv41-kernel-env: register the DeepSeek-V4.1 MXFP4 kernel library before any process opens a Gaudi device.

The graph compiler reads GC_KERNEL_PATH once, when the device opens, so the kernel library must be registered in
every vLLM process (API server, engine core, tensor-parallel workers) before its first HPU op. vllm_gaudi/__init__.py
is the platform plugin that vLLM's plugin loader imports in each of them before the device opens; this puts a call
to dsv41_hpu.mxk.setup_env() at its top. setup_env() does nothing unless DSV41_MXK=1 (dsv41_hpu/mxk.py has what it
sets and why), and an exception in it never breaks the platform import: the model then logs that the kernels are
off and runs the stock op.

DeepSeek-only (it imports the port's plugin), so it lives in the port, not in fixes/. Same contract as a fix:
anchors must match exactly once, nothing is written on drift (exit 2), a marker makes it idempotent, --check
reports without writing (exit 1 = not applied yet).

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "dsv41-kernel-env"
MARKER = f"[gaudi-fix:{FIX_ID}]"

OLD = """import os
import json
import sys
"""
NEW = f"""# {MARKER} DSV41_MXK=1: register the DeepSeek-V4.1 kernel library before the first HPU op
try:
    from dsv41_hpu.mxk import setup_env as _dsv41_mxk_setup_env

    _dsv41_mxk_setup_env()
except Exception as _dsv41_mxk_e:  # never break the platform plugin import
    print(f"[dsv41-mxk] setup_env failed: {{_dsv41_mxk_e}}")
import os
import json
import sys
"""


class Drift(Exception):
    """The installed source does not match what this patch expects."""


def patch_init(src: str, ctx: dict[str, str]) -> str:
    if not src.startswith(OLD) or src.count(OLD) != 1:
        raise Drift("vllm_gaudi/__init__.py does not start with the expected imports")
    return NEW + src[len(OLD) :]


UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (("vllm_gaudi", "__init__.py", patch_init, "init"),)


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
