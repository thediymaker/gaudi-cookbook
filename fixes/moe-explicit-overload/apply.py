#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""moe-explicit-overload: call the FP8 per-channel MoE op by its explicit overload.

VllmMixtureOfExpertsOpFP8PerChannel.forward calls torch.ops.hpu.mixture_of_experts through the
overload packet, which tries the registered overloads in order until one binds. Overloads that
declare a float d_scale_hidden_states reject the Tensor scale, and the schema-mismatch message
PyTorch builds formats that tensor, which copies it to the host and synchronizes the device. That
happens for every MoE layer on every eager forward (prefills above the graph capture size) and
every graph capture, and it also stops the host from running ahead of the device.

The fix names the overload each branch binds to: fp8_fused_weights_dynamic for dynamic activation
scales and fp8_fused_weights for static scales. Kernel and arguments are unchanged.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "moe-explicit-overload"
MARKER = f"[gaudi-fix:{FIX_ID}]"

DYN_OLD = """            x_fp8, x_scale = dynamic_quant(x)
            final_hidden_states = torch.ops.hpu.mixture_of_experts(hidden_states=x_fp8,
"""
DYN_NEW = f"""            x_fp8, x_scale = dynamic_quant(x)
            # {MARKER} call the overload by name: the packet tries overloads with a float
            # d_scale_hidden_states first, and the mismatch message formats x_scale (device sync).
            final_hidden_states = torch.ops.hpu.mixture_of_experts.fp8_fused_weights_dynamic(hidden_states=x_fp8,
"""
STATIC_OLD = """            x_fp8 = torch.ops.hpu.cast_to_fp8_v2(x, 1.0 / x_scale, False, False, torch.float8_e4m3fn)[0]
            final_hidden_states = torch.ops.hpu.mixture_of_experts(hidden_states=x_fp8,
"""
STATIC_NEW = """            x_fp8 = torch.ops.hpu.cast_to_fp8_v2(x, 1.0 / x_scale, False, False, torch.float8_e4m3fn)[0]
            final_hidden_states = torch.ops.hpu.mixture_of_experts.fp8_fused_weights(hidden_states=x_fp8,
"""
CLASS_ANCHOR = "class VllmMixtureOfExpertsOpFP8PerChannel("


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


def patch_ops(src: str, ctx: dict[str, str]) -> str:
    require_once(src, CLASS_ANCHOR, "extension/ops.py")
    start = src.index(CLASS_ANCHOR)
    nxt = src.find("\nclass ", start + 1)
    body = src[start : nxt if nxt > 0 else len(src)]
    for old, label in ((DYN_OLD, "dynamic-scale call"), (STATIC_OLD, "static-scale call")):
        if body.count(old) != 1:
            raise Drift(f"{label} is not inside VllmMixtureOfExpertsOpFP8PerChannel")
    src = replace_once(src, DYN_OLD, DYN_NEW, "dynamic-scale call")
    return replace_once(src, STATIC_OLD, STATIC_NEW, "static-scale call")


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = (("ops", "torch.ops.hpu.mixture_of_experts.fp8_fused_weights_dynamic("),)

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (("vllm_gaudi", "extension/ops.py", patch_ops, "ops"),)


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
