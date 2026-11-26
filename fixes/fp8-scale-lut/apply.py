#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""fp8-scale-lut: halve FP8 weights for the Gaudi 2 scale adjustment through a 256-entry table.

On Gaudi 2 (scale adjustment on) gaudi_weight_wrapper halves every FP8 weight as it is loaded, with
(loaded_weight.float() * 0.5).to(torch.float8_e4m3fn). The tensor is still on the CPU there, and the CPU
float8 casts are slow, so per-expert FP8 MoE checkpoints, where the wrapper runs once per expert slice,
spend most of their load time in this line.

An FP8 value has 256 bit patterns, so the halving is a byte-to-byte map. The table is built once with
the stock expression itself, applied to every byte, so the result is bit-identical to the stock line for
every input (NaN and subnormal rounding included). Applying it is one gather. Scales (the non-FP8 branch)
are unchanged.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "fp8-scale-lut"
MARKER = f"[gaudi-fix:{FIX_ID}]"

HALVE_OLD = """            if loaded_weight.dtype == torch.float8_e4m3fn:
                loaded_weight = (loaded_weight.float() * 0.5).to(torch.float8_e4m3fn)
"""
HALVE_NEW = f"""            if loaded_weight.dtype == torch.float8_e4m3fn:
                loaded_weight = _fp8_half_lut(loaded_weight)  # {MARKER}
"""
HELPER_ANCHOR = "def gaudi_weight_wrapper(weight_loader):\n"
HELPER = f'''_FP8_HALF_LUT = {{}}  # {MARKER} per device


def _fp8_half_lut(w):
    """Bit-identical to (w.float() * 0.5).to(torch.float8_e4m3fn), as one byte-to-byte gather."""
    lut = _FP8_HALF_LUT.get(w.device)
    if lut is None:
        every_byte = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn)
        lut = (every_byte.float() * 0.5).to(torch.float8_e4m3fn).view(torch.uint8).to(w.device)
        _FP8_HALF_LUT[w.device] = lut
    return lut[w.contiguous().view(torch.uint8).long()].view(torch.float8_e4m3fn)


'''


class Drift(Exception):
    """The installed source does not match what this fix expects."""


def replace_once(src: str, old: str, new: str, label: str) -> str:
    n = src.count(old)
    if n != 1:
        raise Drift(f"anchor {label!r} found {n} times, expected 1")
    return src.replace(old, new, 1)


def patch_ops(src: str, ctx: dict[str, str]) -> str:
    if "\nimport torch\n" not in src:
        raise Drift("vllm_gaudi/extension/ops.py does not import torch at module level")
    src = replace_once(src, HALVE_OLD, HALVE_NEW, "FP8 weight halving")
    return replace_once(src, HELPER_ANCHOR, HELPER + HELPER_ANCHOR, "gaudi_weight_wrapper")


# Not fixed upstream.
UPSTREAM_FIXED = ()

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
