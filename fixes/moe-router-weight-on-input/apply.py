#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""moe-router-weight-on-input: honour apply_router_weight_on_input in the HPU fused MoE paths.

Llama 4's MoE builds FusedMoE with apply_router_weight_on_input=True and a top-1 sigmoid router:
the reference computes expert(w * x). The HPU monolithic MoE paths never read the flag and pass
the router weights to the Habana mixture_of_experts op, which applies them to the expert output,
w * expert(x). With SwiGLU experts the two differ (silu(w * a) * (w * b) against
w * silu(a) * b), on every MoE layer, so served outputs drift from the reference model.

The fix scales the expert input by the router weight and passes unit weights to the op for layers
that set apply_router_weight_on_input (top_k must be 1, as upstream requires). Other layers are
unchanged. It covers the unquantized path (apply_monolithic and forward_oot) and the
compressed-tensors W8A8 FP8 and WNA16 paths.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

FIX_ID = "moe-router-weight-on-input"
MARKER = f"[gaudi-fix:{FIX_ID}]"

HELPER = f'''

# ---- {MARKER} ----------------------------------------------------------------
def _gaudi_fix_router_weight_on_input(layer, x, topk_weights):
    """expert(w * x) instead of the op's w * expert(x) for layers that ask for it."""
    if not getattr(layer, "apply_router_weight_on_input", False):
        return x, topk_weights
    if topk_weights.shape[-1] != 1:
        raise ValueError("apply_router_weight_on_input requires top_k == 1")
    x = x * topk_weights.to(x.dtype)
    return x, torch.ones_like(topk_weights)
'''

_FUSED_CALL = """            output = layer.moe_op(
                x,
                topk_ids,
                topk_weights,
                permuted_weights=True,
"""
_CT_CALL = """        output = layer.moe_op(
            x,
            topk_ids.to(torch.int64),
            topk_weights.to(x.dtype),
            permuted_weights={permuted},
"""
# (file key, class, method, call anchor inside that method)
SITES = (
    ("fused", "HPUUnquantizedFusedMoEMethod", "apply_monolithic", _FUSED_CALL),
    ("fused", "HPUUnquantizedFusedMoEMethod", "forward_oot", _FUSED_CALL),
    ("ct", "HPUCompressedTensorsW8A8Fp8MoEMethod", "apply_monolithic", _CT_CALL.format(permuted="True")),
    ("ct", "HPUCompressedTensorsWNA16MoEMethod", "apply_monolithic", _CT_CALL.format(permuted="False")),
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


def _span(src: str, start_re: str, end_re: str, start: int, end: int, label: str) -> tuple[int, int]:
    hits = [m for m in re.finditer(start_re, src[start:end], re.M)]
    if len(hits) != 1:
        raise Drift(f"{label} found {len(hits)} times, expected 1")
    s = start + hits[0].start()
    nxt = re.compile(end_re, re.M).search(src, start + hits[0].end(), end)
    return s, nxt.start() if nxt else end


def _patch_sites(src: str, key: str) -> str:
    for _, cls, method, call in (s for s in SITES if s[0] == key):
        c0, c1 = _span(src, rf"^class {cls}\(", r"^class ", 0, len(src), f"class {cls}")
        m0, m1 = _span(src, rf"^    def {method}\(", r"^    (?:def |@)", c0, c1, f"{cls}.{method}")
        body = src[m0:m1]
        indent = call[: len(call) - len(call.lstrip())]
        hook = f"{indent}x, topk_weights = _gaudi_fix_router_weight_on_input(layer, x, topk_weights)  # {MARKER}\n"
        body = replace_once(body, call, hook + call, f"moe_op call in {cls}.{method}")
        src = src[:m0] + body + src[m1:]
    return src.rstrip("\n") + "\n" + HELPER


def patch_fused(src: str, ctx: dict[str, str]) -> str:
    require_once(src, "\nimport torch\n", "hpu_fused_moe.py")
    return _patch_sites(src, "fused")


def patch_ct(src: str, ctx: dict[str, str]) -> str:
    require_once(src, "\nimport torch\n", "hpu_compressed_tensors.py")
    return _patch_sites(src, "ct")


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm_gaudi", "ops/hpu_fused_moe.py", patch_fused, "fused"),
    ("vllm_gaudi", "ops/hpu_compressed_tensors.py", patch_ct, "ct"),
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
