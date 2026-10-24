#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""mtp-residual-shape: stop the MTP draft layer from mixing rows on HPU decode.

Generic (DeepSeekMTP drafting on vllm-gaudi); a candidate for fixes/.

On HPU the draft pass feeds the MTP layer [N, 1, H] tensors (N verify rows). Inside the decoder layer attention and
MoE work on [N, H] and the fused add-RMSNorm keeps the residual as [N, 1, H]. The MTP layer then ends with a plain
`residual + hidden_states`, which broadcasts to [N, N, H]; the proposer views that as N*N rows and picks rows by
index, so every draft but row 0's mixes row 0's residual with another row's output, across requests too. Target
outputs are never affected (drafts are verified), only acceptance: with k=2 it fell from 0.59 alone to 0.25-0.50
with two requests. The patch reshapes the layer output to the residual's shape before the add (same numel).
Prefill broadcast correctly before and is unchanged.

Usage: mtp_residual_shape.py [--check] [--root SITE_PACKAGES]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 source drifted (nothing written).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "glm-5-mxfp4:mtp-residual-shape"
MARKER = f"[{FIX_ID}]"

OLD = """\
        hidden_states = residual + hidden_states  # pre-final-norm (logits hidden)
"""

NEW = """\
        # [glm-5-mxfp4:mtp-residual-shape] HPU decode: residual [N, 1, H] vs layer output [N, H]; a
        # plain add would broadcast to [N, N, H] and mix rows (and requests) in the draft.
        if hidden_states.shape != residual.shape and hidden_states.numel() == residual.numel():
            hidden_states = hidden_states.reshape(residual.shape)
        hidden_states = residual + hidden_states  # pre-final-norm (logits hidden)
"""


class Drift(Exception):
    """The installed source does not match what this patch expects."""


def replace_once(src: str, old: str, new: str, label: str) -> str:
    n = src.count(old)
    if n != 1:
        raise Drift(f"anchor {label!r} found {n} times, expected 1")
    return src.replace(old, new, 1)


def require_once(src: str, needle: str, label: str) -> None:
    n = src.count(needle)
    if n != 1:
        raise Drift(f"{label}: {needle.strip()!r} found {n} times, expected 1")


def patch_mtp(src: str, ctx: dict[str, str]) -> str:
    return replace_once(src, OLD, NEW, "final residual add")


UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (("vllm", "model_executor/models/deepseek_mtp.py", patch_mtp, "mtp"),)


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
