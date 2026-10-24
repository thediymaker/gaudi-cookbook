#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""glm4-moe-3d-input: Glm4MoE.forward accepts the 3-D hidden states the HPU runner passes.

vllm/model_executor/models/glm4_moe.py opens the MoE block with

    num_tokens, hidden_dim = hidden_states.shape

which assumes [tokens, hidden]. The vllm-gaudi runner drives models with [batch, seq, hidden], so
every GLM-4.5 family MoE model (GLM-4.5, GLM-4.5-Air, GLM-4.5V, GLM-4.6) stops the engine on the
first generated token with "ValueError: too many values to unpack (expected 2)". The server
starts and /health answers before that, so a startup check does not catch it.

The fix flattens to [tokens, hidden] on entry and restores the caller's shape on exit. For 2-D
input it is a no-op.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "glm4-moe-3d-input"
MARKER = f"[gaudi-fix:{FIX_ID}]"

HEAD = """        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
"""
NEW_HEAD = f"""        orig_shape = hidden_states.shape  # {MARKER} the HPU runner passes [batch, seq, hidden]
        hidden_dim = orig_shape[-1]
        hidden_states = hidden_states.reshape(-1, hidden_dim)
"""
TAIL = "        return final_hidden_states.view(num_tokens, hidden_dim)\n"
NEW_TAIL = f"        return final_hidden_states.reshape(orig_shape)  # {MARKER}\n"


class Drift(Exception):
    """The installed source does not match what this fix expects."""


def replace_once(src: str, old: str, new: str, label: str) -> str:
    n = src.count(old)
    if n != 1:
        raise Drift(f"anchor {label!r} found {n} times, expected 1")
    return src.replace(old, new, 1)


def patch_glm4_moe(src: str, ctx: dict[str, str]) -> str:
    head, tail = src.find(HEAD), src.find(TAIL)
    code = [ln.split("#")[0] for ln in src[head + len(HEAD) : tail].splitlines()]
    if head < 0 or tail < head or any("num_tokens" in ln for ln in code):
        raise Drift("Glm4MoE.forward does not have the expected shape")
    src = replace_once(src, HEAD, NEW_HEAD, "forward head")
    return replace_once(src, TAIL, NEW_TAIL, "forward return")


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (("vllm", "model_executor/models/glm4_moe.py", patch_glm4_moe, "glm4_moe"),)


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
