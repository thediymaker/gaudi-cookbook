#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""mtp-tuple: unpack DeepSeekMTPModel's tuple output in the HPU draft proposer.

Generic (MTP drafting for DeepSeek-style models on vllm-gaudi); a candidate for fixes/.

vLLM's DeepSeekMTP forward returns (pre-norm hidden states for the logits, normed hidden states to recycle); vLLM's
own proposer knows this (model_returns_tuple). vllm-gaudi's HpuEagleProposer hard-codes a single tensor in its "mtp"
branches, so the first MTP proposal fails with AttributeError: 'tuple' object has no attribute 'view'. The patch
takes the eagle-style unpack whenever the model returned a tuple, at all three sites. Element 0 feeds
compute_logits (which applies the shared head norm itself); element 1 is recycled into the next draft step.

Usage: mtp_tuple.py [--check] [--root SITE_PACKAGES]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 source drifted (nothing written).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "glm-5-mxfp4:mtp-tuple"
MARKER = f"[{FIX_ID}]"

OLD1 = """\
        # All MTP related method names are now unified to "mtp"
        if self.method == "mtp":
            last_hidden_states = ret_hidden_states
            hidden_states = last_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states
"""

NEW1 = """\
        # All MTP related method names are now unified to "mtp"
        # [glm-5-mxfp4:mtp-tuple] DeepSeekMTPModel returns (pre-norm, normed): unpack it
        _mtp_tuple = isinstance(ret_hidden_states, tuple)
        if self.method == "mtp" and not _mtp_tuple:
            last_hidden_states = ret_hidden_states
            hidden_states = last_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states
"""

OLD2 = """\
        if self.method == "mtp":
            hidden_states = target_hidden_states.view(-1, target_hidden_states.shape[-1])
        else:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
"""

NEW2 = """\
        if self.method == "mtp" and not _mtp_tuple:  # [glm-5-mxfp4:mtp-tuple]
            hidden_states = target_hidden_states.view(-1, target_hidden_states.shape[-1])
        else:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
"""

OLD3 = """\
            if self.method == "mtp":
                last_hidden_states = ret_hidden_states
                hidden_states = ret_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states
"""

NEW3 = """\
            if self.method == "mtp" and not isinstance(ret_hidden_states, tuple):  # [glm-5-mxfp4:mtp-tuple]
                last_hidden_states = ret_hidden_states
                hidden_states = ret_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states
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


def patch_eagle(src: str, ctx: dict[str, str]) -> str:
    src = replace_once(src, OLD1, NEW1, "first draft forward")
    src = replace_once(src, OLD2, NEW2, "recycled hidden states")
    return replace_once(src, OLD3, NEW3, "per-token draft loop")


UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (("vllm_gaudi", "v1/spec_decode/hpu_eagle.py", patch_eagle, "eagle"),)


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
