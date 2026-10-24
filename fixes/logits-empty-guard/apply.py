#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""logits-empty-guard: skip the tensor-parallel logits gather when there are no rows to gather.

With sync scheduling and no structured output, the HPU runner gives a prompt chunk that does not
finish its prompt zero logits rows, so compute_logits runs on a [0, hidden] tensor. The
tensor-parallel all-gather in LogitsProcessor._get_logits then runs over an empty tensor, which
the Gaudi collective library rejects as a fatal error (ValidateSyncInputTensors: tensor_data is
empty, in hccl allgather), and the engine core stops. Any prompt longer than
--max-num-batched-tokens triggers it on a TP > 1 server.

The fix returns an empty [0, org_vocab_size] tensor of the logits dtype before the gather when
the projected logits have no rows. Every tensor-parallel rank sees the same empty shape, so all
ranks skip the collective together. Non-empty batches are unchanged.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "logits-empty-guard"
MARKER = f"[gaudi-fix:{FIX_ID}]"

_HEAD = """        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
"""
_GATHER = (
    # v0.26.0: always gathers
    """        # Gather logits for TP
        logits = self._gather_logits(logits)
""",
    # v0.30.0: gathers only when the lm_head is split
    """        # Gather logits for TP
        if lm_head.tp_size > 1:
            logits = self._gather_logits(logits)
""",
)
GUARD = f"""        # {MARKER} no rows (a prompt chunk with no logits): skip the TP gather,
        # which fails on an empty tensor. Every rank has the same shape, so all skip it.
        if logits.numel() == 0:
            return logits.new_empty((*logits.shape[:-1], self.org_vocab_size))
"""

REQUIRED = (
    "        self.org_vocab_size = org_vocab_size or vocab_size\n",
    "    def _get_logits(\n",
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


def patch_logits(src: str, ctx: dict[str, str]) -> str:
    for needle in REQUIRED:
        require_once(src, needle, "logits_processor.py")
    matches = [g for g in _GATHER if src.count(g) == 1]
    if len(matches) != 1:
        raise Drift("the TP gather in _get_logits has none of the known forms")
    gather = matches[0]
    head_at, gather_at = src.index(_HEAD, src.index("    def _get_logits(\n")), src.index(gather)
    if not 0 < gather_at - head_at < 400:
        raise Drift("the TP gather does not follow the lm_head projection in _get_logits")
    return replace_once(src, gather, GUARD + gather, "TP gather in _get_logits")


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (("vllm", "model_executor/layers/logits_processor.py", patch_logits, "logits"),)


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
