#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""grammar-device-mask: apply the structured-output bitmask on the device, without a host round trip.

Beam's response grammar constrains every chat turn, so every decode step carries a grammar bitmask.
vllm-gaudi v0.30 applies it by copying the logits to the host (`logits.cpu()`, a device sync),
masking them with xgrammar's CPU kernel and copying them back. The sync ends every grammar step:
the host cannot prepare and launch the next step while the device runs this one, so a step costs
host time plus device time. Measured on Beam (8 cards, 4 chat streams, 2K-token prompts): the
grammar steps took 49.5 ms instead of 42 ms, and aggregate decode was 12% below the same load
without a grammar.

Here the bitmask the runner has already sorted into logits rows (rows without a grammar are all
ones, which is exactly what the CPU path leaves unmasked) is copied to the device and unpacked
there: bit j of row i is (bitmask[i, j // 32] >> (j % 32)) & 1, and columns j < min(V, 32 * words)
whose bit is 0 are set to -inf, as xgrammar's kernel does. The CPU path's fp32 round trip is exact
for bf16 and fp32 logits, so the result is the same tensor bit for bit (tests/test_patches.py
checks the inserted code against xgrammar's kernel).

Both places that apply a bitmask are covered: the runner's own apply_grammar_bitmask and the
row-owner path that the cookbook fix so-row-owners adds (in the v0.30.0 profile, applied first).

Switch: BEAM_GRAMMAR_DEVICE_MASK (default 1); 0 runs vllm-gaudi's CPU path unchanged.

Usage: grammar_device_mask.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

MARKER = "[beam-1-0:grammar-device-mask]"
SO_FIX = "[gaudi-fix:so-row-owners]"

STOCK_OLD = """        # Copy async to device as tensor.
        grammar_bitmask = torch.from_numpy(sorted_bitmask).to(logits.device, non_blocking=True)
"""
STOCK_NEW = (
    f"""        if _BEAM_GRAMMAR_DEVICE_MASK:  # {MARKER}
            return _beam_apply_bitmask_device(logits, sorted_bitmask)

"""
    + STOCK_OLD
)

ROWS_OLD = """    if not out_indices:
        return
    index_tensor = None if len(out_indices) == logits.shape[0] else torch.tensor(out_indices, dtype=torch.int32)
"""
ROWS_NEW = f"""    if not out_indices:
        return
    if _BEAM_GRAMMAR_DEVICE_MASK:  # {MARKER}
        return _beam_apply_bitmask_device(logits, sorted_bitmask)
    index_tensor = None if len(out_indices) == logits.shape[0] else torch.tensor(out_indices, dtype=torch.int32)
"""

HELPER = f'''

# ---- {MARKER} ----------------------------------------------------------------------
_BEAM_GRAMMAR_DEVICE_MASK = os.environ.get("BEAM_GRAMMAR_DEVICE_MASK", "1") == "1"


def _beam_apply_bitmask_device(logits, bitmask):
    """In place: logits[i, j] = -inf where bit j of row i of the int32 bitmask [B, W] is 0, for
    j < min(V, 32 * W) (xgrammar's apply_token_bitmask_inplace semantics), on the logits' device.
    `bitmask` (numpy) has one row per logits row; rows without a grammar are all ones."""
    rows, vocab = logits.shape
    words = bitmask.shape[1]
    n = min(vocab, 32 * words)
    bm = torch.from_numpy(bitmask).to(device=logits.device, dtype=torch.int32)
    shifts = torch.arange(32, dtype=torch.int32, device=logits.device)
    bits = torch.bitwise_and(torch.bitwise_right_shift(bm.unsqueeze(-1), shifts), 1).reshape(rows, 32 * words)
    blocked = bits[:, :n] == 0
    if n == vocab:
        logits.masked_fill_(blocked, float("-inf"))
    else:
        logits[:, :n].masked_fill_(blocked, float("-inf"))
'''


class Drift(Exception):
    """The installed source does not match what this patch expects."""


def replace_once(src: str, old: str, new: str, label: str) -> str:
    n = src.count(old)
    if n != 1:
        raise Drift(f"anchor {label!r} found {n} times, expected 1")
    return src.replace(old, new, 1)


def patch_runner(src: str, ctx: dict[str, str]) -> str:
    if SO_FIX not in src:
        raise Drift("so-row-owners is not applied; run fixes/apply_all.py --profile v0.30.0 first")
    if "\nimport os\n" not in src or "\nimport torch\n" not in src:
        raise Drift("module-level os / torch imports not found")
    src = replace_once(src, STOCK_OLD, STOCK_NEW, "apply_grammar_bitmask device copy")
    src = replace_once(src, ROWS_OLD, ROWS_NEW, "so-row-owners bitmask by rows")
    return src.rstrip("\n") + "\n" + HELPER


UPSTREAM_FIXED = ()

TARGETS = (("vllm_gaudi", "v1/worker/hpu_model_runner.py", patch_runner, "runner"),)


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
