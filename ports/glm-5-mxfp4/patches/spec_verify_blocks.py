#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""spec-verify-blocks: per-row paged-attention block lists for speculative verify rows on the HPU runner.

Generic (speculative decoding on vllm-gaudi); a candidate for fixes/.

With speculative decoding the HPU runner flattens each request's 1+k verify tokens into 1+k decode rows. Row j sits
at position context_len + j, but every row gets the same block list, sized for row 0. When a draft row is the first
token of a new KV block, its own block is missing from its list and "last block usage" is applied to the previous
block: the row attends to one token of the previous block and misses the rest and itself. The bonus token and next
draft of that step are then wrong (seen as a greedy answer ending at exactly position 128). Second bug in the same
place: when requests in a batch have different draft counts, the missing slots of a shorter request are padded with
position 0, so their slot mapping writes a padding token's KV into the request's position-0 slot.
The patch gives each verify row the blocks up to its own position and sends padded slots to the pad block. Decode
without speculation (one token per request) is unchanged.

Usage: spec_verify_blocks.py [--check] [--root SITE_PACKAGES]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 source drifted (nothing written).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "glm-5-mxfp4:spec-verify-blocks"
MARKER = f"[{FIX_ID}]"

OLD_BLOCKS = """\
        block_tables_list = []
        for i, n in enumerate(num_blocks):
            seq_block_table = block_table_cpu_tensor[i, :n].tolist()
            assert len(seq_block_table) == n
            block_tables_list.extend([seq_block_table] * num_tokens)
"""

NEW_BLOCKS = """\
        block_tables_list = []
        for i, n in enumerate(num_blocks):
            if num_tokens == 1:
                seq_block_table = block_table_cpu_tensor[i, :n].tolist()
                assert len(seq_block_table) == n
                block_tables_list.append(seq_block_table)
                continue
            # [glm-5-mxfp4:spec-verify-blocks] verify row j sits at position context_len + j: give it the
            # blocks up to and including its own slot (the row-0 list misses a block a draft row starts).
            _n_i = max(1, int(num_tokens_per_req[i]))
            for _j in range(num_tokens):
                _jj = min(_j, _n_i - 1)
                _nj = -(-(int(context_lens[i]) + _jj + 1) // decode_block_size)
                block_tables_list.append(block_table_cpu_tensor[i, :_nj].tolist())
"""

OLD_SLOTS = """\
        slot_mapping[num_decodes:].apply_(lambda _, ds=dummy_slots: next(ds))
"""

NEW_SLOTS = """\
        slot_mapping[num_decodes:].apply_(lambda _, ds=dummy_slots: next(ds))
        if num_tokens > 1:  # [glm-5-mxfp4:spec-verify-blocks] padded verify slots -> the pad block, not position 0
            for _i in range(num_decodes):
                for _j in range(int(num_tokens_per_req[_i]), num_tokens):
                    slot_mapping[_i, _j] = pad_slot_base + _j
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


# locals of _create_decode_input_data the new code reads, bound before the anchors
REQUIRED = (
    "decode_block_size = self.attn_block_size",
    "num_tokens_per_req = num_tokens_per_req + [0]",
    "pad_slot_base = self._PAD_BLOCK_ID * decode_block_size",
)


def patch_runner(src: str, ctx: dict[str, str]) -> str:
    start = src.find("    def _create_decode_input_data(")
    end = src.find("\n    def ", start + 1)
    if start < 0 or end < 0:
        raise Drift("vllm_gaudi/v1/worker/hpu_model_runner.py: no _create_decode_input_data")
    for need in REQUIRED:
        require_once(src[start:end], need, "_create_decode_input_data")
    src = replace_once(src, OLD_BLOCKS, NEW_BLOCKS, "verify-row block lists")
    return replace_once(src, OLD_SLOTS, NEW_SLOTS, "padded verify slots")


UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
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
