#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""swa-decode-window: exact sliding window in paged decode for gpt-oss.

gpt-oss sliding layers attend to the last `window` tokens, the current one included (keys k with
k > q - window). For those layers the HPU decode path takes the sequence's last
window // block_size + 1 KV blocks and leaves every filled slot of all of them visible, so with
block size equal to the window (128) a sliding layer sees between window + 1 and 2 * window
tokens, depending on where the current token sits in its block.

When the window is a multiple of the block size and a sequence's window block list is full
length, the oldest block's usage is stored as -lbu (lbu = usage of the sequence's last block), and
the decode bias reads a negative usage -u as "slots [u, block_size) are valid". Those are exactly
the window positions inside the oldest block, so the visible set becomes (q - window, q].
Positive usages build the same mask as before, so other models and paths are unchanged.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "swa-decode-window"
MARKER = f"[gaudi-fix:{FIX_ID}]"

WINDOW_HEAD = """            if model_type is not None and model_type in ["gpt_oss"]:
                sliding_block_size += 1

            window_block_tables = [block_table[-sliding_block_size:] for block_table in block_tables_list]
"""
WINDOW_HEAD_NEW = f"""            if model_type is not None and model_type in ["gpt_oss"]:
                sliding_block_size += 1
            # {MARKER} with window % block_size == 0, mask the oldest window block down to the window
            _exact_window = model_type == "gpt_oss" and self.sliding_window % decode_block_size == 0

            window_block_tables = [block_table[-sliding_block_size:] for block_table in block_tables_list]
"""
# The window-list call that follows, per release layout.
WINDOW_CALLS = (
    (
        "v0.26.0",
        """                    padded_batch_size * num_tokens,
                    block_size=decode_block_size)
""",
        """                    padded_batch_size * num_tokens,
                    block_size=decode_block_size,
                    head_mask_full_len=sliding_block_size if _exact_window else None)
""",
    ),
    (
        "v0.30.0",
        """                    padded_batch_size * num_tokens,
                    block_size=decode_block_size,
                    force_non_contiguous=True)
""",
        """                    padded_batch_size * num_tokens,
                    block_size=decode_block_size,
                    force_non_contiguous=True,
                    head_mask_full_len=sliding_block_size if _exact_window else None)
""",
    ),
)
SIGNATURES = (
    (
        "    def get_habana_paged_attn_buffers(self, block_tables, slot_mapping, batch_size, block_size=None):\n",
        "    def get_habana_paged_attn_buffers(self, block_tables, slot_mapping, batch_size, block_size=None,\n"
        "                                      head_mask_full_len=None):\n",
    ),
    (
        "                                      block_size=None,\n"
        "                                      force_non_contiguous=False):\n",
        "                                      block_size=None,\n"
        "                                      force_non_contiguous=False,\n"
        "                                      head_mask_full_len=None):\n",
    ),
)
USAGE_OLD = (
    "        block_usage = [[block_size] * (len(bt) - 1) + [lbu] "
    "for bt, lbu in zip(block_tables, last_block_usage) if bt]\n"
)
USAGE_NEW = (
    USAGE_OLD
    + f"""        if head_mask_full_len:  # {MARKER} oldest block of a full window: slots [lbu, bs) only
            block_usage = [([-u[-1]] + u[1:]) if len(u) == head_mask_full_len and len(u) > 1 else u
                           for u in block_usage]
"""
)
MASK_OLD = """        mask = torch.arange(0, block_size, device=device, dtype=torch.int32).unsqueeze(0)
        mask = mask >= block_usage.unsqueeze(-1)
"""
MASK_NEW = f"""        mask = torch.arange(0, block_size, device=device, dtype=torch.int32).unsqueeze(0)
        usage = block_usage.unsqueeze(-1)
        mask = torch.where(usage >= 0, mask >= usage, mask < -usage)  # {MARKER} -u: slots [u, bs) valid
"""


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


def patch_runner(src: str, ctx: dict[str, str]) -> str:
    require_once(src, WINDOW_HEAD, "gpt-oss window block count")
    head_at = src.index(WINDOW_HEAD)
    # the window-list call is the one right after the block count, in one of the known layouts
    matched = [(old, new) for _, old, new in WINDOW_CALLS if 0 < src.find(old, head_at) - head_at < 900]
    if len(matched) != 1:
        raise Drift("window block list call has an unknown layout")
    old, new = matched[0]
    at = src.find(old, head_at)
    src = src[:at] + new + src[at + len(old) :]
    src = replace_once(src, WINDOW_HEAD, WINDOW_HEAD_NEW, "gpt-oss window block count")
    sigs = [(old, new) for old, new in SIGNATURES if src.count(old) == 1]
    if len(sigs) != 1:
        raise Drift("get_habana_paged_attn_buffers signature has an unknown layout")
    src = src.replace(sigs[0][0], sigs[0][1], 1)
    src = replace_once(src, USAGE_OLD, USAGE_NEW, "block usage list")
    return replace_once(src, MASK_OLD, MASK_NEW, "decode block bias")


# Text that shows the upstream fix is present (checked only where the marker is absent).
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
