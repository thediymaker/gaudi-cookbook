#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Keep non-finite values out of the KV cache (VLLM_M3_KV_FINITE=1).

Attention on HPU reads more than a request's own keys: padded batch rows and padded block lists point at the
shared padding block, and masked positions are removed by adding a large negative bias to their scores. A
non-finite K or V written anywhere therefore spreads: NaN plus any bias is NaN, and NaN times a zero
probability is NaN. A MiniMax-M3 server with block-sparse attention that reads other sequences' blocks under a
mask was seen to go from one bad write to every request returning token 0 until restart.

With VLLM_M3_KV_FINITE=1, VLLMKVCache.forward replaces non-finite K/V values with 0 before they are written
(one torch.where per layer and step, on 16- and 32-bit inputs; an FP8 KV cache is not guarded). Off by
default: unset, the write is the stock one.

Usage: kv_finite_writes.py [--check] [--root DIR_HOLDING_vllm_gaudi]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 source drifted (nothing written).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

MARKER = "[minimax-m3:kv-finite]"
TARGET = ("vllm_gaudi", "extension/utils.py")

WRITE_OLD = """        # In cross-attention kv cache forward inputs are None in decode
        # We don't want to store them in the cache in such case
        if input is not None:
            cache.index_copy_(0, slot_mapping, input)
        return cache
"""
WRITE_NEW = f"""        # In cross-attention kv cache forward inputs are None in decode
        # We don't want to store them in the cache in such case
        if input is not None:
            if _M3_KV_FINITE and input.is_floating_point() and input.element_size() > 1:
                # {MARKER} a non-finite K/V would reach every request that reads this block masked
                input = torch.where(torch.isfinite(input), input, torch.zeros_like(input))
            cache.index_copy_(0, slot_mapping, input)
        return cache
"""
CLASS_ANCHOR = "class VLLMKVCache(torch.nn.Module):\n"
SWITCH = f"""# {MARKER} VLLM_M3_KV_FINITE=1 keeps non-finite values out of the KV cache.
_M3_KV_FINITE = os.environ.get("VLLM_M3_KV_FINITE", "0").strip().lower() in ("1", "true", "yes", "on")


"""


class Drift(Exception):
    pass


def patch(src: str) -> str:
    for label, needle in (("import os", "\nimport os\n"), ("import torch", "\nimport torch\n")):
        if needle not in src:
            raise Drift(f"{label} not found at module level")
    for label, old, new in (
        ("VLLMKVCache.forward", WRITE_OLD, WRITE_NEW),
        ("class VLLMKVCache", CLASS_ANCHOR, SWITCH + CLASS_ANCHOR),
    ):
        n = src.count(old)
        if n != 1:
            raise Drift(f"anchor {label!r} found {n} times, expected 1")
        src = src.replace(old, new, 1)
    return src


def locate(root: str | None) -> Path | None:
    pkg, rel = TARGET
    if root:
        p = Path(root) / pkg / rel
        return p if p.is_file() else None
    spec = importlib.util.find_spec(pkg)
    for loc in (spec.submodule_search_locations or []) if spec else []:
        if (Path(loc) / rel).is_file():
            return Path(loc) / rel
    return None


def run(check: bool, root: str | None) -> int:
    path = locate(root)
    if path is None:
        print(f"{MARKER} {'/'.join(TARGET)} not found", file=sys.stderr)
        return 2
    src = path.read_text()
    if MARKER in src:
        print(f"{MARKER} applied")
        return 0
    try:
        new = patch(src)
        compile(new, str(path), "exec", dont_inherit=True)
    except (Drift, SyntaxError) as e:
        print(f"{MARKER} {e}; source drifted, nothing written", file=sys.stderr)
        return 2
    if check:
        print(f"{MARKER} not applied (anchors match)")
        return 1
    tmp = path.with_name(path.name + ".port.tmp")
    tmp.write_text(new)
    os.chmod(tmp, path.stat().st_mode & 0o7777)
    os.replace(tmp, path)
    for pyc in (path.parent / "__pycache__").glob(f"{path.stem}.*.pyc"):
        pyc.unlink(missing_ok=True)
    print(f"{MARKER} patched {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report the state without writing (0 applied, 1 not)")
    ap.add_argument("--root", help="directory holding vllm_gaudi/ (default: the installed package)")
    args = ap.parse_args()
    return run(args.check, args.root)


if __name__ == "__main__":
    sys.exit(main())
