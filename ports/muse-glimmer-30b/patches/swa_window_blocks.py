#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""swa-window-blocks: Muse Glimmer's sliding layers read only their window in chunked prefill.

vllm-gaudi builds a per-request list of the last `sliding_window // block_size + 1` context blocks for
the sliding-window layers of interleaved models, but only when `model_type == "gemma4"`. Every other
interleaved model (Muse Glimmer has 39 sliding layers with a 2,048-token window and 13 global ones)
hands the sliding layers the full context block list and a window mask as wide as the whole context.
On a long prompt that is chunked into 8K pieces, each chunk then attends over the full context in all
52 layers instead of 13. The patch widens the two gates to ("gemma4", "muse_glimmer"); every other
model keeps the stock path.

Usage: swa_window_blocks.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

PATCH_ID = "muse-glimmer-swa-window-blocks"
MARKER = f"[gaudi-port:{PATCH_ID}]"
TARGET = ("vllm_gaudi", "v1/worker/hpu_model_runner.py")

OLD = 'self.interleaved_sliding_window and self._get_model_type() == "gemma4"'
NEW = 'self.interleaved_sliding_window and self._get_model_type() in ("gemma4", "muse_glimmer")'
SITES = 2  # the window block list for chunked prefill, and its padded device copy
NOTE = f"\n# {MARKER} the interleaved sliding-window prefill gates also cover muse_glimmer.\n"


class Drift(Exception):
    """The installed source does not match what this patch expects."""


def patch(src: str) -> str:
    n = src.count(OLD)
    if n != SITES or NEW in src:
        raise Drift(f"gemma4 window gate found {n} times, expected {SITES}")
    return src.replace(OLD, NEW).rstrip("\n") + "\n" + NOTE


def locate(root: str | None) -> Path | None:
    pkg, rel = TARGET
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
    tmp = path.with_name(path.name + ".gaudi-port.tmp")
    tmp.write_text(text)
    os.chmod(tmp, path.stat().st_mode & 0o7777)
    os.replace(tmp, path)
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(f"{path.stem}.*.pyc"):
            pyc.unlink(missing_ok=True)


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
    write_atomic(path, new)
    print(f"{MARKER} patched {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report the state without writing (0 applied, 1 not applied)")
    ap.add_argument("--root", help="directory holding vllm_gaudi/ (default: the installed package)")
    args = ap.parse_args()
    return run(args.check, args.root)


if __name__ == "__main__":
    sys.exit(main())
