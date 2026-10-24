#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""inkling-decode-window: one more decode block for Inkling's sliding-window layers.

Inkling's 35 local layers attend to the last 512 positions. In paged decode the vllm-gaudi runner hands
interleaved sliding-window layers the sequence's last `window // block_size` blocks (4 at block size 128),
and adds one block only for gpt-oss. Depending on where the current token sits in its block, decode then
sees 385 to 512 keys of the window, while prefill sees all 512. This adds the extra block for
model_type inkling_mm_model when INKLING_DECODE_WINDOW_PLUS1=1 (the rel bias in the model masks keys past
the window exactly), and logs the decode window once.

The edit goes before the gpt-oss check and leaves that text alone, so the swa-decode-window fix (which
anchors on it) applies before or after this patch.

Fix contract as in fixes/: every anchor must match once or nothing is written (exit 2); a marker makes a
second run a no-op (exit 0); --check writes nothing (exit 1 = not applied, anchors match).

Usage: runner_decode_window.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

MARKER = "[inkling-small:decode-window]"
TARGET = ("vllm_gaudi", "v1/worker/hpu_model_runner.py")

ANCHOR = """            # Adjust sliding block size for specific model types
            model_type = self._get_model_type()
"""
INSERT = f"""            # {MARKER} Inkling local layers: window // block_size + 1 blocks, as for gpt-oss
            if model_type == "inkling_mm_model" and os.environ.get("INKLING_DECODE_WINDOW_PLUS1", "0") == "1":
                sliding_block_size += 1
            if model_type == "inkling_mm_model" and not getattr(self, "_inkling_window_logged", False):
                self._inkling_window_logged = True
                logger.info("{MARKER} decode sliding_block_size=%d", sliding_block_size)
"""
# The edit uses these module-level names; they must already exist in the runner.
REQUIRED = ("\nimport os\n", "\nlogger = init_logger(")


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


def patch(src: str) -> str:
    for needle in REQUIRED:
        if needle not in src:
            raise ValueError(f"{needle.strip()!r} not found")
    n = src.count(ANCHOR)
    if n != 1:
        raise ValueError(f"anchor found {n} times, expected 1")
    return src.replace(ANCHOR, ANCHOR + INSERT, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    ap.add_argument("--root", help="directory holding vllm_gaudi/ (default: the installed package)")
    args = ap.parse_args()
    path = locate(args.root)
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
    except (ValueError, SyntaxError) as e:
        print(f"{MARKER} {e}; source drifted, nothing written", file=sys.stderr)
        return 2
    if args.check:
        print(f"{MARKER} not applied (anchors match)")
        return 1
    tmp = path.with_name(path.name + ".inkling.tmp")
    tmp.write_text(new)
    os.chmod(tmp, path.stat().st_mode & 0o7777)
    os.replace(tmp, path)
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(f"{path.stem}.*.pyc"):
            pyc.unlink(missing_ok=True)
    print(f"{MARKER} applied now: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
