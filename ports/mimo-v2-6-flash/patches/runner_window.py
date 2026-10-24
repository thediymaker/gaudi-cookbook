#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exact sliding windows for model type mimo_v2 in the vllm-gaudi v0.26.0 HPU runner.

MiMo-V2's 39 sliding layers attend to the last 128 tokens, the current one included, plus a learned sink. The port's
attention (mimo_v2_hpu/attention.py) computes that exactly from window block lists the runner builds only for
some model types:

1. Prefill window lists (2 sites). With cached or chunked context, the runner builds a per-sequence window block
   list only for model type gemma4; every other model gathers the whole context. The gate becomes
   ("gemma4", "mimo_v2").
2. Decode window. fixes/swa-decode-window makes the window exact for gpt-oss: one extra window block, and the oldest
   block's usage stored as -lbu ("slots [lbu, block) valid"). This turns the same path on for mimo_v2. Stock decode
   hands a sliding layer window // block_size blocks, which for window == block size == 128 is one block: 1 to 128
   visible tokens depending on the position. MiMo has sinks, so a fully masked oldest block keeps the softmax finite.

Every change is gated on the model type, so other models on the image are unchanged. Needs fixes/swa-decode-window
applied first (install.sh applies the release profile before this).

Usage: runner_window.py [--check] [--root DIR_HOLDING_vllm_gaudi]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 source drifted (nothing written).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

MARKER = "[mimo-v2-6-flash:runner-window]"
REQUIRES = "[gaudi-fix:swa-decode-window]"
RUNNER = ("vllm_gaudi", "v1/worker/hpu_model_runner.py")

# (label, old, new, expected hits)
EDITS = [
    (
        "prefill window gate",
        'self._get_model_type() == "gemma4"',
        'self._get_model_type() in ("gemma4", "mimo_v2")',
        2,
    ),
    (
        "decode window block count",
        '            if model_type is not None and model_type in ["gpt_oss"]:\n',
        '            if model_type is not None and model_type in ["gpt_oss", "mimo_v2"]:\n',
        1,
    ),
    (
        "decode exact window",
        '            _exact_window = model_type == "gpt_oss" and self.sliding_window % decode_block_size == 0\n',
        f'            _exact_window = model_type in ("gpt_oss", "mimo_v2") and (  # {MARKER}\n'
        "                self.sliding_window % decode_block_size == 0)\n",
        1,
    ),
]


class Drift(Exception):
    pass


def patch(src: str) -> str:
    if REQUIRES not in src:
        raise Drift("fixes/swa-decode-window is not applied (run the release profile first)")
    for label, old, new, hits in EDITS:
        n = src.count(old)
        if n != hits:
            raise Drift(f"anchor {label!r} found {n} times, expected {hits}")
        src = src.replace(old, new)
    return src


def locate(root: str | None) -> Path | None:
    pkg, rel = RUNNER
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
        print(f"{MARKER} hpu_model_runner.py not found", file=sys.stderr)
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
    tmp = path.with_name(path.name + ".mimo.tmp")
    tmp.write_text(new)
    os.replace(tmp, path)
    for pyc in (path.parent / "__pycache__").glob(f"{path.stem}.*.pyc"):
        pyc.unlink(missing_ok=True)
    print(f"{MARKER} patched {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report the state without writing")
    ap.add_argument("--root", help="directory holding vllm_gaudi/ (default: the installed package)")
    args = ap.parse_args()
    return run(args.check, args.root)


if __name__ == "__main__":
    sys.exit(main())
