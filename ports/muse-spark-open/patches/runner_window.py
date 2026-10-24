#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exact sliding windows for model type muse_spark in the vllm-gaudi v0.30.0 HPU runner.

Muse Spark's 46 sliding layers attend to the last 2048 tokens, the current one included
(keys k with q - 2048 < k <= q). Three things in the runner stand between it and that mask:

1. Window gate (2 sites). Prefill builds a per-sequence window block list only for model type
   gemma4; every other model gathers the whole context for its sliding layers. The gate becomes
   ("gemma4", "muse_spark"). Gemma 4's image mask and its other checks keep their own test.
2. Decode window. The generic fix swa-decode-window makes the window exact for gpt-oss: one extra
   window block, and the oldest block's usage stored as -lbu ("slots [lbu, block) valid"). This
   turns the same path on for muse_spark. When the last block is full the oldest block lies
   wholly outside the window; Muse Spark has no attention sinks, so a block with every slot
   masked would turn the blockwise softmax into NaN (max = -inf). For muse_spark that block is
   dropped from the list instead (drop_masked_head).
3. Prefill lower bound (1 + 2 sites). With cached context the past-key bound keeps
   q - p <= window, i.e. window + 1 keys including the current one. For muse_spark the bound
   drops the "- 1"; other models keep it.

Needs fixes/swa-decode-window applied first (install.sh applies the release profile before this).

Usage: runner_window.py [--check] [--root DIR_HOLDING_vllm_gaudi]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 source drifted (nothing
written).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

MARKER = "[muse-spark-open:runner-window]"
REQUIRES = "[gaudi-fix:swa-decode-window]"
RUNNER = ("vllm_gaudi", "v1/worker/hpu_model_runner.py")

# (label, old, new, expected hits)
EDITS = [
    (
        "window gate",
        'self.interleaved_sliding_window and self._get_model_type() == "gemma4"',
        'self.interleaved_sliding_window and self._get_model_type() in ("gemma4", "muse_spark")',
        2,
    ),
    (
        "decode window block count",
        '            if model_type is not None and model_type in ["gpt_oss"]:\n',
        '            if model_type is not None and model_type in ["gpt_oss", "muse_spark"]:\n',
        1,
    ),
    (
        "decode exact window",
        '            _exact_window = model_type == "gpt_oss" and self.sliding_window % decode_block_size == 0\n',
        '            _exact_window = model_type in ("gpt_oss", "muse_spark") and '
        "self.sliding_window % decode_block_size == 0\n",
        1,
    ),
    (
        "decode window call",
        "                    head_mask_full_len=sliding_block_size if _exact_window else None)\n",
        "                    head_mask_full_len=sliding_block_size if _exact_window else None,\n"
        f'                    drop_masked_head=model_type == "muse_spark")  # {MARKER}\n',
        1,
    ),
    (
        "paged attn buffers signature",
        "                                      head_mask_full_len=None):\n",
        "                                      head_mask_full_len=None,\n"
        "                                      drop_masked_head=False):\n",
        1,
    ),
    (
        "drop the fully masked head block",
        "        last_block_usage = [slot[0] % block_size + 1 for slot in slot_mapping]\n"
        "        block_groups = [[i] * len(bt) for i, bt in enumerate(block_tables)]\n",
        "        last_block_usage = [slot[0] % block_size + 1 for slot in slot_mapping]\n"
        f"        if head_mask_full_len and drop_masked_head:  # {MARKER}\n"
        "            # last block full: the oldest block of a full window list is outside the window\n"
        "            block_tables = [bt[1:] if (len(bt) == head_mask_full_len and len(bt) > 1 and i < len(last_block_usage)\n"
        "                                       and last_block_usage[i] == block_size) else bt\n"
        "                            for i, bt in enumerate(block_tables)]\n"
        "        block_groups = [[i] * len(bt) for i, bt in enumerate(block_tables)]\n",
        1,
    ),
    (
        # the metadata processor's __init__ (the runner sets self.sliding_window elsewhere, without dtype)
        "prefill bound offset",
        "        self.dtype = vllm_config.model_config.dtype\n"
        "        self.sliding_window = vllm_config.model_config.get_sliding_window()\n",
        "        self.dtype = vllm_config.model_config.dtype\n"
        "        self.sliding_window = vllm_config.model_config.get_sliding_window()\n"
        '        self._swa_lb_off = 0 if getattr(vllm_config.model_config.hf_config, "model_type", None) == '
        f'"muse_spark" else 1  # {MARKER}\n',
        1,
    ),
    (
        "prefill bound",
        "window_size + torch.arange(seq_len, device=device) - 1",
        "window_size + torch.arange(seq_len, device=device) - self._swa_lb_off",
        2,
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
    tmp = path.with_name(path.name + ".muse.tmp")
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
