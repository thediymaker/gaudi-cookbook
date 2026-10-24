#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""beam-sliding-window: exact 512-token sliding windows for Beam in the vllm-gaudi v0.30 runner.

Beam's 43 sliding layers attend to the last `window` tokens, the current one included (keys k with
q - window < k <= q, as Hugging Face's sliding-window causal mask). vllm-gaudi serves interleaved
sliding windows for Gemma 4, and the cookbook fix swa-decode-window makes paged decode exact for
gpt-oss; this patch extends both to model type "beam" in vllm_gaudi/v1/worker/hpu_model_runner.py:

1. Window gate (2 sites). Prefill builds per-sequence window block lists only for model type
   "gemma4"; every other model gathers the whole context for its sliding layers. The gate becomes
   ("gemma4", "beam"). Gemma 4's image-prefix mask keeps its own "gemma4" check.

2. Exact decode window. Decode takes window // block + 1 blocks for beam as for gpt-oss, and
   swa-decode-window's exact mask (the oldest block's usage stored as -lbu) is turned on for beam.
   When the last block is full (lbu == block) the oldest block lies wholly outside the window. That
   block would be masked completely, and with no attention sinks (gpt-oss has them, Beam does not)
   a fully masked block turns the blockwise softmax into NaN, so for beam it is dropped from the
   list instead.

3. Prefill window bound with cached context (1 + 2 sites). With context from earlier chunks or
   prefix-cache hits, the bias keeps past key p iff p > ctx - window + i - 1, i.e. window + 1 keys
   including the query itself. For beam the "- 1" goes, so prefill sees exactly the window; other
   models keep their bound.

Needs swa-decode-window (in the v0.30.0 profile) applied first. Requires window % block == 0
(512 and 128 for Beam). Every anchor must match exactly once (the gate twice), or nothing is written.

Usage: beam_sliding_window.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

MARKER = "[beam-1-0:sliding-window]"
SWA_FIX = "[gaudi-fix:swa-decode-window]"

GATE_OLD = 'self.interleaved_sliding_window and self._get_model_type() == "gemma4"'
GATE_NEW = 'self.interleaved_sliding_window and self._get_model_type() in ("gemma4", "beam")'

DEC_OLD = f"""            if model_type is not None and model_type in ["gpt_oss"]:
                sliding_block_size += 1
            # {SWA_FIX} with window % block_size == 0, mask the oldest window block down to the window
            _exact_window = model_type == "gpt_oss" and self.sliding_window % decode_block_size == 0
"""
DEC_NEW = f"""            if model_type is not None and model_type in ["gpt_oss", "beam"]:  # {MARKER}
                sliding_block_size += 1
            # {SWA_FIX} with window % block_size == 0, mask the oldest window block down to the window
            _exact_window = model_type in ("gpt_oss", "beam") and self.sliding_window % decode_block_size == 0
"""

DROP_OLD = """        last_block_usage = [slot[0] % block_size + 1 for slot in slot_mapping]
        block_groups = [[i] * len(bt) for i, bt in enumerate(block_tables)]
"""
DROP_NEW = f"""        last_block_usage = [slot[0] % block_size + 1 for slot in slot_mapping]
        if head_mask_full_len and self._get_model_type() == "beam":  # {MARKER}
            # last block full: the oldest block of a full window list is outside the window; no sinks
            # to keep a fully masked block finite, so drop it
            block_tables = [bt[1:] if (len(bt) == head_mask_full_len and len(bt) > 1 and i < len(last_block_usage)
                                       and last_block_usage[i] == block_size) else bt
                            for i, bt in enumerate(block_tables)]
        block_groups = [[i] * len(bt) for i, bt in enumerate(block_tables)]
"""

LB_INIT_OLD = """        self.dtype = vllm_config.model_config.dtype
        self.sliding_window = vllm_config.model_config.get_sliding_window()
"""
LB_INIT_NEW = LB_INIT_OLD + (
    "        # " + MARKER + " beam's window includes the query: no extra past key with cached context\n"
    '        self._swa_lb_off = 0 if getattr(vllm_config.model_config.hf_config, "model_type", None) == "beam" else 1\n'
)
LB_USE_OLD = "window_size + torch.arange(seq_len, device=device) - 1"
LB_USE_NEW = "window_size + torch.arange(seq_len, device=device) - self._swa_lb_off"


class Drift(Exception):
    """The installed source does not match what this patch expects."""


def replace_n(src: str, old: str, new: str, n: int, label: str) -> str:
    hits = src.count(old)
    if hits != n:
        raise Drift(f"anchor {label!r} found {hits} times, expected {n}")
    return src.replace(old, new)


def patch_runner(src: str, ctx: dict[str, str]) -> str:
    if SWA_FIX not in src:
        raise Drift("swa-decode-window is not applied; run fixes/apply_all.py --profile v0.30.0 first")
    src = replace_n(src, GATE_OLD, GATE_NEW, 2, "window gate")
    src = replace_n(src, DEC_OLD, DEC_NEW, 1, "decode window blocks")
    src = replace_n(src, DROP_OLD, DROP_NEW, 1, "paged attention buffers")
    src = replace_n(src, LB_INIT_OLD, LB_INIT_NEW, 1, "metadata processor init")
    return replace_n(src, LB_USE_OLD, LB_USE_NEW, 2, "prefill window bound")


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
