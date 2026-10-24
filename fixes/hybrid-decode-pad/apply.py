#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""hybrid-decode-pad: keep decode padding out of live KV blocks on hybrid models.

On hybrid models (GDN, linear attention or Mamba layers next to attention) vLLM sizes the
attention KV block so that one attention page holds one recurrent state, for example 512, 640,
896 or 1152 tokens. The HPU attention kernel works in smaller kernel blocks, so
initialize_kv_cache splits every KV block into k kernel blocks, allocates num_blocks * k + 1
kernel blocks (the last one is a spare), and the attention block tables hold kernel-block ids.
The pad ids are still set from the KV-block count (_PAD_BLOCK_ID = num_blocks), and kernel block
num_blocks is not the spare: it lies inside KV block num_blocks // k, a live block of the pool.
Decode batches are padded up to bucket sizes, so every odd-sized decode batch writes the padding
rows' K/V over context tokens of whichever request owns that block, in every attention layer, and
with prefix caching a cached copy of the block is corrupted for later hits too.

The fix wraps initialize_kv_cache: when an attention group's KV block is split into kernel
blocks, the pad ids move to the last kernel block of the attention tensors, which is the spare
block after the pool. Models whose KV block equals the kernel block are unchanged (one log
line). KV-cache initialisation fails with RuntimeError if the layout is split but there is no
spare kernel block outside the pool. The Mamba pad is not touched.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "hybrid-decode-pad"
MARKER = f"[gaudi-fix:{FIX_ID}]"

# Each must occur exactly once: they pin the mechanism this fix corrects.
ANCHORS = (
    "class HPUModelRunner(",
    "    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:",
    # the spare kernel block after the pool (block splitting for hybrid models)
    "get_kv_cache_shape(num_kernel_blocks + 1,",
    # the pad ids, set from the KV-block count
    "        self._PAD_BLOCK_ID = num_blocks\n        self._PAD_SLOT_ID = num_blocks * self.attn_block_size\n",
    # decode padding rows write to kernel block _PAD_BLOCK_ID
    "        pad_slot_base = self._PAD_BLOCK_ID * decode_block_size\n",
)

HELPERS = f'''

# ---- {MARKER} ----------------------------------------------------------------------------
# Decode padding must write to the spare kernel block, not to kernel block num_blocks, which is
# inside a live KV block when the attention KV block is split into kernel blocks.
def _gaudi_fix_hybrid_pad_kv(runner, layer_name):
    kv = runner.vllm_config.compilation_config.static_forward_context[layer_name].kv_cache
    if isinstance(kv, list) and len(kv) == 1 and isinstance(kv[0], (tuple, list)):
        kv = kv[0]
    return kv


def _gaudi_fix_hybrid_pad(runner):
    from vllm.v1.kv_cache_interface import AttentionSpec, EncoderOnlyAttentionSpec
    cfg = runner.kv_cache_config
    kb = int(runner.attn_block_size)
    groups = [g for g in cfg.kv_cache_groups
              if isinstance(g.kv_cache_spec, AttentionSpec)
              and not isinstance(g.kv_cache_spec, EncoderOnlyAttentionSpec)]
    split = sorted({{int(g.kv_cache_spec.block_size) for g in groups}} - {{kb}})
    if not split:
        logger.info("{MARKER} inert: attention KV block size equals the kernel block size %d", kb)
        return
    skip = getattr(runner, "runner_only_attn_layers", set())
    rows = set()
    for g in groups:
        for ln in g.layer_names:
            if ln in skip:
                continue
            rows.add(int(_gaudi_fix_hybrid_pad_kv(runner, ln)[0].shape[0]))
    if len(rows) != 1:
        raise RuntimeError("{MARKER} attention KV tensors differ in token rows %s; cannot place the decode pad"
                           % sorted(rows))
    rows = rows.pop()
    if rows % kb:
        raise RuntimeError("{MARKER} attention KV tensor rows %d are not a multiple of the kernel block %d"
                           % (rows, kb))
    block = max(split)
    spare = rows // kb - 1
    pool_end = cfg.num_blocks * block  # token rows the KV manager can hand out
    if spare * kb < pool_end:
        raise RuntimeError("{MARKER} no spare kernel block after the pool: %d rows, %d KV blocks x %d tokens"
                           % (rows, cfg.num_blocks, block))
    if runner._PAD_BLOCK_ID == spare and runner._PAD_SLOT_ID == spare * kb:
        return
    old = runner._PAD_BLOCK_ID
    runner._PAD_BLOCK_ID = spare
    runner._PAD_SLOT_ID = spare * kb
    logger.info("{MARKER} attention KV block %s split into %d-token kernel blocks; decode pad moved from "
                "kernel block %d to spare kernel block %d", split, kb, old, spare)


def _gaudi_fix_hybrid_pad_wrap(orig):
    def initialize_kv_cache(self, kv_cache_config):
        out = orig(self, kv_cache_config)
        _gaudi_fix_hybrid_pad(self)
        return out
    return initialize_kv_cache


HPUModelRunner.initialize_kv_cache = _gaudi_fix_hybrid_pad_wrap(HPUModelRunner.initialize_kv_cache)
'''


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
    for a in ANCHORS:
        require_once(src, a, "hpu_model_runner.py")
    require_once(src, "\nlogger = init_logger()\n", "hpu_model_runner.py")
    return src.rstrip("\n") + "\n" + HELPERS


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = (("runner", "self._PAD_BLOCK_ID = num_blocks * split"),)

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
