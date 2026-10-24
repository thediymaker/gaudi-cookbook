#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""mm-encoder-budget: bound the vision-encoder batch with VLLM_HPU_MM_ENCODER_MAX_PATCHES.

vLLM's GPU runner encodes only the scheduler's scheduled_encoder_inputs, which the scheduler keeps
inside the encoder compute budget. vllm-gaudi's HPUModelRunner._execute_mm_encoder instead
collects every image not yet in the encoder cache, for every scheduled request, and runs them
through embed_multimodal as one batch. The vision tower's transient memory grows with the number
of patches in that batch (about 98 KB per patch on Qwen3-VL-32B at TP 2), so one request with
many large images asks for tens of GiB in one call and fails with "PT_DEVMEM Allocation failed"
inside the encoder attention, which stops the engine.

The fix feeds embed_multimodal consecutive groups whose patch total stays within
VLLM_HPU_MM_ENCODER_MAX_PATCHES; an item larger than the budget runs alone. Item order is kept,
so the outputs still line up with their cache keys, and each image is still encoded whole, so the
results match the single batch up to bf16 rounding. Unset or 0 keeps the stock single batch.
Items without an image_grid_thw or video_grid_thw field each run alone when the budget is set.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "mm-encoder-budget"
MARKER = f"[gaudi-fix:{FIX_ID}]"

# The grouping helper was renamed between the releases: v0.26.0 group_mm_kwargs_by_modality,
# v0.30.0 group_and_batch_mm_kwargs. Both take (mm_kwargs, device=, pin_memory=) and yield
# (modality, num_items, kwargs).
GROUPERS = ("group_mm_kwargs_by_modality", "group_and_batch_mm_kwargs")
CALL = """        for _, num_items, mm_kwargs_group in {fn}(
                mm_kwargs,
                device=self.device,
                pin_memory=self.pin_memory,
        ):
            # Run the encoder.
"""
NEW_CALL = """        # {marker} groups bounded by VLLM_HPU_MM_ENCODER_MAX_PATCHES
        for _, num_items, mm_kwargs_group in _encoder_batches(
                {fn},
                mm_kwargs,
                device=self.device,
                pin_memory=self.pin_memory,
        ):
            # Run the encoder.
"""
HELPER = '''

# {marker} ---------------------------------------------------------------------------------
def _encoder_item_patches(item, budget):
    """Patch count of one multimodal item; an item without a grid runs alone."""
    for key in ("image_grid_thw", "video_grid_thw"):
        if key in item:
            return int(item[key].data.prod().item())
    return budget


def _encoder_batches(grouper, mm_kwargs, device, pin_memory):
    budget = int(os.environ.get("VLLM_HPU_MM_ENCODER_MAX_PATCHES", "0") or 0)
    if budget <= 0 or len(mm_kwargs) <= 1:
        yield from grouper(mm_kwargs, device=device, pin_memory=pin_memory)
        return
    chunk, used = [], 0
    for entry in mm_kwargs:
        n = _encoder_item_patches(entry[1], budget)
        if chunk and used + n > budget:
            yield from grouper(chunk, device=device, pin_memory=pin_memory)
            chunk, used = [], 0
        chunk.append(entry)
        used += n
    if chunk:
        yield from grouper(chunk, device=device, pin_memory=pin_memory)
'''


class Drift(Exception):
    """The installed source does not match what this fix expects."""


def patch_runner(src: str, ctx: dict[str, str]) -> str:
    if "\nimport os\n" not in src:
        raise Drift("module does not import os")
    start = src.find("    def _execute_mm_encoder(")
    end = src.find("\n    def ", start + 1)
    if start < 0 or end < 0 or src.count("    def _execute_mm_encoder(") != 1:
        raise Drift("_execute_mm_encoder not found once")
    body = src[start:end]
    hits = [fn for fn in GROUPERS if body.count(CALL.format(fn=fn)) == 1]
    if len(hits) != 1:
        raise Drift("encoder batching loop not found once in _execute_mm_encoder")
    fn = hits[0]
    body = body.replace(CALL.format(fn=fn), NEW_CALL.format(fn=fn, marker=MARKER), 1)
    return src[:start] + body + src[end:] + HELPER.format(marker=MARKER)


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
