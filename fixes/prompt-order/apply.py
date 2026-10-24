#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""prompt-order: run a step's prompts in order of context length under prefix caching.

The scheduler publishes a request's KV blocks to the prefix cache when it schedules them, so a
request scheduled later in the same step can get cache hits on blocks that another request only
computes in that step. The GPU runner runs the step as one forward and every layer writes its KV
before attention reads it. The HPU runner runs a step's prompts as separate forwards in
input-batch row order, and a new request takes the lowest free row, so a reader can sit above its
writer. Its forward then attends to whatever the reused block held before, which can be KV from
an unrelated earlier request.

A reader's hits always reach past what its writer has computed, so sorting the prompt rows by
num_computed_tokens (ascending, stable) puts every writer first. The sort uses swap_states, so all
row-indexed state follows the request. It runs after the decodes-first ordering and only when
prefix caching is enabled.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "prompt-order"
MARKER = f"[gaudi-fix:{FIX_ID}]"

FUNC_ANCHOR = "\n\ndef get_target_layer_suffix_list(model_type) -> list[str]:\n"
FUNC_NEW = (
    f'''

def sort_prompts_by_context(b, first_prompt: int) -> None:
    """{MARKER} Sort prompt rows [first_prompt, num_reqs) by num_computed_tokens (stable).

    A request scheduled later in a step can get prefix-cache hits on blocks another request
    computes in that step. Prompts run as separate forwards in row order, so writers must run
    before their readers; a reader's num_computed_tokens is always larger than its writer's.
    """
    num_reqs = b.num_reqs
    if num_reqs - first_prompt < 2:
        return
    order = sorted(range(first_prompt, num_reqs), key=lambda i: int(b.num_computed_tokens_cpu[i]))
    target_req_ids = [b.req_ids[i] for i in order]
    for i, req_id in enumerate(target_req_ids, start=first_prompt):
        j = b.req_id_to_index[req_id]
        if j != i:
            b.swap_states(i, j)
'''
    + FUNC_ANCHOR
)

CALL_ANCHOR = """        if not self.vllm_config.speculative_config:
            ensure_multi_token_decodes_last(self.input_batch, scheduler_output.num_scheduled_tokens)
        # Prepare prompts/decodes info
"""
CALL_NEW = f"""        if not self.vllm_config.speculative_config:
            ensure_multi_token_decodes_last(self.input_batch, scheduler_output.num_scheduled_tokens)
        # {MARKER} run prompts in order of context length, so no prompt reads a
        # prefix-cache hit before the request computing it has written it.
        if self.use_prefix_caching:
            sort_prompts_by_context(self.input_batch, self._get_num_decodes(scheduler_output))
        # Prepare prompts/decodes info
"""

# Code the inserted lines rely on; each must be present exactly once.
REQUIRED = (
    "self._ensure_decodes_first(scheduler_output)\n",
    "    def _get_num_decodes(self, scheduler_output",
    "        self.use_prefix_caching = (",
    "    def swap_states(self, i1: int, i2: int)",
)


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
    for needle in REQUIRED[:3]:
        require_once(src, needle, "hpu_model_runner.py")
    require_once(ctx["input_batch"], REQUIRED[3], "hpu_input_batch.py")
    src = replace_once(src, FUNC_ANCHOR, FUNC_NEW, "module function insertion point")
    return replace_once(src, CALL_ANCHOR, CALL_NEW, "decodes-first ordering in sample_tokens")


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = (("runner", "def sort_prompts_by_context("),)

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm_gaudi", "v1/worker/hpu_input_batch.py", None, "input_batch"),
    ("vllm_gaudi", "v1/worker/hpu_model_runner.py", patch_runner, "runner"),
)


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
