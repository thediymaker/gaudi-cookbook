#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""gather-logprobs-int32: cast logprob token ids to int32 before concatenating them.

vllm-gaudi replaces vLLM's Sampler.gather_logprobs with _hpu_gather_logprobs (vllm_gaudi/patches.py),
which builds the (tokens, k + 1) id tensor as an int64 torch.cat of the sampled ids and the top-k
indices and then casts the result to int32. On HPU in lazy mode that int64 cat followed by the cast
is read back with wrong contents under load: ids outside the vocabulary, and column 0 not equal to
the sampled token. The float logprobs built next to it are correct. The bad ids either crash the
detokenizer in the API server or come back as truncated or wrong top_logprobs.

Casting each operand to int32 before the cat avoids the int64 path. It is numerically identical
(vocabulary ids fit in int32) and adds no synchronization. The runner's prompt-logprobs gather
(_gather_prompt_logprobs_hpu) builds its ids the same way and gets the same change.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "gather-logprobs-int32"
MARKER = f"[gaudi-fix:{FIX_ID}]"

SAMPLER_OLD = """    token_ranks = _sampler_mod.batched_count_greater_than(logprobs, token_logprobs)
    indices = torch.cat((token_ids, topk_indices), dim=1)
    logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)
    indices = indices.to(torch.int32)
    return LogprobsTensors(indices, logprobs, token_ranks)
"""
SAMPLER_NEW = f"""    token_ranks = _sampler_mod.batched_count_greater_than(logprobs, token_logprobs)
    # {MARKER} cast to int32 before the cat: on HPU the int64 cat followed by an
    # int32 cast can read back wrong ids. Vocabulary ids fit in int32.
    indices = torch.cat((token_ids.to(torch.int32), topk_indices.to(torch.int32)), dim=1)
    logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)
    return LogprobsTensors(indices, logprobs, token_ranks)
"""

PROMPT_OLD = """        token_ranks = (logprobs >= token_logprobs).sum(-1)

        indices = torch.cat((token_ids, topk_indices), dim=1).to(torch.int32)
        combined_logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)
"""
PROMPT_NEW = f"""        token_ranks = (logprobs >= token_logprobs).sum(-1)

        # {MARKER} cast to int32 before the cat (see vllm_gaudi/patches.py).
        indices = torch.cat((token_ids.to(torch.int32), topk_indices.to(torch.int32)), dim=1)
        combined_logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)
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


def patch_patches(src: str, ctx: dict[str, str]) -> str:
    require_once(src, "def _hpu_gather_logprobs(", "patches.py")
    return replace_once(src, SAMPLER_OLD, SAMPLER_NEW, "_hpu_gather_logprobs id concat")


def patch_runner(src: str, ctx: dict[str, str]) -> str:
    require_once(src, "    def _gather_prompt_logprobs_hpu(", "hpu_model_runner.py")
    return replace_once(src, PROMPT_OLD, PROMPT_NEW, "_gather_prompt_logprobs_hpu id concat")


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = (
    ("patches", "indices = torch.cat((token_ids.to(torch.int32), topk_indices.to(torch.int32)), dim=1)"),
    ("runner", "indices = torch.cat((token_ids.to(torch.int32), topk_indices.to(torch.int32)), dim=1)"),
)

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm_gaudi", "patches.py", patch_patches, "patches"),
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
