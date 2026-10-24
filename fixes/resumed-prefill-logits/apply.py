#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""resumed-prefill-logits: one sampled token for a request resumed after preemption.

When a preempted request resumes, the scheduler recomputes its prompt plus the tokens it had
already generated. InputBatch.add_request folds those earlier outputs into num_prompt_tokens, and
the runner sizes a prefill's logits as num_computed + num_scheduled - num_prompt_tokens + 1, which
is then 1. The fold only happens when num_computed_tokens == 0, but with prefix caching a resumed
request usually gets cache hits on its own freed blocks and resumes with num_computed_tokens > 0.
The runner then computes one logit, and samples one token, at every recomputed output position
and credits all of them to the request: the generated tail is appended again (duplicated text),
an extra [tokens x vocab] logits tensor is allocated, and with speculative decoding the scheduler's
accounting assertion stops the engine.

The fix treats a request as recomputing whenever num_computed_tokens < known tokens - 1, as vLLM
main does. Running decodes and speculative decode requests always have num_computed_tokens >=
known tokens - 1 and are unchanged.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "resumed-prefill-logits"
MARKER = f"[gaudi-fix:{FIX_ID}]"

OLD = """        if request.num_computed_tokens == 0:
            self.num_prompt_tokens[req_index] = num_prompt_tokens + len(request.output_token_ids)
"""
NEW = f"""        # {MARKER} with prefix caching a resumed request can have num_computed_tokens > 0
        # (cache hits on its own blocks), so detect recompute by any gap before the last known token.
        num_known_tokens = num_prompt_tokens + len(request.output_token_ids)
        if request.num_computed_tokens < num_known_tokens - 1:
            self.num_prompt_tokens[req_index] = num_known_tokens
"""

# The runner must size prefill logits from num_prompt_tokens, as the fix assumes.
RUNNER_FORMULA = "seq_num_computed_tokens + seq_num_scheduled_tokens - num_prompt_tokens + 1"


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


def patch_input_batch(src: str, ctx: dict[str, str]) -> str:
    require_once(src, "        num_prompt_tokens = len(request.prompt_token_ids)\n", "hpu_input_batch.py")
    if RUNNER_FORMULA not in ctx["runner"]:
        raise Drift("hpu_model_runner.py no longer sizes prefill logits from num_prompt_tokens")
    out = replace_once(src, OLD, NEW, "preemption recompute in add_request")
    if not 0 < out.index(NEW) - out.index("        num_prompt_tokens = len(request.prompt_token_ids)\n") < 1500:
        raise Drift("the edited lines are not in add_request")
    return out


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = (("input_batch", "if request.num_computed_tokens < num_known_tokens - 1:"),)

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm_gaudi", "v1/worker/hpu_model_runner.py", None, "runner"),
    ("vllm_gaudi", "v1/worker/hpu_input_batch.py", patch_input_batch, "input_batch"),
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
