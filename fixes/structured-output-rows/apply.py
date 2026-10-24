#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""structured-output-rows: sample structured-output steps with each row's own request params.

When a step contains a request with a grammar (json_schema, json_object, guided choice or regex,
tool_choice "required" or a named tool), sample_tokens defers sampling and stacks the step's logits
decodes first (logits_decode + logits_prompt). The sampling metadata for those rows is built from
prompt_req_ids + decode_req_ids, so in any step with both a finishing prefill and decodes every row
is sampled with another request's temperature, top-p/top-k, seed generator and penalties (the
penalties also read the other request's output tokens). The logprobs ids and the token scatter
that follow already use decodes first.

The fix builds the metadata in decode-first order, matching the logits.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "structured-output-rows"
MARKER = f"[gaudi-fix:{FIX_ID}]"

OLD = """            logits_combined = logits_decode + logits_prompt
            logits = torch.cat(logits_combined, dim=0)
            # Apply structured output bitmasks if present
            if grammar_output:
                self.apply_grammar_bitmask(scheduler_output, grammar_output, logits)
            sampler_output, _sampling_metadata = self._run_sampling(batch_changed, logits,
                                                                    pd_info.prompt_req_ids + pd_info.decode_req_ids,
                                                                    logits.shape[0])
"""
NEW = f"""            logits_combined = logits_decode + logits_prompt
            logits = torch.cat(logits_combined, dim=0)
            # Apply structured output bitmasks if present
            if grammar_output:
                self.apply_grammar_bitmask(scheduler_output, grammar_output, logits)
            # {MARKER} rows are decodes first, so the per-request sampling params must be too.
            sampler_output, _sampling_metadata = self._run_sampling(batch_changed, logits,
                                                                    list(pd_info.decode_req_ids) +
                                                                    list(pd_info.prompt_req_ids),
                                                                    logits.shape[0])
"""

# The logprobs bookkeeping right after the call must already be decodes first.
LOGPROBS_ORDER = "struct_logprobs_req_ids = (list(pd_info.decode_req_ids) + list(pd_info.prompt_req_ids))"


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
    require_once(src, LOGPROBS_ORDER, "hpu_model_runner.py")
    out = replace_once(src, OLD, NEW, "structured-output sampling call")
    if not 0 < out.index(LOGPROBS_ORDER) - out.index(NEW) < 1500:
        raise Drift("the logprobs ordering is not next to the structured-output sampling call")
    return out


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = (("runner", "# Rows are decodes first, so the per-request sampling params must be too."),)

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
