#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""spec-structured-output: structured outputs (JSON schema, required / named tool calls) under speculative decoding.

Generic (speculative decoding on vllm-gaudi; the same code is in vllm-gaudi main); a candidate for fixes/.

When a step carries a grammar bitmask the HPU runner takes a separate structured-output branch that keeps only the
first num_decodes decode logits (under speculation those are the padded verify rows, so request b > 0 is sampled
from request 0's rows), never runs the rejection sampler, applies the bitmask with the GPU's packed row arithmetic
and concatenates CPU and HPU tokens. response_format json_object stopped the engine (IndexError in
apply_grammar_bitmask), tool_choice "required" returned HTTP 500, and other requests in the step got tokens from the
wrong rows. The patch, only when a speculative config is set: masks prefill rows in place and samples them on the
normal path; masks every verify row of every structured decode (row b*M + i gets the grammar's row i, computed
through the drafts) and runs the normal speculative path, so drafts the grammar forbids are rejected and recovered
and bonus tokens come from the masked distribution, as on vLLM's GPU runner. Masking uses the runner's own xgrammar
CPU kernel and full-tensor round trip (fixed shapes, no new HPU graphs).

Usage: spec_structured_output.py [--check] [--root SITE_PACKAGES]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 source drifted (nothing written).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "glm-5-mxfp4:spec-structured-output"
MARKER = f"[{FIX_ID}]"

OLD_SETUP = """\
        if grammar_output is not None:
            logits_prompt = []
            logits_decode = []
            self.use_structured_output = True
"""

NEW_SETUP = """\
        if grammar_output is not None:
            logits_prompt = []
            logits_decode = []
            self.use_structured_output = True
        # [glm-5-mxfp4:spec-structured-output] under speculation, mask rows in place and keep the normal paths
        _hpu_so_spec = self.use_structured_output and bool(self.speculative_config)
        _hpu_so_rows = _hpu_so_bitmask_rows(scheduler_output, grammar_output) if _hpu_so_spec else None
"""

OLD_PREFILL = """\
                # Skip separate sampling for structured output
                if self.use_structured_output:
                    logits_prompt.append(logits_device)
                    prefill_sampled_requests.extend(logits_requests)
                else:
"""

NEW_PREFILL = """\
                # Skip separate sampling for structured output
                if self.use_structured_output and not _hpu_so_spec:  # [glm-5-mxfp4:spec-structured-output]
                    logits_prompt.append(logits_device)
                    prefill_sampled_requests.extend(logits_requests)
                else:
                    if _hpu_so_spec and logits_device.numel() > 0:  # [glm-5-mxfp4:spec-structured-output]
                        _hpu_so_apply(logits_device, [(i, _hpu_so_rows[r][0]) for i, r in enumerate(logits_requests)
                                                      if r in _hpu_so_rows])
"""

OLD_DECODE = """\
            if self.use_structured_output:
                logits_decode.append(logits_device[:num_decodes])
                decode_sampled_requests.extend(self.input_batch.req_ids[:num_decodes])
            else:
"""

NEW_DECODE = """\
            if self.use_structured_output and not _hpu_so_spec:  # [glm-5-mxfp4:spec-structured-output]
                logits_decode.append(logits_device[:num_decodes])
                decode_sampled_requests.extend(self.input_batch.req_ids[:num_decodes])
            else:
                if _hpu_so_spec:  # [glm-5-mxfp4:spec-structured-output] mask all verify rows, then the normal spec path
                    _hpu_so_apply(logits_device, _hpu_so_decode_pairs(pd_info.decode_req_ids, _hpu_so_rows,
                                                                      decode_data.spec_decode_metadata))
"""

OLD_COMBINED = """\
        if self.use_structured_output:
            # Scheduler places cached before prompt
"""

NEW_COMBINED = """\
        if self.use_structured_output and not _hpu_so_spec:  # [glm-5-mxfp4:spec-structured-output]
            # Scheduler places cached before prompt
"""

OLD_CONCAT = """\
        if self.use_async_scheduling or self.use_structured_output:
            # For async scheduling: keep tokens on HPU and avoid CPU sync
"""

NEW_CONCAT = """\
        if self.use_async_scheduling or (self.use_structured_output and not _hpu_so_spec):
            # [glm-5-mxfp4:spec-structured-output] For async scheduling: keep tokens on HPU and avoid CPU sync
"""

HELPER = '''\


# ---- [glm-5-mxfp4:spec-structured-output] ----
def _hpu_so_bitmask_rows(scheduler_output, grammar_output):
    """req_id -> its rows of the compacted grammar bitmask (1 + its scheduled drafts, in
    structured_output_request_ids order, as the StructuredOutputManager builds it)."""
    rows = {}
    bm = grammar_output.grammar_bitmask
    spec = scheduler_output.scheduled_spec_decode_tokens
    start = 0
    for req_id in grammar_output.structured_output_request_ids:
        n = 1 + len(spec.get(req_id, ()))
        rows[req_id] = bm[start:start + n]
        start += n
    return rows


def _hpu_so_decode_pairs(decode_req_ids, so_rows, spec_md):
    """(logits row, bitmask row) for every verify row of every structured decode."""
    pairs = []
    if spec_md is None:  # no drafts this step: one row per decode slot
        for b, req_id in enumerate(decode_req_ids):
            if req_id in so_rows:
                pairs.append((b, so_rows[req_id][0]))
        return pairs
    n_slots = len(spec_md.num_draft_tokens)
    m = int(spec_md.logits_indices.numel()) // max(1, n_slots)  # rows per slot (padded)
    for b, req_id in enumerate(decode_req_ids):
        if req_id not in so_rows or b >= n_slots:
            continue
        rows = so_rows[req_id]
        n = min(int(spec_md.num_draft_tokens[b]) + 1, len(rows), m)
        for i in range(n):
            pairs.append((b * m + i, rows[i]))
    return pairs


def _hpu_so_apply(logits, pairs):
    """Mask the given rows of logits in place with their grammar bitmask rows."""
    if not pairs:
        return
    width = pairs[0][1].shape[-1]
    bitmask = np.full((logits.shape[0], width), -1, dtype=pairs[0][1].dtype)
    idx = []
    for r, row in pairs:
        bitmask[r] = row
        idx.append(r)
    logits_cpu = logits.cpu().to(torch.float32)
    xgr_cpu.apply_token_bitmask_inplace_cpu(logits_cpu, torch.from_numpy(bitmask),
                                            indices=torch.tensor(idx, dtype=torch.int32))
    logits.copy_(logits_cpu.to(logits.device).to(logits.dtype))
'''


class Drift(Exception):
    """The installed source does not match what this patch expects."""


def replace_once(src: str, old: str, new: str, label: str) -> str:
    n = src.count(old)
    if n != 1:
        raise Drift(f"anchor {label!r} found {n} times, expected 1")
    return src.replace(old, new, 1)


def require_once(src: str, needle: str, label: str) -> None:
    n = src.count(needle)
    if n != 1:
        raise Drift(f"{label}: {needle.strip()!r} found {n} times, expected 1")


EDITS = (
    ("setup", OLD_SETUP, NEW_SETUP),
    ("prefill", OLD_PREFILL, NEW_PREFILL),
    ("decode", OLD_DECODE, NEW_DECODE),
    ("combined", OLD_COMBINED, NEW_COMBINED),
    ("concat", OLD_CONCAT, NEW_CONCAT),
)


def patch_runner(src: str, ctx: dict[str, str]) -> str:
    for need in ("\nimport numpy as np\n",):
        require_once(src, need, "vllm_gaudi/v1/worker/hpu_model_runner.py")
    if "xgr_cpu" not in src or "self.speculative_config" not in src:
        raise Drift("vllm_gaudi/v1/worker/hpu_model_runner.py: no xgr_cpu or speculative_config")
    for label, old, new in EDITS:
        src = replace_once(src, old, new, label)
    return src + HELPER


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
