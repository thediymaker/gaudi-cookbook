#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""so-row-owners: give grammar masks, sampling params and logprobs to the logits rows that exist.

When a step contains a request with a grammar, sample_tokens stacks the logits of the step
(logits_decode + logits_prompt) and samples them together. A prompt chunk that does not finish its
prompt produces no logits row, but three things still count it as one:

* apply_grammar_bitmask places each grammar mask at the request's persistent-batch index (the GPU
  rule, where every request has a row). With a partial chunk at a lower index the mask lands one
  row late: on another request's row, so a plain request is forced into someone else's grammar and
  the grammar request samples unconstrained (the grammar then rejects its token and the request
  fails), or past the last row, which raises IndexError on every rank and stops the engine.
* the sampling params are taken from decode_req_ids + prompt_req_ids, so rows after a partial
  chunk use their neighbour's params;
* the logprobs of those rows are filed under the wrong request.

Separately, _prepare_prefill_inputs gives a partial chunk one logits row when structured output
is on, but that flag still holds the previous step's value there, and without async scheduling
nothing discards the row: the token sampled at a mid-prompt position is returned as the request's
output.

The fix uses decode_sampled_requests + prefill_sampled_requests (built next to the logits, one
entry per row) as the row owners for the bitmask, the sampling params and the logprobs, whenever
that list matches the logits one to one; otherwise the step runs as before. A partial chunk gets a
logits row only when async scheduling discards it.

When both are used, apply structured-output-rows first (apply_all.py does). Applied alone, this fix
also puts the fallback order decodes first.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "so-row-owners"
MARKER = f"[gaudi-fix:{FIX_ID}]"

# Edit 1: the structured-output sampling call: stock, after structured-output-rows, or upstream #1916.
BITMASK_OLD = """            if grammar_output:
                self.apply_grammar_bitmask(scheduler_output, grammar_output, logits)
"""
CALL_VARIANTS = (
    (
        "structured-output-rows fix",
        "            # [gaudi-fix:structured-output-rows] rows are decodes first, so the per-request sampling"
        " params must be too.\n",
        """            sampler_output, _sampling_metadata = self._run_sampling(batch_changed, logits,
                                                                    list(pd_info.decode_req_ids) +
                                                                    list(pd_info.prompt_req_ids),
                                                                    logits.shape[0])
""",
    ),
    (
        "upstream decode-first order",
        "            # Rows are decodes first, so the per-request sampling params must be too.\n",
        """            sampler_output, _sampling_metadata = self._run_sampling(batch_changed, logits,
                                                                    pd_info.decode_req_ids + pd_info.prompt_req_ids,
                                                                    logits.shape[0])
""",
    ),
    (
        "stock prompts-first order (the fallback below also puts decodes first)",
        "",
        """            sampler_output, _sampling_metadata = self._run_sampling(batch_changed, logits,
                                                                    pd_info.prompt_req_ids + pd_info.decode_req_ids,
                                                                    logits.shape[0])
""",
    ),
)
OWNERS_NEW = f"""            # {MARKER} a prompt chunk that does not end its prompt has no logits row, so the
            # rows belong to the requests that produced logits, not to decode + prompt request ids.
            _so_rows = _gaudi_fix_so_row_owners(decode_sampled_requests, prefill_sampled_requests, logits.shape[0])
            if grammar_output:
                self._gaudi_fix_so_rows = _so_rows
                try:
                    self.apply_grammar_bitmask(scheduler_output, grammar_output, logits)
                finally:
                    self._gaudi_fix_so_rows = None
"""
CALL_NEW = """            sampler_output, _sampling_metadata = self._run_sampling(
                batch_changed, logits,
                _so_rows if _so_rows is not None else list(pd_info.decode_req_ids) + list(pd_info.prompt_req_ids),
                logits.shape[0])
"""

# Edit 2: logprobs of the structured-output step.
LOGPROBS_OLD = """            struct_logprobs_req_ids = (list(pd_info.decode_req_ids) + list(pd_info.prompt_req_ids))
"""
LOGPROBS_NEW = f"""            struct_logprobs_req_ids = (_so_rows if _so_rows is not None else  # {MARKER}
                                       list(pd_info.decode_req_ids) + list(pd_info.prompt_req_ids))
"""

# Edit 3: apply_grammar_bitmask places masks by row owner when they are known.
APPLY_OLD = """        grammar_bitmask = grammar_output.grammar_bitmask

        # We receive the structured output bitmask from the scheduler,
        # compacted to contain bitmasks only for structured output requests.
"""
APPLY_NEW = f"""        grammar_bitmask = grammar_output.grammar_bitmask
        _rows = getattr(self, "_gaudi_fix_so_rows", None)  # {MARKER}
        if _rows is not None:
            return _gaudi_fix_so_apply_bitmask_by_rows(scheduler_output, grammar_output, logits, _rows)

        # We receive the structured output bitmask from the scheduler,
        # compacted to contain bitmasks only for structured output requests.
"""

# Edit 4: a partial prompt chunk gets a logits row only when async scheduling discards it.
PARTIAL_OLD = """            if self.use_async_scheduling or self.use_structured_output:
                # NOTE(tianmu-li): align behavior of incomplete prompt with gpu_model_runner
"""
PARTIAL_NEW = f"""            # {MARKER} use_structured_output is still the previous step's value here, and only async
            # scheduling discards the token of a partial chunk's row; otherwise it gets no row.
            if self.use_async_scheduling:
                # NOTE(tianmu-li): align behavior of incomplete prompt with gpu_model_runner
"""

# The row lists are built beside the logits, one entry per row; the bitmask uses the CPU kernel.
GUARDS = (
    "logits_combined = logits_decode + logits_prompt",
    'xgr_cpu.apply_token_bitmask_inplace_cpu(logits_cpu, grammar_bitmask.to("cpu"), indices=index_tensor)',
    "logits_prompt.append(logits_device)\n                    prefill_sampled_requests.extend(logits_requests)",
    "logits_decode.append(logits_device[:num_decodes])\n"
    "                decode_sampled_requests.extend(self.input_batch.req_ids[:num_decodes])",
    "\nimport numpy as np\n",
    "\nlogger = init_logger()\n",
)

HELPERS = f'''

# ---- {MARKER} ----------------------------------------------------------------------------
def _gaudi_fix_so_row_owners(decode_reqs, prefill_reqs, n_rows):
    """The request behind each logits row of a structured-output step, or None to run as before."""
    rows = list(decode_reqs) + list(prefill_reqs)
    if len(rows) != int(n_rows) or len(set(rows)) != len(rows) or any(r is None for r in rows):
        logger.warning_once("{MARKER} %d logits rows for %d sampled requests (%d distinct); "
                            "structured-output step runs unchanged", int(n_rows), len(rows), len(set(rows)))
        return None
    return rows


def _gaudi_fix_so_apply_bitmask_by_rows(scheduler_output, grammar_output, logits, rows):
    """apply_grammar_bitmask with each request's mask on the row it owns (one row per request)."""
    grammar_bitmask = grammar_output.grammar_bitmask
    row_of = {{req_id: i for i, req_id in enumerate(rows)}}
    spec = scheduler_output.scheduled_spec_decode_tokens
    sorted_bitmask = np.full(shape=(logits.shape[0], grammar_bitmask.shape[1]),
                             fill_value=-1,
                             dtype=grammar_bitmask.dtype)
    out_indices = []
    cumulative_index = 0
    for req_id in grammar_output.structured_output_request_ids:
        row = row_of.get(req_id)
        if row is not None:
            sorted_bitmask[row] = grammar_bitmask[cumulative_index]
            out_indices.append(row)
        cumulative_index += 1 + len(spec.get(req_id, ()))
    if not out_indices:
        return
    index_tensor = None if len(out_indices) == logits.shape[0] else torch.tensor(out_indices, dtype=torch.int32)
    logits_cpu = logits.cpu().to(torch.float32)
    xgr_cpu.apply_token_bitmask_inplace_cpu(logits_cpu, torch.from_numpy(sorted_bitmask), indices=index_tensor)
    logits.copy_(logits_cpu.to(logits.device, non_blocking=True).to(logits.dtype))
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
    for g in GUARDS:
        require_once(src, g, "hpu_model_runner.py")
    matches = [v for v in CALL_VARIANTS if src.count(BITMASK_OLD + v[1] + v[2]) == 1]
    if len(matches) != 1:
        raise Drift("the structured-output sampling call has none of the known forms")
    _, comment, call = matches[0]
    src = replace_once(src, BITMASK_OLD + comment + call, OWNERS_NEW + comment + CALL_NEW, "structured-output sampling")
    src = replace_once(src, LOGPROBS_OLD, LOGPROBS_NEW, "structured-output logprobs request ids")
    src = replace_once(src, APPLY_OLD, APPLY_NEW, "apply_grammar_bitmask")
    src = replace_once(src, PARTIAL_OLD, PARTIAL_NEW, "partial prompt chunk logits row")
    g = src.index(GUARDS[0])
    if not (0 < src.index("_so_rows = _gaudi_fix_so_row_owners(") - g < 400 and 0 < src.index(LOGPROBS_NEW) - g < 2500):
        raise Drift("the edits are not in the structured-output branch of sample_tokens")
    if "xgr_cpu" not in src.split("\nclass ", 1)[0]:
        raise Drift("module-level xgr_cpu import not found")
    return src.rstrip("\n") + "\n" + HELPERS


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
