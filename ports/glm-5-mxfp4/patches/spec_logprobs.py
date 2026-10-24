#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""spec-logprobs: return sampled-token logprobs under speculative decoding on HPU.

Generic (speculative decoding on vllm-gaudi); a candidate for fixes/.

With MTP or EAGLE a request can emit several tokens per step. The HPU runner's speculative branch never hands the
rejection sampler's logprobs to the output, and the logprobs builder makes exactly one row per request: a request
with logprobs=true failed with HTTP 500 ("list index out of range") as soon as it got two tokens in one step. The
patch, mirroring vLLM's GPU runner:
  1. in the speculative branch, filters the rejection sampler's logprobs to the emitted tokens and adds them as a
     segment;
  2. lays such segments out per request and sets cu_num_generated_tokens, so the scheduler finds each request's rows;
  3. sets SpecDecodeMetadata.cu_num_sampled_tokens to the padded offsets (i+1)*(1+k): the HPU verify rows are padded
     (request i owns rows i*(1+k) ..), and the rejection sampler read every request after one with fewer drafts from
     the wrong rows (the previous token's distribution).
Without speculation the original code path runs.

Usage: spec_logprobs.py [--check] [--root SITE_PACKAGES]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 source drifted (nothing written).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "glm-5-mxfp4:spec-logprobs"
MARKER = f"[{FIX_ID}]"

OLD_BRANCH = """\
                        sampled_token_ids = sampler_output.sampled_token_ids
                        decode_sampled_token_ids = \\
                            self.rejection_sampler.parse_output(
                                sampled_token_ids,
                                self.input_batch.vocab_size,
                        )
"""

NEW_BRANCH = """\
                        sampled_token_ids = sampler_output.sampled_token_ids
                        decode_sampled_token_ids = \\
                            self.rejection_sampler.parse_output(
                                sampled_token_ids,
                                self.input_batch.vocab_size,
                        )
                        if sampler_output.logprobs_tensors is not None:  # [glm-5-mxfp4:spec-logprobs]
                            _hpu_lt = sampler_output.logprobs_tensors
                            # land on the CPU first (sync; ids converted there), then filter there
                            _hpu_lt = LogprobsTensors(_hpu_lt.logprob_token_ids.cpu().to(torch.int64),
                                                      _hpu_lt.logprobs.cpu(),
                                                      _hpu_lt.selected_token_ranks.cpu().to(torch.int64))
                            _hpu_sl = self.rejection_sampler.parse_output(
                                sampled_token_ids,
                                self.input_batch.vocab_size,
                                logprobs_tensors=_hpu_lt,
                            )
                            if isinstance(_hpu_sl, tuple) and _hpu_sl[1] is not None:
                                logprobs_segments.append((list(pd_info.decode_req_ids), _hpu_sl[1]))
"""

OLD_BUILD = """\
        if not active_segments:
            return None

        # Determine the number of logprob columns from the first
        # active segment
"""

NEW_BUILD = """\
        if not active_segments:
            return None
        if any(isinstance(lp, LogprobsLists) for _, lp in active_segments):  # [glm-5-mxfp4:spec-logprobs]
            return _hpu_spec_build_logprobs(self, active_segments, num_output_rows)

        # Determine the number of logprob columns from the first
        # active segment
"""

OLD_CU = """\
            cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(num_sampled_tokens, cumsum_dtype=np.int32)
"""

NEW_CU = """\
            cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(num_sampled_tokens, cumsum_dtype=np.int32)
            # [glm-5-mxfp4:spec-logprobs] HPU verify rows are padded (request i owns rows i*max ..);
            # RejectionSampler._get_logprobs_tensors reads each request's rows from these offsets.
            cu_num_sampled_tokens = (np.arange(1, len(num_sampled_tokens) + 1, dtype=np.int32) *
                                     np.int32(max_num_sampled_tokens))
"""

HELPER = '''\


# ---- [glm-5-mxfp4:spec-logprobs] ----
def _hpu_spec_build_logprobs(runner, segments, num_output_rows):
    """Per-request rows + cu_num_generated_tokens when a step mixes speculative segments
    (LogprobsLists filtered by RejectionSampler.parse_output, already carrying cu offsets
    over the decode rows) with plain one-row-per-request LogprobsTensors."""
    rows = {}
    ncols = 0
    for req_ids, lp in segments:
        if isinstance(lp, LogprobsLists):
            ids, lps, rks = lp.logprob_token_ids, lp.logprobs, lp.sampled_token_ranks
            cu = lp.cu_num_generated_tokens
            for i, rid in enumerate(req_ids):
                a, b = int(cu[i]), int(cu[i + 1])
                rows[runner.input_batch.req_id_to_index[rid]] = (ids[a:b], lps[a:b], rks[a:b])
            ncols = max(ncols, ids.shape[1])
        else:
            ll = lp.tolists()
            for i, rid in enumerate(req_ids):
                rows[runner.input_batch.req_id_to_index[rid]] = (ll.logprob_token_ids[i:i + 1],
                                                                 ll.logprobs[i:i + 1],
                                                                 ll.sampled_token_ranks[i:i + 1])
            ncols = max(ncols, ll.logprob_token_ids.shape[1])
    ids_out, lps_out, rks_out, cu = [], [], [], [0]
    for idx in range(num_output_rows):
        r = rows.get(idx)
        n = 0 if r is None else int(r[0].shape[0])
        if n:
            ids, lps, rks = r
            if ids.shape[1] < ncols:
                ids = np.pad(ids, ((0, 0), (0, ncols - ids.shape[1])))
                lps = np.pad(lps, ((0, 0), (0, ncols - lps.shape[1])), constant_values=-np.inf)
            ids_out.append(ids.astype(np.int64))
            lps_out.append(lps.astype(np.float32))
            rks_out.append(rks.astype(np.int64))
        cu.append(cu[-1] + n)
    if not ids_out:
        return None
    return LogprobsLists(np.concatenate(ids_out), np.concatenate(lps_out), np.concatenate(rks_out), cu)
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


REQUIRED = (
    "\nimport numpy as np\n",
    "\nimport torch\n",
    "LogprobsLists, LogprobsTensors",
    "    def _prepare_spec_decode_inputs(self, scheduler_output, logits_indices, token_ids_device, "
    "max_num_sampled_tokens):",
)


def patch_runner(src: str, ctx: dict[str, str]) -> str:
    for need in REQUIRED:
        require_once(src, need, "vllm_gaudi/v1/worker/hpu_model_runner.py")
    src = replace_once(src, OLD_BRANCH, NEW_BRANCH, "speculative branch")
    src = replace_once(src, OLD_BUILD, NEW_BUILD, "logprobs builder")
    src = replace_once(src, OLD_CU, NEW_CU, "padded cu_num_sampled_tokens")
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
