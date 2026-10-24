#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""prompt-logprobs: make prompt_logprobs safe and correct in the HPU model runner.

1. Crash. sample_tokens keeps each prefill request's hidden states as
   non_flattened_hidden_states[i], assuming a [batch, seq, hidden] output. Models that return a
   flat [batch * seq, hidden] tensor (flatten_input MoE adapters, and multimodal models with one
   prompt in the batch) hand compute_logits a single token's vector, and the shape error is raised
   in the worker, which stops the engine. Each request's rows are now located in the flat view by
   the batch layout the runner built (row-padded or merged prefill). A layout that does not match
   is logged once and that batch's prompt logprobs are skipped.
2. Wrong values. The gathered ids, logprobs and ranks were copied to the CPU with non_blocking
   copy_ into buffers of another dtype, and the ids were built with an int64 cat followed by an
   int32 cast, which can read back wrong on HPU. Now: synchronous .cpu() in the device dtype, the
   casts on the CPU, and the prompt-token column written from the CPU ids.
3. Late reads. Hidden states of every prefill batch were kept until the end of the step, after
   later forwards and graph replays. Each request's rows are cloned right after its forward.
4. Memory. compute_logits and log_softmax ran over every prompt token of a chunk at once (several
   GB of fp32 at a large vocabulary). They now run in row chunks of VLLM_HPU_PROMPT_LOGPROBS_ROWS
   (default 512, a multiple of 128), with the tail padded to a multiple of 128.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "prompt-logprobs"
MARKER = f"[gaudi-fix:{FIX_ID}]"

COLLECT_OLD = """                for i, rid in enumerate(req_id):
                    if rid in self.input_batch.num_prompt_logprobs:
                        prefill_hidden_states_for_logprobs[rid] = \\
                            non_flattened_hidden_states[i]
"""
COLLECT_NEW = f"""                # {MARKER} each prompt-logprobs request's own rows of the (possibly
                # flat) output, cloned right after the forward.
                if self.input_batch.num_prompt_logprobs:
                    _gaudi_plp_collect(self, prefill_hidden_states_for_logprobs, non_flattened_hidden_states,
                                       token_ids, req_id, scheduler_output)
"""

H2D_OLD = """            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(self.device, non_blocking=True)
"""
H2D_NEW = f"""            num_prompt_tokens = len(request.prompt_token_ids)
            # {MARKER} target ids are sliced on the CPU per chunk in _gaudi_plp_score
"""

SCORE_OLD = """            hs = prefill_hidden_states[req_id]
            if hs.dim() == 3:
                hs = hs.squeeze(0)  # [seq_len, hidden_dim]
            prompt_hidden_states = hs[:num_logits]
            logits = self.model.compute_logits(prompt_hidden_states)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok:start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            gathered = self._gather_prompt_logprobs_hpu(logprobs, num_prompt_logprobs, tgt_token_ids)

            # Transfer HPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(gathered.logprob_token_ids, non_blocking=True)
            logprobs_tensors.logprobs[chunk_slice].copy_(gathered.logprobs, non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(gathered.selected_token_ranks, non_blocking=True)
"""
SCORE_NEW = f"""            # {MARKER} row-chunked logits, synchronous device-to-host copies, casts on the CPU
            _gaudi_plp_score(self, prefill_hidden_states[req_id], request.prompt_token_ids, start_idx, start_tok,
                             num_logits, num_prompt_logprobs, logprobs_tensors)
"""

HELPERS = f'''

# ---- {MARKER} prompt-logprobs helpers ----
_GAUDI_PLP_WARNED = set()
_GAUDI_PLP_ROWS = max(128, int(os.environ.get("VLLM_HPU_PROMPT_LOGPROBS_ROWS", "512")) // 128 * 128)


def _gaudi_plp_warn(key, msg):
    if key not in _GAUDI_PLP_WARNED:
        _GAUDI_PLP_WARNED.add(key)
        logger.warning("prompt_logprobs: %s", msg)


def _gaudi_plp_rows_of(hidden, token_ids, req_ids, num_scheduled, merged):
    """(start, count) of each request's tokens in the flat [B*T, H] view of a prefill batch's
    output, or None when the output does not match the batch layout."""
    if token_ids.dim() != 2:
        return None
    b, t = int(token_ids.shape[0]), int(token_ids.shape[1])
    total = 1
    for d in hidden.shape[:-1]:
        total *= int(d)
    if total != b * t:
        return None
    out, off = [], 0
    for i, rid in enumerate(req_ids):
        n = int(num_scheduled[rid])
        start = off if merged else i * t
        off += n
        if n <= 0 or start + n > total or (not merged and n > t):
            return None
        out.append((start, n))
    return out


def _gaudi_plp_collect(runner, store, hidden, token_ids, req_ids, scheduler_output):
    want = runner.input_batch.num_prompt_logprobs
    if not any(rid in want for rid in req_ids):
        return
    spans = _gaudi_plp_rows_of(hidden, token_ids, req_ids, scheduler_output.num_scheduled_tokens,
                               bool(getattr(runner, "use_merged_prefill", False)))
    if spans is None:
        _gaudi_plp_warn("layout", "prefill output %s does not match the batch layout %s for %d request(s); "
                        "prompt logprobs skipped for this batch" % (tuple(hidden.shape), tuple(token_ids.shape),
                                                                     len(req_ids)))
        return
    flat = hidden.reshape(-1, hidden.shape[-1])
    for rid, (start, n) in zip(req_ids, spans):
        if rid in want:
            store[rid] = flat[start:start + n].clone()


def _gaudi_plp_score(runner, hs, prompt_token_ids, start_idx, start_tok, num_logits, k, dst):
    """Fill dst rows [start_idx, start_idx + num_logits) with the prompt logprobs of prompt
    tokens start_tok.. given this chunk's hidden states hs [n_scheduled, H]."""
    hs = hs.reshape(-1, hs.shape[-1])
    tgt_all = torch.tensor(prompt_token_ids[start_tok:start_tok + num_logits], dtype=torch.int64)
    if hs.shape[0] < num_logits or tgt_all.shape[0] != num_logits:
        _gaudi_plp_warn("short", "chunk has %d hidden rows / %d targets for %d prompt logprobs; filled with NaN"
                        % (hs.shape[0], tgt_all.shape[0], num_logits))
        sl = slice(start_idx, start_idx + num_logits)
        dst.logprob_token_ids[sl] = 0
        dst.logprobs[sl] = float("nan")
        dst.selected_token_ranks[sl] = 0
        return
    rows = _GAUDI_PLP_ROWS
    for a in range(0, num_logits, rows):
        n = min(rows, num_logits - a)
        pad = min(rows, -(-n // 128) * 128)
        h = hs[a:a + n]
        if pad != n:
            h = torch.nn.functional.pad(h, (0, 0, 0, pad - n))
        tgt = torch.zeros(pad, dtype=torch.int64)
        tgt[:n] = tgt_all[a:a + n]
        logits = runner.model.compute_logits(h)
        if logits is None:  # non-driver rank of a gather-to-rank-0 logits processor
            continue
        lp = runner.sampler.compute_logprobs(logits)  # [pad, V] float32
        tok = lp.gather(-1, tgt.to(lp.device).unsqueeze(-1))  # [pad, 1]
        rank = (lp >= tok).sum(-1)  # [pad] int64
        if k > 0:
            top_lp, top_ix = torch.topk(lp, k, dim=-1)
            top_lp, top_ix = top_lp.cpu(), top_ix.cpu()
        tok, rank = tok.cpu(), rank.cpu()
        sl = slice(start_idx + a, start_idx + a + n)
        dst.logprob_token_ids[sl, 0] = tgt[:n].to(dst.logprob_token_ids.dtype)
        dst.logprobs[sl, 0] = tok[:n, 0].to(dst.logprobs.dtype)
        if k > 0:
            dst.logprob_token_ids[sl, 1:] = top_ix[:n].to(dst.logprob_token_ids.dtype)
            dst.logprobs[sl, 1:] = top_lp[:n].to(dst.logprobs.dtype)
        dst.selected_token_ranks[sl] = rank[:n].to(dst.selected_token_ranks.dtype)
'''

# Module-level names the helpers use; each must be present exactly once.
REQUIRED = ("\nimport os\n", "\nimport torch\n", "\nlogger = ", "    def _get_prompt_logprobs_dict(")


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
    for needle in REQUIRED:
        require_once(src, needle, "hpu_model_runner.py")
    src = replace_once(src, COLLECT_OLD, COLLECT_NEW, "collect prefill hidden states")
    src = replace_once(src, H2D_OLD, H2D_NEW, "full-prompt host-to-device copy")
    src = replace_once(src, SCORE_OLD, SCORE_NEW, "score prompt logprobs")
    return src + HELPERS


# No single upstream change covers this fix (the crash part alone is fixed on vllm-gaudi main).
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
