#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""logitsprocs-rows: make min_p, logit_bias and min_tokens take effect on HPU.

vLLM implements these three request options as built-in logits processors whose per-request
state arrives through InputBatch.batch_update_builder and LogitsProcessor.update_state. The HPU
input batch creates the builder but never fills it (its source carries a TODO for this), so the
processors never receive any state and the three options are silently ignored on every model.

The fix does not use the builder protocol, because the HPU batch reorders its rows in ways that
protocol does not track. Instead:

* the input batch keeps each request's SamplingParams by request id (added in add_request,
  dropped in remove_request);
* make_selective_sampling_metadata replaces the three built-ins with small views built from
  those parameters and the live output-token lists for exactly the rows being sampled (row i is
  req_indices[i]). Custom logits processors pass through unchanged. When the logits hold k rows
  per request (speculative verify rows), each request's view covers its k rows, and the
  min_tokens view handles the padded verify layout.

Under speculative decoding vLLM still refuses min_p and logit_bias at request validation; that
is unchanged.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

FIX_ID = "logitsprocs-rows"
MARKER = f"[gaudi-fix:{FIX_ID}]"

EDITS = (
    (
        "init",
        "        self.batch_update_builder = BatchUpdateBuilder()\n",
        "        self.batch_update_builder = BatchUpdateBuilder()\n"
        f"        self._lp_rows_params: dict = {{}}  # {MARKER} req_id -> SamplingParams\n",
    ),
    (
        "add_request",
        "        if sampling_params := request.sampling_params:\n"
        "            if sampling_params.sampling_type == SamplingType.GREEDY:\n",
        "        if sampling_params := request.sampling_params:\n"
        f"            self._lp_rows_params[req_id] = sampling_params  # {MARKER}\n"
        "            if sampling_params.sampling_type == SamplingType.GREEDY:\n",
    ),
    (
        "remove_request",
        "        req_index = self.req_id_to_index.pop(req_id, None)\n"
        "        if req_index is None:\n"
        "            return None\n",
        f"        self._lp_rows_params.pop(req_id, None)  # {MARKER}\n"
        "        req_index = self.req_id_to_index.pop(req_id, None)\n"
        "        if req_index is None:\n"
        "            return None\n",
    ),
)
SELECT_OLD = "            logitsprocs=self.logitsprocs,\n"
SELECT_NEW = f"            logitsprocs=_lp_rows(self, req_indices),  # {MARKER}\n"

SIG = re.compile(r"^    def make_selective_sampling_metadata\(\n", re.M)
NEXT_MEMBER = re.compile(r"^    (?:def |@)", re.M)

HELPER = f'''

# --- {MARKER} ---------------------------------------------------------------
# min_p, logit_bias and min_tokens as views over the rows being sampled, built from each
# request's SamplingParams (the batch_update_builder protocol is not fed on HPU).
import math as _lp_rows_math

from vllm.v1.sample.logits_processor import LogitsProcessors as _LpRowsProcessors
from vllm.v1.sample.logits_processor.builtin import LogitBiasLogitsProcessor as _LpRowsLogitBias
from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor as _LpRowsMinP
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor as _LpRowsMinTokens

# The views build small index tensors from Python lists every step; keep them out of
# torch.compile graphs (a no-op in lazy mode).
_lp_rows_eager = getattr(getattr(torch, "compiler", None), "disable", None) or (lambda f: f)


def _lp_rows_expand(m, n_rows):
    """Row r of the selection -> the rows it owns in a tensor of n_rows rows."""
    if m == 0:
        return []
    if n_rows == m:
        return [[r] for r in range(m)]
    if n_rows % m == 0:  # speculative verify layout: k rows per request
        k = n_rows // m
        return [list(range(r * k, r * k + k)) for r in range(m)]
    return [[r] for r in range(min(m, n_rows))]


class _LpRowsMinPView(_LpRowsMinP):
    """min_p for the selected rows (argmax invariant, applied after temperature)."""

    def __init__(self, vals):  # a view: no super().__init__
        self.vals = vals
        self.min_p_count = sum(1 for v in vals if v > 0)

    def update_state(self, batch_update):
        pass

    @_lp_rows_eager
    def apply(self, logits):
        if not self.min_p_count:
            return logits
        col = [0.0] * logits.shape[0]
        for r, rows in enumerate(_lp_rows_expand(len(self.vals), logits.shape[0])):
            for j in rows:
                col[j] = self.vals[r]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        max_probs = torch.amax(probs, dim=-1, keepdim=True)
        min_p = torch.tensor(col, dtype=max_probs.dtype).to(logits.device).unsqueeze(1)
        logits.masked_fill_(probs < max_probs.mul_(min_p), -float("inf"))
        return logits


class _LpRowsLogitBiasView(_LpRowsLogitBias):
    """logit_bias for the selected rows."""

    def __init__(self, sel):
        self.sel = sel
        self.biases = {{r: b for r, b in enumerate(sel) if b}}

    def update_state(self, batch_update):
        pass

    @_lp_rows_eager
    def apply(self, logits):
        if not self.biases:
            return logits
        own = _lp_rows_expand(len(self.sel), logits.shape[0])
        rows, toks, vals = [], [], []
        for r, b in self.biases.items():
            for j in own[r] if r < len(own) else ():
                for tok, v in b.items():
                    rows.append(j)
                    toks.append(int(tok))
                    vals.append(float(v))
        if rows:
            idx = (
                torch.tensor(rows, dtype=torch.int64).to(logits.device),
                torch.tensor(toks, dtype=torch.int64).to(logits.device),
            )
            logits[idx] += torch.tensor(vals, dtype=logits.dtype).to(logits.device)
        return logits


class _LpRowsMinTokensView(_LpRowsMinTokens):
    """min_tokens for the selected rows; the speculative path knows the padded verify layout."""

    def __init__(self, sel):
        self.sel = sel
        self.min_toks = {{r: e for r, e in enumerate(sel) if e is not None}}

    def update_state(self, batch_update):
        pass

    def _put(self, logits, rows, toks):
        if rows:
            idx = (
                torch.tensor(rows, dtype=torch.int64).to(logits.device),
                torch.tensor(toks, dtype=torch.int64).to(logits.device),
            )
            fill = torch.full((len(rows),), -float("inf"), dtype=logits.dtype).to(logits.device)
            logits.index_put_(idx, fill)
        return logits

    @_lp_rows_eager
    def apply(self, logits):
        if not self.min_toks:
            return logits
        own = _lp_rows_expand(len(self.sel), logits.shape[0])
        rows, toks = [], []
        for r, (min_tok, out_ids, stop_ids) in self.min_toks.items():
            if len(out_ids) >= min_tok or r >= len(own):
                continue
            for j in own[r]:
                for t in stop_ids:
                    rows.append(j)
                    toks.append(int(t))
        return self._put(logits, rows, toks)

    @_lp_rows_eager
    def apply_with_spec_decode(self, logits, num_draft_tokens):
        if not self.min_toks:
            return logits
        nd = [int(x) for x in num_draft_tokens]
        n = logits.shape[0]
        if n == sum(nd):  # packed layout
            starts, acc = [], 0
            for d in nd:
                starts.append(acc)
                acc += d
        elif nd and n % len(nd) == 0:  # padded layout: k rows per request
            k = n // len(nd)
            starts = [r * k for r in range(len(nd))]
        else:
            return logits
        rows, toks = [], []
        for r, (min_tok, out_ids, stop_ids) in self.min_toks.items():
            if r >= len(nd):
                continue
            for i in range(min(max(min_tok - len(out_ids), 0), nd[r])):
                for t in stop_ids:
                    rows.append(starts[r] + i)
                    toks.append(int(t))
        return self._put(logits, rows, toks)


def _lp_rows(batch, req_indices):
    """A LogitsProcessors whose built-ins see exactly the selected rows."""
    params = []
    for i in req_indices:
        rid = batch._req_ids[i] if i < len(batch._req_ids) else None
        params.append(batch._lp_rows_params.get(rid) if rid is not None else None)
    outs = [batch.req_output_token_ids[i] if i < len(batch.req_output_token_ids) else None for i in req_indices]
    min_p = [float(p.min_p) if p is not None else 0.0 for p in params]
    bias = [(p.logit_bias or None) if p is not None else None for p in params]
    min_tok = []
    for i, p, out in zip(req_indices, params, outs):
        stop = p.all_stop_token_ids if p is not None else None
        if p is None or not p.min_tokens or not stop:
            min_tok.append(None)
            continue
        # Tokens generated before this sample. With async scheduling the output list lags, so
        # also derive the count from the computed-token bookkeeping.
        n_done = len(out) if out is not None else 0
        try:
            n_done = max(n_done, int(batch.num_computed_tokens_cpu[i]) + 1 - int(batch.num_prompt_tokens[i]))
        except (AttributeError, IndexError):
            pass
        min_tok.append((p.min_tokens, range(n_done), stop) if n_done < p.min_tokens else None)
    builtins = (_LpRowsMinP, _LpRowsLogitBias, _LpRowsMinTokens)
    others = [p for p in (batch.logitsprocs.all if batch.logitsprocs is not None else ()) if type(p) not in builtins]
    views = [_LpRowsMinPView(min_p), _LpRowsLogitBiasView(bias), _LpRowsMinTokensView(min_tok)]
    return _LpRowsProcessors(views + others)
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


def method_span(src: str) -> tuple[int, int]:
    """Start and end offsets of make_selective_sampling_metadata's body."""
    sigs = list(SIG.finditer(src))
    if len(sigs) != 1:
        raise Drift(f"make_selective_sampling_metadata found {len(sigs)} times, expected 1")
    start = sigs[0].end()
    nxt = NEXT_MEMBER.search(src, start)
    return start, nxt.start() if nxt else len(src)


def patch_input_batch(src: str, ctx: dict[str, str]) -> str:
    for needle in ("self._req_ids", "self.req_output_token_ids", "self.num_prompt_tokens = np.", "import torch\n"):
        if needle not in src:
            raise Drift(f"hpu_input_batch.py lacks {needle.strip()!r}")
    for label, old, new in EDITS:
        src = replace_once(src, old, new, label)
    start, end = method_span(src)
    body = src[start:end]
    if "for i, req_idx in enumerate(req_indices)" not in body:
        raise Drift("make_selective_sampling_metadata does not re-key rows by req_indices")
    body = replace_once(body, SELECT_OLD, SELECT_NEW, "logitsprocs in make_selective_sampling_metadata")
    return src[:start] + body + src[end:] + HELPER


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (("vllm_gaudi", "v1/worker/hpu_input_batch.py", patch_input_batch, "input_batch"),)


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
