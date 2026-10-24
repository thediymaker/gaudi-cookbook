#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""spec-penalties: penalties, bad_words and allowed_token_ids under speculative decoding on HPU.

Generic (speculative decoding on vllm-gaudi); a candidate for fixes/. Needs spec-sampling first.

vLLM's RejectionSampler.apply_logits_processors assumes the GPU's packed verify layout (request i owns
num_draft_tokens[i] consecutive rows; histories built from sampling_metadata.spec_token_ids). The HPU runner hands it
the padded layout (every request owns k rows) and never sets spec_token_ids. With any request carrying presence,
frequency or repetition penalties, bad_words or allowed_token_ids in a speculative step, the penalty tensors are
expanded to sum(num_draft_tokens) rows against B*k logits rows: shape error, the engine dies. The patch, only when
such a request is in the batch: sets spec_token_ids to each request's real drafts (so the bonus sampler counts them),
and calls the upstream code with num_draft_tokens = [k] * B and drafts padded to k, which makes the packed
arithmetic land on the padded rows. Other batches take the upstream path unchanged.

Usage: spec_penalties.py [--check] [--root SITE_PACKAGES]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 source drifted (nothing written).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "glm-5-mxfp4:spec-penalties"
MARKER = f"[{FIX_ID}]"

BLOCK = """\


# ---- [glm-5-mxfp4:spec-penalties] logits processors on the HPU padded verify layout ----
import dataclasses as _hpu_dc

_hpu_orig_rs_forward = rejection_sampler.RejectionSampler.forward
_hpu_orig_rs_apply_lp = rejection_sampler.RejectionSampler.apply_logits_processors


def _hpu_spec_needs_lp(sm):
    return (not sm.no_penalties) or bool(sm.bad_words_token_ids) or sm.allowed_token_ids_mask is not None


def _hpu_spec_draft_lists(metadata, pad_to_k):
    n_req = len(metadata.num_draft_tokens)
    flat = metadata.draft_token_ids.reshape(-1).tolist()
    k = max(1, len(flat) // max(1, n_req))
    out = []
    for i, n in enumerate(metadata.num_draft_tokens):
        row = [int(t) for t in flat[i * k:(i + 1) * k]]
        n = int(n)
        if pad_to_k:
            out.append(row[:n] + [row[n - 1] if n > 0 else 0] * (k - n))
        else:
            out.append(row[:n])
    return out, k


def _hpu_spec_rs_forward(self, metadata, draft_probs, logits, sampling_metadata):
    if _hpu_spec_needs_lp(sampling_metadata) and sampling_metadata.spec_token_ids is None:
        real, _ = _hpu_spec_draft_lists(metadata, pad_to_k=False)
        sampling_metadata = _hpu_dc.replace(sampling_metadata, spec_token_ids=real)
    return _hpu_orig_rs_forward(self, metadata, draft_probs, logits, sampling_metadata)


def _hpu_spec_rs_apply_lp(self, logits, sampling_metadata, metadata):
    n_req = len(metadata.num_draft_tokens)
    if n_req == 0 or not _hpu_spec_needs_lp(sampling_metadata):
        return _hpu_orig_rs_apply_lp(self, logits, sampling_metadata, metadata)
    k = logits.shape[0] // n_req
    if k * n_req != logits.shape[0] or k < max(metadata.num_draft_tokens):
        return _hpu_orig_rs_apply_lp(self, logits, sampling_metadata, metadata)  # not the HPU layout
    padded, _ = _hpu_spec_draft_lists(metadata, pad_to_k=True)
    md = _hpu_dc.replace(metadata, num_draft_tokens=[k] * n_req)
    sm = _hpu_dc.replace(sampling_metadata, spec_token_ids=padded)
    return _hpu_orig_rs_apply_lp(self, logits, sm, md)


rejection_sampler.RejectionSampler.forward = _hpu_spec_rs_forward
rejection_sampler.RejectionSampler.apply_logits_processors = _hpu_spec_rs_apply_lp
"""


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


def patch_sampler(src: str, ctx: dict[str, str]) -> str:
    need = "\nfrom vllm.v1.sample import rejection_sampler\n"
    require_once(src, need, "vllm_gaudi/v1/sample/hpu_rejection_sampler.py")
    require_once(src, "def _hpu_spec_rejection_sample(", "spec-sampling must be applied first")
    return src + BLOCK


UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (("vllm_gaudi", "v1/sample/hpu_rejection_sampler.py", patch_sampler, "sampler"),)


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
