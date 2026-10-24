#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""penalty-shapes: bucket the penalty token tensors so HPU graphs repeat across decode steps.

Repetition, presence and frequency penalties build two token tensors on every sampling step,
each padded to exactly the longest list in the batch: vLLM's _convert_to_tensors pads the output
ids to the longest output so far (one longer every decode step), and the HPU input batch pads the
prompt ids to the longest running prompt. On HPU every new shape is a new graph, so a batch that
uses penalties recompiles on nearly every step and throughput collapses as streams are added.

The fix pads both lengths up to a multiple of PAD_MULTIPLE with the vocab_size pad id the code
already uses and drops, so shapes repeat; results are unchanged. On HPU tensors it also counts
token bins in float32 instead of int64 (exact far beyond any context length, and int64 is
emulated on Gaudi) and broadcasts the repetition penalty as a (rows, 1) column instead of
materializing it at (rows, vocab). Other devices keep the stock arithmetic.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "penalty-shapes"
MARKER = f"[gaudi-fix:{FIX_ID}]"
PAD_MULTIPLE = 256

BINS_OLD = """    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros(
"""
BINS_NEW = f"""    if tokens.device.type == "hpu":  # {MARKER} float32 bins, not emulated int64
        bins = torch.zeros((num_seqs, vocab_size + 1), dtype=torch.float32, device=tokens.device)
        bins.scatter_add_(1, tokens, torch.ones(tokens.shape, dtype=torch.float32, device=tokens.device))
        bins = bins[:, :vocab_size]
        return bins, bins > 0
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros(
"""

REP_OLD = """    repetition_penalties = repetition_penalties.unsqueeze(dim=1).repeat(
        1, logits.size(1)
    )
"""
REP_NEW = f"""    if logits.device.type == "hpu":  # {MARKER} broadcast the (rows, 1) column
        repetition_penalties = repetition_penalties.unsqueeze(dim=1).to(logits.dtype)
    else:
        repetition_penalties = repetition_penalties.unsqueeze(dim=1).repeat(
            1, logits.size(1)
        )
"""

OUT_OLD = """        pad=vocab_size,
        device="cpu",
        dtype=torch.int64,
        pin_memory=PIN_MEMORY,
    )
"""
OUT_NEW = f"""        pad=vocab_size,
        device="cpu",
        dtype=torch.int64,
        pin_memory=PIN_MEMORY,
        # {MARKER} a bucketed length, so the HPU graph repeats
        max_len=_penalty_pad(max((len(t) for t in output_token_ids), default=0)),
    )
"""
FUNC_OLD = "\n\ndef _convert_to_tensors(\n"
FUNC_NEW = f"""

def _penalty_pad(n: int) -> int:  # {MARKER}
    return max({PAD_MULTIPLE}, -(-n // {PAD_MULTIPLE}) * {PAD_MULTIPLE})


def _convert_to_tensors(
"""

PROMPT_OLD = """        max_prompt_len = self.num_prompt_tokens[:self.num_reqs].max()
        prompt_token_ids_cpu_tensor = torch.empty(
            (self.num_reqs, max_prompt_len),
            device="cpu",
            dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        prompt_token_ids = prompt_token_ids_cpu_tensor.numpy()
        prompt_token_ids[:] = self.token_ids_cpu[:self.num_reqs, :max_prompt_len]
"""
PROMPT_NEW = f"""        max_prompt_len = self.num_prompt_tokens[:self.num_reqs].max()
        # {MARKER} a bucketed length, so the HPU graph repeats
        padded_len = max({PAD_MULTIPLE}, -(-int(max_prompt_len) // {PAD_MULTIPLE}) * {PAD_MULTIPLE})
        padded_len = min(padded_len, self.token_ids_cpu.shape[1])
        prompt_token_ids_cpu_tensor = torch.empty(
            (self.num_reqs, padded_len),
            device="cpu",
            dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        prompt_token_ids = prompt_token_ids_cpu_tensor.numpy()
        prompt_token_ids[:] = self.vocab_size
        prompt_token_ids[:, :max_prompt_len] = self.token_ids_cpu[:self.num_reqs, :max_prompt_len]
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


def patch_bins(src: str, ctx: dict[str, str]) -> str:
    require_once(src, "def get_token_bin_counts_and_mask(", "model_executor/layers/utils.py")
    return replace_once(src, BINS_OLD, BINS_NEW, "bin counts in get_token_bin_counts_and_mask")


def patch_rep(src: str, ctx: dict[str, str]) -> str:
    require_once(src, "def apply_repetition_penalties_torch(", "_custom_ops.py")
    return replace_once(src, REP_OLD, REP_NEW, "repetition penalty column in apply_repetition_penalties_torch")


def patch_penalties(src: str, ctx: dict[str, str]) -> str:
    require_once(src, "    output_tokens_tensor = make_tensor_with_pad(\n", "v1/sample/ops/penalties.py")
    src = replace_once(src, OUT_OLD, OUT_NEW, "output token padding in _convert_to_tensors")
    return replace_once(src, FUNC_OLD, FUNC_NEW, "_convert_to_tensors definition")


def patch_input_batch(src: str, ctx: dict[str, str]) -> str:
    require_once(src, "    def _make_prompt_token_ids_cpu_tensor(self)", "hpu_input_batch.py")
    return replace_once(src, PROMPT_OLD, PROMPT_NEW, "prompt token tensor in _make_prompt_token_ids_cpu_tensor")


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm", "model_executor/layers/utils.py", patch_bins, "layer_utils"),
    ("vllm", "_custom_ops.py", patch_rep, "custom_ops"),
    ("vllm", "v1/sample/ops/penalties.py", patch_penalties, "penalties"),
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
