#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""topk-subset-sampler: top-k / top-p sampling on HPU without sorting the whole vocabulary.

On HPU, TopKTopPSampler takes forward_native and, without Triton, apply_top_k_top_p_pytorch, which
sorts the full [batch, vocab] logits on every decode step whenever any row has top_k or top_p (the
top-k-only shortcut is enabled only on CPU). Models whose generation config sets top_k / top_p put
every request on that path. With a large vocabulary the sort can take a large share of each decode
step at high batch sizes, and the cost grows with the batch.

When every row has a top_k <= 1024, the fix takes torch.topk(K) once (K the smallest of 64, 128,
256, 512, 1024 covering the batch's largest top_k, so few graph shapes), applies each row's top_k
and top_p inside those K values and draws the token with the stock exponential race. Tokens
outside a row's top-k have probability 0 in the stock code, so the distribution is the same (ties
at the k-th value are kept by the stock mask and cut by topk, which is measure zero on real
logits). The batch's largest top_k comes from the input batch's CPU copy, so there is no device
sync. A batch with a row without top_k, per-request generators (seed), fp64 Gumbel noise, or a
processed-logits logprobs mode keeps the stock path unchanged.

This changes which random numbers are drawn, not the sampling distribution. It is a performance
fix and is opt-in.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "topk-subset-sampler"
MARKER = f"[gaudi-fix:{FIX_ID}]"

SAMPLER_OLD = '''        PyTorch-native implementation of top-k and top-p sampling.

        The logits tensor may be updated in-place.
        """
        logits = apply_top_k_top_p(logits, k, p)
'''
SAMPLER_NEW = f'''        PyTorch-native implementation of top-k and top-p sampling.

        The logits tensor may be updated in-place.
        """
        # {MARKER} on HPU, sample inside torch.topk(K) instead of sorting the
        # whole vocabulary; None means this batch keeps the stock path below.
        if logits.device.type == "hpu":
            _hpu_out = _hpu_topk_subset_sample(self, logits, generators, k, p)
            if _hpu_out is not None:
                return _hpu_out
        logits = apply_top_k_top_p(logits, k, p)
'''

SAMPLER_BLOCK = (
    f"""

# ---- {MARKER} top-k/top-p sampling on HPU without a full-vocabulary sort ----
"""
    + r'''_HPU_TOPK_BUCKETS = (64, 128, 256, 512, 1024)


def hpu_topk_subset_probs(logits, k, p, K):
    """(probs float32 [B, K], ids [B, K]): each row's top-k/top-p distribution over its K largest
    logits. Equals the stock sort-based masks followed by softmax, restricted to those K ids."""
    vals, ids = logits.topk(K, dim=-1)  # descending
    pos = torch.arange(K, device=logits.device, dtype=k.dtype).unsqueeze(0)
    vals = vals.masked_fill(pos >= k.unsqueeze(1), float("-inf"))
    if p is not None:
        probs = vals.softmax(dim=-1, dtype=torch.float32)
        higher = probs.cumsum(dim=-1) - probs  # mass of the strictly larger tokens
        # stock: drop the smallest tokens whose cumulative mass is <= 1 - p, keep at least one;
        # i.e. keep a token while the mass above it is < p (position 0 always has 0 above it).
        vals = vals.masked_fill(higher >= p.to(torch.float32).unsqueeze(1), float("-inf"))
    return vals.softmax(dim=-1, dtype=torch.float32), ids


def _hpu_topk_subset_sample(sampler, logits, generators, k, p):
    if (k is None or generators or getattr(sampler, "use_fp64_gumbel", False)
            or sampler.logprobs_mode in ("processed_logits", "processed_logprobs")):
        return None
    max_k = getattr(k, "hpu_max_top_k", None)
    if max_k is None:
        max_k = int(k.max().item())
    K = next((b for b in _HPU_TOPK_BUCKETS if b >= max_k), 0)
    if not K or K >= logits.shape[-1]:
        return None
    probs, ids = hpu_topk_subset_probs(logits, k, p, K)
    q = torch.empty_like(probs).exponential_()
    choice = probs.div_(q).argmax(dim=-1, keepdim=True)
    return ids.gather(1, choice).view(-1), None
'''
)

BATCH_OLD = "            top_k=None if self.no_top_k else self.top_k[req_indices],\n"
BATCH_NEW = f"""            # {MARKER} carry the batch's max top_k from the CPU copy, so the
            # sampler can pick its topk size without a device sync.
            top_k=None if self.no_top_k else _hpu_tag_max_top_k(self.top_k[req_indices], self.top_k_cpu,
                                                                req_indices),
"""
BATCH_BLOCK = f"""

# ---- {MARKER} ----
def _hpu_tag_max_top_k(top_k, top_k_cpu, req_indices):
    try:
        top_k.hpu_max_top_k = int(top_k_cpu[req_indices].max()) if len(req_indices) else 0
    except Exception:  # the sampler falls back to a device max
        pass
    return top_k
"""

SAMPLER_CONTEXT = (
    "\nimport torch\n",
    "\n    def forward_native(\n",
    "            self.forward = self.forward_native\n",
)
BATCH_CONTEXT = ("    def make_selective_sampling_metadata(", "self.top_k_cpu = self.top_k_cpu_tensor.numpy()")


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


def require_present(src: str, needle: str, label: str) -> None:
    if needle not in src:
        raise Drift(f"{label}: {needle.strip()!r} not found")


def patch_sampler(src: str, ctx: dict[str, str]) -> str:
    for needle in SAMPLER_CONTEXT:
        require_present(src, needle, "topk_topp_sampler.py")
    return replace_once(src, SAMPLER_OLD, SAMPLER_NEW, "forward_native entry") + SAMPLER_BLOCK


def patch_batch(src: str, ctx: dict[str, str]) -> str:
    for needle in BATCH_CONTEXT:
        require_present(src, needle, "hpu_input_batch.py")
    return replace_once(src, BATCH_OLD, BATCH_NEW, "selective top_k") + BATCH_BLOCK


# Not fixed upstream.
UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm", "v1/sample/ops/topk_topp_sampler.py", patch_sampler, "sampler"),
    ("vllm_gaudi", "v1/worker/hpu_input_batch.py", patch_batch, "batch"),
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
