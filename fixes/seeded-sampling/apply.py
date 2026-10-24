#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""seeded-sampling: seeded random sampling on HPU without torch.Generator ops on the device.

A request with `seed` and temperature > 0 gets its own torch.Generator on the HPU, and vLLM's
random_sample draws its noise one row at a time with q[i].exponential_(generator=g). In lazy mode
with lazy collectives (the vllm-gaudi default) this races in the Habana bridge under load and a
tensor-parallel worker segfaults in habana::exponential_, which stops the engine for every request.
Requests without a seed use q.exponential_() and are not affected. Separately, the runner pads a
batch by repeating its first request, so a seeded first request's generator advanced once per
padded row and its tokens depended on the batch bucket.

The fix:
1. topk_topp_sampler.py, random_sample: on an HPU tensor, rows whose generator is an
   HpuSeededGenerator get counter-based Exp(1) noise, -log(u) with u from a 32-bit integer hash of
   (seed, step, token id), computed on the device in one batched op and written with one
   index_copy_. step is the request's own count of sampling calls, so the same seed and prompt give
   the same tokens whatever else is in the batch. Rows without a seed keep q.exponential_(); other
   generator types keep the stock per-row loop. The seeded subset is padded to a power of two to
   bound the number of graph shapes.
2. hpu_model_runner.py: seeded requests get HpuSeededGenerator(seed), a CPU torch.Generator
   subclass that also carries the seed and the step counter. No generator is created on the HPU.
3. hpu_model_runner.py: with async scheduling a partial-prefill sample is discarded; that call
   gives its step back (as the GPU runner rewinds its generator), so tokens do not depend on how a
   prompt was chunked.

Seeded outputs stay deterministic but differ from the stock noise stream. vllm-gaudi replaces the
stock rejection sampler (whose generator loops would meet a CPU generator); the fix refuses to
apply if that replacement is gone.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "seeded-sampling"
MARKER = f"[gaudi-fix:{FIX_ID}]"

SAMPLER_OLD = """    if generators:
        # TODO(woosuk): This can be slow because we handle each request
        # one by one. Optimize this.
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return sample_with_exponential_noise(probs, q)
"""
SAMPLER_NEW = f"""    if generators:
        # {MARKER} exponential_(generator=...) on a lazy HPU tensor can crash the
        # worker. Seeded rows get counter-based noise on the device; generators that
        # are returned (other kinds) keep the stock per-row loop.
        if q.device.type == "hpu":
            generators = _hpu_seeded_noise_(q, generators)
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return sample_with_exponential_noise(probs, q)
"""

SAMPLER_BLOCK = (
    f"""

# ---- {MARKER} seeded random sampling on HPU without torch.Generator ops ----
"""
    + r'''_HPU_M64 = (1 << 64) - 1
_HPU_MIX1 = 0x7FEB352D
_HPU_MIX2 = 0x846CA68B - (1 << 32)  # as a signed int32


class HpuSeededGenerator(torch.Generator):
    """Per-request generator for HPU: a CPU torch.Generator seeded with the request seed (so a
    stock CPU consumer still works), carrying the seed and the count of sampling calls."""

    @classmethod
    def create(cls, seed):
        g = cls(device="cpu")
        g.manual_seed(seed)
        g.hpu_seed = int(seed)
        g.hpu_step = 0
        return g


def _hpu_splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & _HPU_M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _HPU_M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _HPU_M64
    return x ^ (x >> 31)


def _hpu_int32(v):
    return v - (1 << 32) if v >= (1 << 31) else v


def hpu_seeded_key(seed, step):
    """Two signed int32 words for (seed, step)."""
    k = _hpu_splitmix64(_hpu_splitmix64(int(seed) & _HPU_M64) ^ (int(step) & _HPU_M64))
    return _hpu_int32(k & 0xFFFFFFFF), _hpu_int32(k >> 32)


def _hpu_lsr32(x, s):
    # logical right shift of int32 (torch's >> is arithmetic on signed types)
    return torch.bitwise_and(torch.bitwise_right_shift(x, s), (1 << (32 - s)) - 1)


def _hpu_mix32(x):
    # lowbias32: a bijection on 32-bit words with full avalanche
    x = torch.bitwise_xor(x, _hpu_lsr32(x, 16))
    x = x * _HPU_MIX1
    x = torch.bitwise_xor(x, _hpu_lsr32(x, 15))
    x = x * _HPU_MIX2
    return torch.bitwise_xor(x, _hpu_lsr32(x, 16))


def hpu_counter_exponential(keys, vocab_size):
    """Exp(1) noise [n, vocab_size], float32, a pure function of (int32 keys [n, 2], column)."""
    cols = torch.arange(vocab_size, dtype=torch.int32, device=keys.device)
    x = _hpu_mix32(cols.unsqueeze(0) + keys[:, 0:1])
    x = _hpu_mix32(torch.bitwise_xor(x, keys[:, 1:2]))
    # 23 random bits -> u in [2^-24, 1 - 2^-24], exact in float32: -log(u) is finite and > 0.
    u = (_hpu_lsr32(x, 9).to(torch.float32) + 0.5) * (2.0 ** -23)
    return u.log().neg_().clamp_min_(torch.finfo(torch.float32).tiny)


def _hpu_seeded_noise_(q, generators):
    """Fill q's rows owned by an HpuSeededGenerator; return the generators left to the stock loop."""
    rest = {}
    rows, keys, seen = [], [], {}
    n_rows = q.shape[0]
    for i, g in generators.items():
        if i >= n_rows:
            continue
        if not isinstance(g, HpuSeededGenerator):
            rest[i] = g
            continue
        k = seen.get(id(g))
        if k is None:  # one step per request per call: padded copies of a row reuse its key
            k = seen[id(g)] = hpu_seeded_key(g.hpu_seed, g.hpu_step)
            g.hpu_step += 1
        rows.append(i)
        keys.append(k)
    if rows:
        n = len(rows)
        nb = 1 << (n - 1).bit_length()  # bucket the seeded subset: few graph shapes
        rows += [rows[0]] * (nb - n)
        keys += [keys[0]] * (nb - n)
        meta = torch.tensor([[a, b, r] for (a, b), r in zip(keys, rows)], dtype=torch.int32).to(q.device)
        noise = hpu_counter_exponential(meta[:, 0:2], q.shape[1])
        q.index_copy_(0, meta[:, 2].to(torch.long), noise.to(q.dtype))
    return rest
'''
)

RUNNER_GEN_OLD = """                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
"""
RUNNER_GEN_NEW = f"""                # {MARKER} no torch.Generator on the HPU: a CPU generator carrying
                # (seed, step); random_sample derives this request's noise from it on the device.
                from vllm.v1.sample.ops.topk_topp_sampler import HpuSeededGenerator
                generator = HpuSeededGenerator.create(sampling_params.seed)
"""

RUNNER_DISCARD_OLD = """        if self.use_async_scheduling:
            self.input_batch.prev_sampled_token_ids = sampled_token_ids.flatten()
"""
RUNNER_DISCARD_NEW = f"""        if self.use_async_scheduling:
            # {MARKER} a discarded partial-prefill sample gives its noise step back
            # (the GPU runner rewinds its generator), so tokens do not depend on chunking.
            for _gaudi_fix_i in self.invalid_req_indices:
                _gaudi_fix_g = self.input_batch.generators.get(int(_gaudi_fix_i))
                if _gaudi_fix_g is not None and getattr(_gaudi_fix_g, "hpu_step", 0) > 0:
                    _gaudi_fix_g.hpu_step -= 1
            self.input_batch.prev_sampled_token_ids = sampled_token_ids.flatten()
"""

# Context that must be present (not edited here).
SAMPLER_CONTEXT = ("\nimport torch\n", "\ndef random_sample(", "            self.forward = self.forward_native\n")
RUNNER_CONTEXT = (
    "sampling_params.sampling_type == SamplingType.RANDOM_SEED:",
    "self.invalid_req_indices",
    "self.sampler = Sampler()",
)
# vllm-gaudi must keep replacing the stock rejection sampler, whose generator loops would meet a
# CPU generator with HPU tensors.
REJECTION_REBIND = "rejection_sampler.rejection_sample = rejection_sample"


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
    require_present(ctx["rejection"], REJECTION_REBIND, "hpu_rejection_sampler.py")
    return replace_once(src, SAMPLER_OLD, SAMPLER_NEW, "random_sample generator loop") + SAMPLER_BLOCK


def patch_runner(src: str, ctx: dict[str, str]) -> str:
    for needle in RUNNER_CONTEXT:
        require_present(src, needle, "hpu_model_runner.py")
    src = replace_once(src, RUNNER_GEN_OLD, RUNNER_GEN_NEW, "per-request generator")
    return replace_once(src, RUNNER_DISCARD_OLD, RUNNER_DISCARD_NEW, "discarded partial-prefill step")


# Not fixed upstream.
UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm_gaudi", "v1/sample/hpu_rejection_sampler.py", None, "rejection"),
    ("vllm", "v1/sample/ops/topk_topp_sampler.py", patch_sampler, "sampler"),
    ("vllm_gaudi", "v1/worker/hpu_model_runner.py", patch_runner, "runner"),
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
