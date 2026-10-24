#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""sink-once: count attention sinks once per sequence (paged decode, and prefill on v0.26.0).

An attention sink is one extra logit per head in the softmax denominator. Paged decode (flat_pa)
runs the softmax per KV block and merges the blocks of a sequence afterwards; pipelined_pa adds
exp(sink - block_max) to every block's partial sum, so a sequence spread over n blocks ends up
with n copies of the sink instead of one, and heads with a sink attend more weakly as the context
grows. Each block now gets sink - log(n), n = blocks of its sequence, so the n copies add up to
exactly one after the merge. Single-block sequences are unchanged; padding blocks map to no
sequence and use n = 1.

Prefill: FusedSDPA on Gaudi software 1.24.1 accepts a sink argument and does not apply it. On the
v0.26.0 layout the fix gives the kernel one extra key whose logit is the sink and whose value is
zero (the head dimension grows by one, padded to a multiple of 64), which is exact on the same
kernel. Layers that pass window_size keep the stock call. On the v0.30.0 layout (query tiling
around the FusedSDPA call) only the decode part is applied.

Models without sinks never reach the edited branches.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "sink-once"
MARKER = f"[gaudi-fix:{FIX_ID}]"

DECODE_OLD = """        sink = sink.expand(query.shape[0], -1, query.shape[-2], -1)
        if kv_heads != q_heads:
            sink = sink.unflatten(1, (kv_heads, -1))
"""
DECODE_NEW = f"""        sink = sink.expand(query.shape[0], -1, query.shape[-2], -1)
        # {MARKER} pipelined_pa adds exp(sink) to the partial sum of every block. Give each
        # block sink - log(n), with n the number of blocks of its sequence, so that the merged
        # sum holds exactly one exp(sink). Padding blocks map to no sequence and get n = 1.
        mapping = block_mapping.float()
        blocks_per_seq = torch.matmul(mapping, mapping.sum(0, keepdim=True).t()).clamp(min=1.0)
        sink = (sink.float() - torch.log(blocks_per_seq).view(-1, 1, 1, 1)).to(sinks.dtype)
        if kv_heads != q_heads:
            sink = sink.unflatten(1, (kv_heads, -1))
"""
# flat_pa's own context: the decode edit is only valid inside it.
DECODE_GUARD = "def flat_pa(query, key_cache, value_cache, block_list, block_mapping, block_bias"

# v0.26.0 prefill: one untiled FusedSDPA call.
PREFILL_OLD = """    args = [
        query, key, value, attn_bias, 0.0, is_causal, scale, softmax_mode, recompute_mode, valid_seq_lengths,
        padding_side
    ]
    if sinks is not None:
        args += [window_size] if window_size else [None]
    else:
        args += [window_size] if window_size else []
    # use sinks in fsdpa
    if sinks is not None:
        args += [sinks]
    attn_weights = fsdpa_op(*args)
"""
PREFILL_NEW = f"""    if sinks is not None and not window_size:  # {MARKER} FusedSDPA ignores `sink`
        attn_weights = _fsdpa_sink_extra_key(query, key, value, attn_bias, fsdpa_op, scale, softmax_mode,
                                             recompute_mode, padding_side, sinks)
    else:
        args = [
            query, key, value, attn_bias, 0.0, is_causal, scale, softmax_mode, recompute_mode, valid_seq_lengths,
            padding_side
        ]
        if sinks is not None:
            args += [window_size] if window_size else [None]
        else:
            args += [window_size] if window_size else []
        # use sinks in fsdpa
        if sinks is not None:
            args += [sinks]
        attn_weights = fsdpa_op(*args)
"""
HELPER_OLD = "def _fsdpa_prompt_attention(query: torch.Tensor,\n"
HELPER_NEW = (
    f'''def _fsdpa_sink_extra_key(query, key, value, attn_bias, fsdpa_op, scale, softmax_mode, recompute_mode,
                          padding_side, sinks):
    """{MARKER} Apply the sink in FusedSDPA through one extra key.

    An extra head dimension holds sink / scale in the query and 1 in the extra key, so that key's
    logit is exactly the sink for every query; its value is zero. query [B, H, T, D],
    key/value [B, Hkv, L, D] -> [B, H, T, D].
    """
    B, H, T, D = query.shape
    Hkv, L = key.shape[1], key.shape[2]
    Dp = (D + 1 + 63) // 64 * 64
    dt = query.dtype
    qx = (sinks.float() / scale).to(dt).view(1, H, 1, 1).expand(B, H, T, 1)
    query = torch.cat([query, qx, query.new_zeros(B, H, T, Dp - D - 1)], dim=-1)
    ek = key.new_zeros(B, Hkv, 1, Dp)
    ek[..., D] = 1.0
    key = torch.cat([torch.cat([key, key.new_zeros(B, Hkv, L, Dp - D)], dim=-1), ek], dim=2)
    value = torch.cat([torch.cat([value, value.new_zeros(B, Hkv, L, Dp - D)], dim=-1),
                       value.new_zeros(B, Hkv, 1, Dp)], dim=2)
    if attn_bias is None:
        # a fresh prompt runs causal; with the sink key appended that needs an explicit mask
        keep = torch.ones(T, L, dtype=torch.bool, device=query.device).tril(L - T)
        attn_bias = torch.zeros(T, L, dtype=dt, device=query.device).masked_fill_(~keep, -math.inf).view(1, 1, T, L)
    attn_bias = torch.cat([attn_bias, attn_bias.new_zeros(*attn_bias.shape[:-1], 1)], dim=-1)
    out = fsdpa_op(query, key, value, attn_bias, 0.0, False, scale, softmax_mode, recompute_mode, None,
                   padding_side)
    return out[..., :D]


'''
    + HELPER_OLD
)
# v0.30.0 prefill: FusedSDPA is called through a local helper with query tiling (decode part only).
TILED_PREFILL = "    def call_fsdpa(q, bias):\n"


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


def patch_ops(src: str, ctx: dict[str, str]) -> str:
    require_once(src, DECODE_GUARD, "extension/ops.py")
    require_once(src, "\nimport math\n", "extension/ops.py")
    start = src.index(DECODE_GUARD)
    if not 0 < src.find(DECODE_OLD, start) - start < 2000:
        raise Drift("decode sink block is not inside flat_pa")
    src = replace_once(src, DECODE_OLD, DECODE_NEW, "flat_pa sink expansion")
    n_untiled, n_tiled = src.count(PREFILL_OLD), src.count(TILED_PREFILL)
    if (n_untiled, n_tiled) == (1, 0):
        src = replace_once(src, HELPER_OLD, HELPER_NEW, "_fsdpa_prompt_attention definition")
        src = replace_once(src, PREFILL_OLD, PREFILL_NEW, "FusedSDPA prompt call")
    elif (n_untiled, n_tiled) != (0, 1):
        raise Drift(f"FusedSDPA prompt call has an unknown layout (untiled {n_untiled}, tiled {n_tiled})")
    return src


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = (("ops", "blocks_per_seq = torch.matmul(mapping, mapping.sum(0, keepdim=True).t())"),)

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (("vllm_gaudi", "extension/ops.py", patch_ops, "ops"),)


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
