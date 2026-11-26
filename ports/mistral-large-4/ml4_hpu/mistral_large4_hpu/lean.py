"""Lean decode rows for Mistral Large 4 (ML4_DECODE_LEAN=1 or control key decode_lean).

The decode step is device-bound on node count, not on math: a 1-stream step ran ~11,000 device nodes in the 9 layer
graphs, with the compute engines busy for ~19 of its ~45 ms (rank-0 trace). This module gives
decode rows (S <= DECODE_S, the rows the layer graphs carry) the same functions with fewer nodes. Prompt chunks keep
attention.py / indexer.py / stack.py unchanged, so prompt logprobs (PPL) do not move.

  * one GEMV per layer for everything that reads the attention input `a`: wq | wkv_a | indexer wk | indexer
    weights_proj. prepare() concatenates the weights once and points each module's weight at its rows of the
    concatenation (no extra memory; prompt chunks use the same rows through the original modules);
  * RoPE as x * C + swap_pairs(x) * S on the whole head (no slices, stack or cat): C / S per position are built once
    per decode graph (BatchCtx memo) with ones / zeros on the dims that do not rotate. y_2i = x_2i cos - x_2i+1 sin
    is the same IEEE operations as x_2i cos + x_2i+1 (-sin);
  * RMS statistics as sum x (1/n) (the device's mean is that constant, a two-stage sum and a multiply);
  * SSMax: log(p + 1) rounded to bf16 once per graph instead of once per layer and indexer.
Measured: +3.8% at 1 stream, and NOT bit-identical to the plain path on the HPU
(greedy tokens differ after 3-81 tokens on 6 of 8 prompts); off by default. Two more sub-levers were measured and
removed: the router's top-4 as four max + mask steps (bit-identical, no gain) and the absorbed attention with the
logits as [keys, heads] (not identical, slower).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

from .attention import FAULTS, HEAD_DIM, KV_RANK, RMS_EPS, ROPE_DIM
from .indexer import N_HEADS, NORM_EPS, short_topk
from .indexer import HEAD_DIM as IDX_DIM
from .numerics import fake_fp8_ue8m0, layer_norm, round_bf16, ssmax_apply

ENABLED = os.environ.get("ML4_DECODE_LEAN", "0") == "1"


def enabled() -> bool:
    from .control import get as ctl_get

    return ctl_get("decode_lean", ENABLED)


# ------------------------------------------------------------------ numerics
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """numerics.rms_norm with the mean written as sum x (1/n)."""
    xf = x.float()
    y = round_bf16(xf * torch.rsqrt(xf.square().sum(-1, keepdim=True) * (1.0 / xf.shape[-1]) + eps))
    return (y * weight.float()).to(x.dtype)


def ssmax_lambda(lpb: torch.Tensor, s: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """numerics.ssmax_lambda with log(p + 1) already rounded to bf16 (lpb, fp32 [B, S])."""
    prod = round_bf16(lpb.unsqueeze(-1) * s.to(torch.bfloat16).float())
    return (prod + b.to(torch.bfloat16).float()).to(torch.bfloat16)


def _swap(xf: torch.Tensor) -> torch.Tensor:
    return xf.unflatten(-1, (-1, 2)).flip(-1).flatten(-2)


def rot(x: torch.Tensor, C: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """x * C + swap_pairs(x) * S in fp32, one rounding to x.dtype (rope.rotate / rope_tail / rope_head)."""
    xf = x.float()
    return (xf * C + _swap(xf) * S).to(x.dtype)


def tables(ctx, rope):
    """Per-position RoPE multipliers and the rounded log(p + 1), once per forward (memo)."""

    def make():
        cos, sin, lp = rope.at(ctx.positions)  # [B, S, 32] fp32
        C = torch.stack([cos, cos], -1).flatten(-2)  # [B, S, 64]
        S = torch.stack([-sin, sin], -1).flatten(-2)
        one, zero = torch.ones_like(C), torch.zeros_like(C)
        return {
            "C": C,
            "S": S,
            "Ct": torch.cat([one, C], -1),
            "St": torch.cat([zero, S], -1),
            "Ch": torch.cat([C, one], -1),
            "Sh": torch.cat([S, zero], -1),
            "lpb": round_bf16(lp),
        }

    return ctx.shared(("lean_tables",), make)


# ------------------------------------------------------------------ weights
def prepare(stack) -> int:
    """Concatenate the weights that read the attention input, once; each module keeps a view of its rows."""
    n = 0
    for layer in stack.layers:
        m = layer.attn
        if getattr(m, "lean_w", None) is not None:
            continue
        mods = [m.wq] + ([m.wkv_a] if m.role.owns_kv else [])
        if m.indexer is not None:
            mods += [m.indexer.wk, m.indexer.weights_proj]
        W = torch.cat([md.weight.data for md in mods], 0)
        off = 0
        for md in mods:
            r = md.weight.shape[0]
            md.weight = nn.Parameter(W[off : off + r], requires_grad=False)
            off += r
        m.lean_w = W
        n += 1
    return n


def ready(layer) -> bool:
    return getattr(layer.attn, "lean_w", None) is not None


# ------------------------------------------------------------------ forward
def layer_forward(layer, x, ctx, caches, shared, rope, allreduce):
    """stack.ML4Layer.forward for decode rows."""
    h = attn_forward(layer.attn, rms_norm(x, layer.attn_norm, RMS_EPS), ctx, caches, shared, rope)
    x = x + allreduce(h)
    y = layer.ffn(rms_norm(x, layer.ffn_norm, RMS_EPS))
    return x + allreduce(y)


def attn_forward(m, a, ctx, caches_all, shared, rope):
    """attention.ML4Attention.forward for decode rows (absorbed form)."""
    role = m.role
    pos, valid = ctx.positions, ctx.valid
    t = tables(ctx, rope)
    y = F.linear(a, m.lean_w)
    nq = m.Hl * HEAD_DIM
    q = y[..., :nq].unflatten(-1, (m.Hl, HEAD_DIM))
    if role.use_rope:
        q = rot(q, t["Ct"].unsqueeze(-2), t["St"].unsqueeze(-2))
    q = ssmax_apply(q, ssmax_lambda(t["lpb"], m.ssmax_s, m.ssmax_b))
    off = nq
    if role.owns_kv:
        w = KV_RANK + (ROPE_DIM if role.use_rope else 0)
        kv = y[..., off : off + w]
        off += w
        c = rms_norm(kv[..., :KV_RANK], m.kv_a_norm, RMS_EPS)
        if role.use_rope:
            kpe = rot(kv[..., KV_RANK:], t["C"], t["S"])
        else:
            kpe = torch.zeros(*c.shape[:-1], ROPE_DIM, dtype=c.dtype, device=c.device)
        caches_all[role.layer_id]["latent"].write(ctx, pos, torch.cat([c, kpe], -1), valid, key="pos")
    if m.indexer is not None:
        shared.topk[role.layer_id] = index_forward(
            m.indexer,
            a,
            y[..., off : off + IDX_DIM],
            y[..., off + IDX_DIM : off + IDX_DIM + N_HEADS],
            ctx,
            caches_all[role.layer_id],
            t,
        )
    topk = shared.topk[role.index_source]
    src = caches_all[role.kv_source]["latent"]
    if FAULTS["no_kv_share"] and not role.owns_kv:
        from .paged import TaggedCache

        src = TaggedCache(torch.zeros_like(src.data), torch.full_like(src.tags, -1), "latent", role.layer_id)
    o = m.absorbed(q, topk, ctx, src)
    out = m.wo(o.flatten(2))
    return out[0] if isinstance(out, tuple) else out


def index_forward(ix, a, k_raw, w_raw, ctx, caches, t):
    """indexer.ML4Indexer.forward for decode rows; k_raw / w_raw are its wk / weights_proj rows of the merged GEMV."""
    pos, valid = ctx.positions, ctx.valid
    k = layer_norm(k_raw, ix.k_norm_w, ix.k_norm_b, NORM_EPS)
    if ix.use_rope:
        k = rot(k, t["Ch"], t["Sh"])
    k = fake_fp8_ue8m0(k)
    caches["index_k"].write(ctx, pos, k, valid, key="pos")
    k_out = min(ix.topk, ctx.max_len)
    host_max = getattr(ctx, "host_max_pos", None)
    if host_max is not None and host_max <= k_out - 1:
        return short_topk(ctx, k_out, caches["index_k"])
    q = ix.wq_b(layer_norm(a, ix.q_norm_w, ix.q_norm_b, NORM_EPS))
    q = (q[0] if isinstance(q, tuple) else q).unflatten(-1, (N_HEADS, IDX_DIM))
    q = ssmax_apply(q, ssmax_lambda(t["lpb"], ix.ssmax_s, ix.ssmax_b))
    if ix.use_rope:
        q = rot(q, t["Ch"].unsqueeze(-2), t["Sh"].unsqueeze(-2))
    w = w_raw.float() * (IDX_DIM**-0.5 * N_HEADS**-0.5)
    from . import k1ix

    if k1ix.ENABLED and k1ix.eligible(pos.shape[1], ctx.max_len, k_out, caches["index_k"]):
        return k1ix.select(fake_fp8_ue8m0(q), w, ctx, caches["index_k"], k_out)  # ML4_INDEX_KERNEL=1
    return ix.select(fake_fp8_ue8m0(q), w, ctx, caches["index_k"], k_out)


__all__ = ["enabled", "prepare", "ready", "layer_forward", "rms_norm", "rot", "tables"]
