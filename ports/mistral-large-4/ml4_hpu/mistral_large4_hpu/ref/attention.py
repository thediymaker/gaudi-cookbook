"""Sparse MLA: projections, latent cache entries, both attention forms.

Weights dict (Mistral-native names under `layers.N.attention.`):
  wq [H*128, d]; wkv_a [512 (+64 if RoPE), d] and kv_a_norm [512] (absent on KV-shared layers);
  wkv_b [H*(P+V), 512] with per-head rows [k_nope (P) ; v (V)], P = 64 (RoPE) or 128 (NoPE);
  wo [d, H*V]; ssmax_s, ssmax_b [H].

Query head layout [nope (P) | rope (64)] for RoPE layers, [nope (128)] for NoPE layers.
Softmax scale is qk_head_dim^-0.5 = 128^-0.5 on every layer (NoPE included) (no YaRN mscale; the `mscale` option is the control).
"""

from __future__ import annotations

import math

import torch

from . import numerics as N
from .rope import rope_interleaved, rope_neox

F32 = torch.float32
MSCALE_SQ = (0.1 * 1.0 * math.log(2048.0) + 1.0) ** 2  # 3.10627: the trap


def _rot(x, cos, sin, interleaved):
    return rope_interleaved(x, cos, sin) if interleaved else rope_neox(x, cos, sin)


def mla_q(
    a,
    w,
    use_rope: bool,
    n_heads: int,
    rope_dim: int,
    lam,
    cos,
    sin,
    dt,
    interleaved: bool = True,
    rope_on_nope: bool = False,
) -> torch.Tensor:
    """q [T, H, 128] (dt): wq a, RoPE on the last `rope_dim` dims (RoPE layers), then SSMax
    (`lam` [T, H]; pass ones for --no-ssmax). NoPE layers do not rotate (unless rope_on_nope)."""
    T = a.shape[0]
    q = N.linear(a, w["wq"], dt).view(T, n_heads, -1)
    if use_rope or rope_on_nope:
        q = torch.cat([q[..., :-rope_dim], N.rnd(_rot(q[..., -rope_dim:], cos, sin, interleaved), dt)], -1)
    return N.apply_ssmax(q, lam, dt)


def mla_kv(
    a, w, use_rope: bool, kv_lora: int, rope_dim: int, cos, sin, dt, eps: float, interleaved: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Latent cache entry of a KV-owning layer: c [T, 512] = RMSNorm(wkv_a a)[:512] and
    k_pe [T, 64] = RoPE(wkv_a a [512:]) (zeros on NoPE layers, as the PR pads)."""
    kv = N.linear(a, w["wkv_a"], dt)
    c = N.rmsnorm(kv[:, :kv_lora], w["kv_a_norm"], eps, dt)
    if use_rope:
        kpe = N.rnd(_rot(kv[:, kv_lora:], cos, sin, interleaved), dt)
    else:
        kpe = torch.zeros(a.shape[0], rope_dim, dtype=dt)
    return c, kpe


def _split_kvb(wkv_b, n_heads, v_dim):
    kvb = wkv_b.to(F32).view(n_heads, -1, wkv_b.shape[1])
    P = kvb.shape[1] - v_dim
    return kvb[:, :P], kvb[:, P:], P  # W_UK [H, P, c], W_UV [H, V, c]


def _softmax(logits, bf16_softmax):
    if bf16_softmax:
        return torch.softmax(logits.to(torch.bfloat16), -1).to(F32)
    return torch.softmax(logits, -1)


def attend_unabsorbed(
    q,
    c,
    kpe,
    wkv_b,
    sel,
    v_dim: int,
    dt,
    use_rope: bool,
    mscale: bool = False,
    bf16_softmax: bool = False,
    rope_on_nope_kcs=None,
    interleaved: bool = True,
    head_chunk: int = 16,
) -> torch.Tensor:
    """Reference form (HF): expand k_nope, v through this layer's own wkv_b, dense softmax over the
    selected keys. q [T, H, 128]; c [S, 512], kpe [S, 64] (the source layer's cache rows);
    sel [T, S] bool. Returns o [T, H*V] (dt)."""
    T, H, Dq = q.shape
    S = c.shape[0]
    scale = Dq**-0.5 * (MSCALE_SQ if mscale else 1.0)  # q head = qk_head_dim on every layer
    W_UK, W_UV, P = _split_kvb(wkv_b, H, v_dim)
    kv = N.rnd(c.to(F32) @ torch.cat([W_UK, W_UV], 1).reshape(-1, c.shape[1]).t(), dt).view(S, H, -1)
    k_nope, v = kv[..., :P], kv[..., P:]
    if use_rope:
        k = torch.cat([k_nope, kpe.to(dt)[:, None, :].expand(S, H, kpe.shape[1])], -1)
    else:
        k = k_nope
        if rope_on_nope_kcs is not None:  # control: rotate NoPE keys too
            kc, ks = rope_on_nope_kcs
            rd = kpe.shape[1]
            k = torch.cat([k[..., :-rd], N.rnd(_rot(k[..., -rd:], kc, ks, interleaved), dt)], -1)
    neg = ~sel
    out = torch.empty(T, H, v_dim, dtype=F32)
    for h0 in range(0, H, head_chunk):
        h1 = min(H, h0 + head_chunk)
        lg = torch.einsum("thd,shd->hts", q[:, h0:h1].to(F32), k[:, h0:h1].to(F32)) * scale
        p = N.rnd(_softmax(lg.masked_fill(neg[None], float("-inf")), bf16_softmax), dt)
        out[:, h0:h1] = torch.einsum("hts,shd->thd", p.to(F32), v[:, h0:h1].to(F32))
    return N.rnd(out, dt).reshape(T, H * v_dim)


def attend_absorbed(
    q,
    c,
    kpe,
    wkv_b,
    sel,
    v_dim: int,
    dt,
    use_rope: bool,
    mscale: bool = False,
    bf16_softmax: bool = False,
    head_chunk: int = 16,
) -> torch.Tensor:
    """PR form (MQA in latent space): ql = W_UK^T q_nope, logits over [c | k_pe], o = W_UV (p @ c).
    Same arguments and result as attend_unabsorbed (equal in exact arithmetic)."""
    T, H, Dq = q.shape
    scale = Dq**-0.5 * (MSCALE_SQ if mscale else 1.0)
    W_UK, W_UV, P = _split_kvb(wkv_b, H, v_dim)
    rd = kpe.shape[1]
    q_nope = q[..., :P].to(F32)
    q_pe = q[..., P:].to(F32) if use_rope else torch.zeros(T, H, rd)
    ql = N.rnd(torch.einsum("thp,hpc->thc", q_nope, W_UK), dt).to(F32)
    Kc = torch.cat([c.to(F32), kpe.to(F32)], -1)  # [S, 576]
    neg = ~sel
    out = torch.empty(T, H, v_dim, dtype=F32)
    for h0 in range(0, H, head_chunk):
        h1 = min(H, h0 + head_chunk)
        Q = torch.cat([ql[:, h0:h1], q_pe[:, h0:h1]], -1)
        lg = torch.einsum("thd,sd->hts", Q, Kc) * scale
        p = N.rnd(_softmax(lg.masked_fill(neg[None], float("-inf")), bf16_softmax), dt)
        ol = N.rnd(torch.einsum("hts,sc->thc", p.to(F32), c.to(F32)), dt).to(F32)
        out[:, h0:h1] = torch.einsum("thc,hvc->thv", ol, W_UV[h0:h1])
    return N.rnd(out, dt).reshape(T, H * v_dim)
