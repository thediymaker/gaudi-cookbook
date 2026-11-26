"""DSA lightning indexer: q/k projections, FP8 fake-quant, scores, top-k selection.

Weights dict (Mistral-native names under `layers.N.attention.indexer.`):
  q_norm_w, q_norm_b [d]; wq_b [H*D, d]; wk [D, d]; k_norm_w, k_norm_b [D];
  weights_proj [H, d]; ssmax_s, ssmax_b [H]   (H = 32 heads, D = 128, d = hidden)

Layout: indexer rope dims are the FIRST `rope_dim` of each head ([rope | nope]); MLA's are the last.
"""

from __future__ import annotations

import torch

from . import numerics as N
from .rope import rope_interleaved, rope_neox

F32 = torch.float32


def _rot(x, cos, sin, interleaved):
    return rope_interleaved(x, cos, sin) if interleaved else rope_neox(x, cos, sin)


def indexer_q(
    a, w, lam_idx, cos, sin, use_rope: bool, rope_dim: int, dt, fp8: bool, interleaved: bool = True
) -> torch.Tensor:
    """q [T, H, D] fp32: LayerNorm(a) -> wq_b -> SSMax -> RoPE on dims [0:rope_dim] -> FP8 fake-quant
    (one ue8m0 group per (token, head)). `lam_idx` [T, H] is the SSMax factor (ones to disable)."""
    T = a.shape[0]
    H = w["ssmax_s"].shape[0]
    q = N.linear(N.layernorm(a, w["q_norm_w"], w["q_norm_b"], 1e-6, dt), w["wq_b"], dt).view(T, H, -1)
    q = N.apply_ssmax(q, lam_idx, dt)
    if use_rope:
        q = torch.cat([N.rnd(_rot(q[..., :rope_dim], cos, sin, interleaved), dt), q[..., rope_dim:]], -1)
    if fp8:
        return N.fp8_fakequant_ue8m0(q, q.shape[-1])
    return q.to(F32)


def indexer_k(a, w, cos, sin, use_rope: bool, rope_dim: int, dt, fp8: bool, interleaved: bool = True) -> torch.Tensor:
    """k [T, D] fp32: LayerNorm(wk a) -> RoPE on [0:rope_dim] -> FP8 fake-quant (one group per token).
    This is what the indexer-K cache holds (dequantized)."""
    k = N.layernorm(N.linear(a, w["wk"], dt), w["k_norm_w"], w["k_norm_b"], 1e-6, dt)
    if use_rope:
        k = torch.cat([N.rnd(_rot(k[:, :rope_dim], cos, sin, interleaved), dt), k[:, rope_dim:]], -1)
    if fp8:
        return N.fp8_fakequant_ue8m0(k, k.shape[-1])
    return k.to(F32)


def indexer_weights(a, w, dt) -> torch.Tensor:
    """Per-head weights [T, H] fp32 with both scales folded: w * D^-0.5 * H^-0.5.
    bf16 mode: bf16 GEMM output (PR merges wk and weights_proj into one bf16 GEMM);
    fp32 mode: fp32 (HF keeps weights_proj in fp32)."""
    H = w["weights_proj"].shape[0]
    D = w["wk"].shape[0]
    return N.linear(a, w["weights_proj"], dt).to(F32) * (D**-0.5) * (H**-0.5)


def indexer_scores(
    q: torch.Tensor, k: torch.Tensor, wts: torch.Tensor, q_pos: torch.Tensor, k_pos: torch.Tensor
) -> torch.Tensor:
    """I[t, s] = sum_h wts[t, h] * ReLU(q[t, h] . k[s]), -inf where k_pos > q_pos. fp32 [T, S]."""
    T, H, D = q.shape
    out = torch.zeros(T, k.shape[0], dtype=F32)
    kf = k.to(F32)
    for h0 in range(0, H, 8):
        s = torch.relu(torch.einsum("thd,sd->ths", q[:, h0 : h0 + 8].to(F32), kf))
        out += torch.einsum("ths,th->ts", s, wts[:, h0 : h0 + 8])
    return out.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))


def select_topk(
    scores: torch.Tensor,
    q_pos: torch.Tensor,
    k_pos: torch.Tensor,
    topk: int,
    skip_short: bool = True,
    ties: str = "stable",
) -> torch.Tensor:
    """Selection mask [T, S] bool = causal AND in top-k(t). Queries with pos+1 <= topk select every
    visible key (exact: all of them are in the top-k), the skip the HPU port uses.

    Ties break toward the EARLIER key (stable descending sort). Exact ties are real: a key whose
    q.k is negative on every head scores exactly 0 after the ReLU, so several keys can tie at 0 on
    the top-k boundary; torch.topk then picks among them by an order that depends on the row length,
    which made chunked and single-pass prefill disagree on the tiny model (2 indexer heads).
    ties="torch" uses torch.topk on the row instead (what HF does; only for the HF comparison)."""
    causal = k_pos[None, :] <= q_pos[:, None]
    if scores.shape[1] <= topk:
        return causal
    if ties == "torch":
        idx = scores.topk(topk, dim=-1).indices
    else:
        idx = torch.sort(scores, dim=-1, descending=True, stable=True).indices[:, :topk]
    sel = torch.zeros_like(causal).scatter_(1, idx, True) & causal
    if skip_short:
        short = (q_pos + 1) <= topk
        sel[short] = causal[short]
    return sel


def mask_to_indices(sel: torch.Tensor, topk: int) -> torch.Tensor:
    """[T, S] selection mask -> [T, topk] int64 key rows, ascending, padded with -1 (PR buffer form)."""
    T, S = sel.shape
    ar = torch.arange(S).expand(T, S)
    keyed = torch.where(sel, ar, torch.full_like(ar, S + 1))
    srt = keyed.sort(-1).values[:, :topk]
    if srt.shape[1] < topk:
        srt = torch.cat([srt, torch.full((T, topk - srt.shape[1]), S + 1)], 1)
    return torch.where(srt > S, torch.full_like(srt, -1), srt)


def boundary_ties(scores: torch.Tensor, q_pos: torch.Tensor, topk: int) -> int:
    """Rows (with more than topk visible keys) whose top-k boundary score is tied with a key left out."""
    s = scores[(q_pos + 1) > topk]
    if s.shape[0] == 0 or s.shape[1] <= topk:
        return 0
    v = torch.sort(s, dim=-1, descending=True).values
    return int((v[:, topk - 1] == v[:, topk]).sum())
