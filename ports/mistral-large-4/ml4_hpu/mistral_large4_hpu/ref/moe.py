"""MoE and dense FFN: softmax top-4 router with renormalisation, SiLU-gated experts,
one always-on shared expert. Routed experts come from an ExpertSource (MXFP4 bytes dequantized
per use = W4A16, or FP8/BF16 for the tiny model); `act_mxfp4=True` adds MXFP4 activations (W4A4).
"""

from __future__ import annotations

import torch

from . import numerics as N

F32 = torch.float32


def route(
    f: torch.Tensor, gate_w: torch.Tensor, top_k: int, renormalize: bool = True, scaling: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 router logits (bf16 values in, fp32 out), softmax over all experts, top-k,
    g = p / sum(p over top-k) (stock-routing control: renormalize=False).
    Returns (ids [T, k] int64, weights [T, k] fp32, probs [T, E] fp32)."""
    logits = f.to(F32) @ gate_w.to(F32).t()
    p = logits.softmax(-1)
    w, ids = p.topk(top_k, -1)
    if renormalize:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return ids, w * scaling, p


def swiglu_ffn(x, w1, w3, w2, dt, act_mxfp4: bool = False) -> torch.Tensor:
    """W2 (SiLU(W1 x) * W3 x). Weights [out, in]."""
    if act_mxfp4:
        x = N.act_mxfp4(x)
    h = N.silu_mul(N.linear(x, w1, dt), N.linear(x, w3, dt), dt)
    if act_mxfp4:
        h = N.act_mxfp4(h)
    return N.linear(h, w2, dt)


def moe_forward(
    f,
    gate_w,
    experts,
    shared,
    top_k: int,
    dt,
    renormalize: bool = True,
    scaling: float = 1.0,
    act_mxfp4: bool = False,
    return_routing: bool = False,
):
    """y = sum_e g_e * Expert_e(f) + Shared(f).

    experts: callable e -> (w1, w3, w2) fp32/bf16 tensors (an ExpertSource);
    shared: dict w1, w3, w2. The routed sum accumulates in fp32 and is rounded once."""
    ids, wts, _ = route(f, gate_w, top_k, renormalize, scaling)
    acc = torch.zeros(f.shape[0], f.shape[1], dtype=F32)
    for e in torch.unique(ids).tolist():
        tok, slot = torch.where(ids == e)
        w1, w3, w2 = experts(e)
        y = swiglu_ffn(f[tok], w1, w3, w2, dt, act_mxfp4)
        acc.index_add_(0, tok, y.to(F32) * wts[tok, slot, None])
    y = N.rnd(acc, dt)
    s = swiglu_ffn(f, shared["w1"], shared["w3"], shared["w2"], dt)
    out = N.rnd(y.to(F32) + s.to(F32), dt)
    return (out, ids, wts) if return_routing else out
