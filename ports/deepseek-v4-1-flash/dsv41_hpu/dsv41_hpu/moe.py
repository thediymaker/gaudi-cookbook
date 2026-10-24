# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 MoE routing and a CPU reference expert path.

route() is the reference Gate (inference/model.py in the checkpoint). vllm-gaudi's MXFP4 MoE method would instead
route non-gpt-oss models with plain softmax and drop the correction bias (vllm_gaudi/ops/hpu_mxfp4.py at v0.30.0),
which costs a large share of perplexity. Every HPU MoE call site routes here.

RefMoE is the dense reference used by the CPU tests and the CPU oracle.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def route(x: torch.Tensor, gate_w: torch.Tensor, bias: torch.Tensor, topk: int, route_scale: float):
    """x [N, D] -> (weights [N, topk] fp32, ids [N, topk] long).
    sqrt(softplus) scores in fp32; the bias only SELECTS (noaux_tc); weights are the
    unbiased scores renormalized with +1e-20 (training's constant, not norm_eps),
    then times route_scale (1.5)."""
    scores = F.softplus(F.linear(x.float(), gate_w.float())).sqrt()
    ids = (scores + bias.float()).topk(topk, dim=-1).indices
    w = scores.gather(1, ids)
    if topk > 1:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return w * route_scale, ids


def swiglu_clamped(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """silu(min(gate, L)) * clamp(up, -L, L) in fp32: up clamped on both sides, gate from above."""
    gate, up = gate.float(), up.float()
    if limit > 0:
        up = up.clamp(-limit, limit)
        gate = gate.clamp(max=limit)
    return F.silu(gate) * up


class RefMoE(nn.Module):
    """Dense reference: routed experts as plain [E, I, D] / [E, D, I] weights + one
    shared expert. Matches MoE.forward/Expert.forward of the reference."""

    def __init__(self, dim, inter, n_experts, topk, route_scale, limit, dtype=torch.bfloat16):
        super().__init__()
        self.topk, self.route_scale, self.limit = topk, route_scale, limit
        self.gate_w = nn.Parameter(torch.empty(n_experts, dim, dtype=dtype), requires_grad=False)
        self.gate_bias = nn.Parameter(torch.zeros(n_experts, dtype=torch.float32), requires_grad=False)
        self.w1 = nn.Parameter(torch.empty(n_experts, inter, dim, dtype=dtype), requires_grad=False)
        self.w3 = nn.Parameter(torch.empty(n_experts, inter, dim, dtype=dtype), requires_grad=False)
        self.w2 = nn.Parameter(torch.empty(n_experts, dim, inter, dtype=dtype), requires_grad=False)
        self.s1 = nn.Parameter(torch.empty(inter, dim, dtype=dtype), requires_grad=False)
        self.s3 = nn.Parameter(torch.empty(inter, dim, dtype=dtype), requires_grad=False)
        self.s2 = nn.Parameter(torch.empty(dim, inter, dtype=dtype), requires_grad=False)

    def _expert(self, x, w1, w3, w2, weight=None):
        h = swiglu_clamped(F.linear(x, w1), F.linear(x, w3), self.limit)
        if weight is not None:
            h = weight * h
        return F.linear(h.to(x.dtype), w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        w, ids = route(x, self.gate_w, self.gate_bias, self.topk, self.route_scale)
        y = torch.zeros_like(x, dtype=torch.float32)
        for e in torch.unique(ids).tolist():
            tok, slot = torch.where(ids == e)
            y[tok] += self._expert(x[tok], self.w1[e], self.w3[e], self.w2[e], w[tok, slot, None]).float()
        y += self._expert(x, self.s1, self.s3, self.s2).float()
        return y.to(x.dtype).reshape(shape)
