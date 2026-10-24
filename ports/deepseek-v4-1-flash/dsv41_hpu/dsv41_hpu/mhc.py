# SPDX-License-Identifier: Apache-2.0
"""Single-pass (lagged) manifold hyper-connections, as DeepSeek-V4.1's reference Block.

The residual stream is hc_mult=4 copies. Each sublayer computes (pre, post, comb) from its own input stream, applies
its OWN post/comb, but collapses its input with the PREVIOUS sublayer's pre (reference model.py, Block.forward):
attention uses the previous layer's FFN pre (one-hot for layer 0), the FFN uses this layer's attention pre, and the
final collapse before the head uses layer 39's FFN pre. There is no hc_head (that was V4).

Switches (off by default; same values as the reference loop up to fp32 summation order):
  DSV41_MHC_SLICED=1         Sinkhorn row/column sums as elementwise adds of slices instead of reductions: on HPU
                             every reduction is its own kernel launch (~100 per layer at hc=4), while slice adds fuse
                             with the divisions (~1e-7 from the reduction form).
  DSV41_HC_SINKHORN=kernel   the whole Sinkhorn loop as ONE TPC kernel (kernels/dsv41_sinkhorn/) instead of ~40
                             dependent tiny nodes per call (80 calls per decode step). Same fp32 arithmetic in the same
                             order as the sliced form; the division is the TPC compiler's. Needs the kernel library
                             registered (DSV41_MXK=1, mxk.py), otherwise it logs why and keeps the torch loop. Calls of
                             at most SINKHORN_KERNEL_MAX matrices (one per token): decode steps and short prompts. Full
                             4096-token chunks keep the loop: there the kernel was 4% faster but raised the chunk's
                             transient HBM by ~1.7 GiB.
  DSV41_MHC_TPC_DOT=1        for decode-sized inputs (<= 16 rows) the mix projection x [N, 4 x 5120] fp32 . fn
                             [24, 20480] fp32 as a multiply + row sum on the TPC instead of an fp32 GEMM on the MME
                             (~54 -> ~8 us per call, 80 calls per decode step; max relative error vs fp64 1.9e-7
                             instead of 3.4e-6).
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F

from .numerics import mean_sq

SINKHORN_SLICED = os.environ.get("DSV41_MHC_SLICED", "0") == "1"
SINKHORN_KERNEL = os.environ.get("DSV41_HC_SINKHORN", "torch") == "kernel"
SINKHORN_KERNEL_MAX = 1024
MHC_TPC_DOT = os.environ.get("DSV41_MHC_TPC_DOT", "0") == "1"
MHC_TPC_DOT_MAX_ROWS = 16
_kernel_state = {"ok": None}


def _kernel_ok(device) -> bool:
    if _kernel_state["ok"] is None:
        ok, why = False, ""
        if device.type != "hpu":
            why = "not on HPU"
        else:
            from . import mxk

            if not (mxk.ENABLED and mxk.env_ok()):
                why = "DSV41_MXK kernel library not registered"
            else:
                mxk.load_ops()
                ok = hasattr(torch.ops.custom_op, "dsv41_sinkhorn")
                why = "" if ok else "op library lacks dsv41_sinkhorn"
        _kernel_state["ok"] = ok
        logging.getLogger("vllm.dsv41_hpu").warning(
            "dsv41: mHC Sinkhorn %s", "TPC kernel" if ok else "torch (DSV41_HC_SINKHORN=kernel ignored: " + why + ")"
        )
    return _kernel_state["ok"]


def sinkhorn_kernel(comb: torch.Tensor, iters: int, hc_eps: float) -> torch.Tensor:
    """comb [..., 4, 4] fp32 (softmax + eps applied) -> the sliced loop's result, in one kernel."""
    lead = comb.shape[:-2]
    ct = comb.reshape(-1, 16).t().contiguous()  # [16, N]: one matrix per lane
    return torch.ops.custom_op.dsv41_sinkhorn(ct, iters, hc_eps).t().reshape(*lead, 4, 4)


def _row_sums(c: torch.Tensor, hc: int) -> torch.Tensor:
    s = c[..., 0:1]
    for j in range(1, hc):
        s = s + c[..., j : j + 1]
    return s


def _col_sums(c: torch.Tensor, hc: int) -> torch.Tensor:
    s = c[..., 0:1, :]
    for i in range(1, hc):
        s = s + c[..., i : i + 1, :]
    return s


def hc_mixes(
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc: int,
    iters: int,
    hc_eps: float,
    norm_eps: float,
):
    """x [..., hc, D] -> pre [..., hc], post [..., hc], comb [..., hc, hc] (fp32).
    Block.hc_mixes + kernel.hc_split_sinkhorn."""
    xf = x.flatten(-2).float()
    rs = torch.rsqrt(mean_sq(xf) + norm_eps)
    if MHC_TPC_DOT and xf[..., :1].numel() <= MHC_TPC_DOT_MAX_ROWS:
        mixes = (xf.unsqueeze(-2) * fn.float()).sum(-1) * rs
    else:
        mixes = F.linear(xf, fn.float()) * rs
    scale, base = scale.float(), base.float()
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + hc_eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = torch.softmax(comb, dim=-1) + hc_eps
    if SINKHORN_KERNEL and hc == 4 and comb[..., 0, 0].numel() <= SINKHORN_KERNEL_MAX and _kernel_ok(comb.device):
        return pre, post, sinkhorn_kernel(comb, iters, hc_eps)
    if SINKHORN_SLICED:
        comb = comb / (_col_sums(comb, hc) + hc_eps)
        for _ in range(iters - 1):
            comb = comb / (_row_sums(comb, hc) + hc_eps)
            comb = comb / (_col_sums(comb, hc) + hc_eps)
        return pre, post, comb
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    return pre, post, comb


def hc_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """[..., hc, D] x [..., hc] -> [..., D]"""
    return (pre.unsqueeze(-1) * x.float()).sum(-2).to(x.dtype)


def hc_post(y: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """[..., D], [..., hc, D], [..., hc], [..., hc, hc] -> [..., hc, D]:
    out_k = post_k * y + sum_j comb[j, k] * residual_j"""
    out = post.unsqueeze(-1) * y.unsqueeze(-2) + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(-3)
    return out.to(y.dtype)


def identity_pre(like: torch.Tensor, hc: int) -> torch.Tensor:
    pre = torch.zeros(*like.shape[:-2], hc, dtype=torch.float32, device=like.device)
    pre[..., 0] = 1.0
    return pre
