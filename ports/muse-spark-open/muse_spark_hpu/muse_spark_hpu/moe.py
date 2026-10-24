# SPDX-License-Identifier: Apache-2.0
"""Muse Spark latent MoE on the native vllm-gaudi MoE ops.

Muse normalises EACH expert's output (post_expert_norm) before the weighted top-k sum, and the HPU fused MoE
op only returns the sum. So every (token, route) pair becomes its own row, routed with top_k = 1 and weight 1,
and one op call per layer returns per-route expert outputs; the norm, the routing weights and the sum then run
here in fp32:

    s = sigmoid(f @ Wr^T * output_multiplier)            fp32 router, K-split GEMM (32 chunks of 256 + fp32 sum)
    ids = top-k(s + e_score_correction_bias)             the bias only selects
    tw = s[ids] / (sum + 1e-15) * routed_scaling_factor  unbiased weights
    p = rms(pre_expert_proj(f)) * (pre_expert_norm + 1)  [T, 4096] bf16 (the "pooler")
    yr = op(p repeated k times, ids as [T*k, 1], ones)   [T*k, 4096] bf16, one row per route
    z = where(route is local, yr, 0)                     fp32
    z = all_reduce(z)                                    complete rows (see the layout below)
    z = rms(z * (post_expert_norm + 1), 1e-8)            per route; zero rows stay zero
    y = sum_k tw * z -> post_expert_proj(y.bf16())       [T, 8192]

The op costs per touched expert slice, not per row, so the expansion costs about as much as a fused top-8 call.

Layout (MUSE_SPARK_MOE_TP_SPLIT = s, 4 in the recipe): EP over tp / s groups of s ranks. Rank r owns the experts
of group r // s and a 1/s slice of each one's intermediate dimension, so its per-route rows are partial sums. They
are all-reduced over the whole TP group before the norm (a norm of a partial sum is wrong; ranks of the other
group add exact zeros). At TP 8: EP2 x TP4. Pure EP8 (s = 1) keeps the norm local but was 33% slower per stream
at batch 1, because every all-reduce waits for the rank whose experts drew the most routes.

MXFP4 expert layers (U8 E2M1 + E8M0, group 32) use VllmMixtureOfExpertsOpMXFP4 (int32 routing); the bf16 expert
layers (0 and 61 of the checkpoint) use VllmMixtureOfExpertsOp (int64 routing). w13 is CONCAT [gate; up]. A plain
fp32 F.linear router misses the fp64 result by 3.6e-6 relative on HPU; the K-split GEMM gives 3.1e-7.

MUSE_SPARK_MXFP4_KERNELS=1 (mxfp4_kernels.py): pure decode steps of up to 4 tokens on MXFP4 layers run the expert
call on two TPC kernels that read the packed weights and return the same per-route rows.
"""

from __future__ import annotations

import logging
import os

import torch
from torch import nn
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.linear import ReplicatedLinear

from . import mxfp4_kernels as mxk
from .numerics import gain, rms

logger = logging.getLogger("vllm.muse_spark_hpu")

MXFP4_BLOCK = 32
KSPLIT = 256


def router_logits(f: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """fp32 f @ w^T as one batched GEMM over 256-wide K chunks plus an fp32 sum of the partials.

    Same FLOPs as one GEMM; the matrix engine's error grows with the accumulation length."""
    T, K = f.shape
    if K % KSPLIT or K == KSPLIT:
        return torch.nn.functional.linear(f.float(), w)
    C = K // KSPLIT
    xs = f.float().reshape(T, C, KSPLIT).transpose(0, 1)  # [C, T, 256]
    ws = w.reshape(w.shape[0], C, KSPLIT).permute(1, 2, 0)  # [C, 256, E]
    return torch.bmm(xs, ws).sum(0)


class _Gate(nn.Module):
    """Router weight and selection bias (checkpoint mlp.gate.*), fp32, replicated."""

    def __init__(self, E: int, H: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(E, H, dtype=torch.float32), requires_grad=False)
        self.e_score_correction_bias = nn.Parameter(torch.empty(E, dtype=torch.float32), requires_grad=False)


class _NormWeight(nn.Module):
    """A raw fp32 norm weight; the gain is built after loading."""

    def __init__(self, n: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, dtype=torch.float32), requires_grad=False)


class MuseSparkMoEHPU(nn.Module):
    def __init__(self, tcfg, layer_id: int, kind: str, layout, quant_config, prefix: str):
        super().__init__()
        H, Hm, inter, E = tcfg.hidden_size, tcfg.moe_hidden_size, tcfg.moe_intermediate_size, tcfg.num_local_experts
        self.layer_id, self.kind, self.layout = layer_id, kind, layout
        self.Hm, self.E, self.k = Hm, E, tcfg.num_experts_per_tok
        self.mult = float(tcfg.output_multiplier)
        self.scale = float(getattr(tcfg, "routed_scaling_factor", 1.0) or 1.0)
        self.eps, self.post_eps = tcfg.rms_norm_eps, tcfg.post_norm_eps
        self.split = layout.moe_tp_split
        self.tp = layout.tp_size
        self.e0, self.n = layout.local_experts(E)
        Is = inter // self.split
        self.gate = _Gate(E, H)
        self.pre_expert_proj = ReplicatedLinear(
            H, Hm, bias=False, quant_config=quant_config, prefix=f"{prefix}.pre_expert_proj"
        )
        self.post_expert_proj = ReplicatedLinear(
            Hm, H, bias=False, quant_config=quant_config, prefix=f"{prefix}.post_expert_proj"
        )
        self.pre_expert_norm = _NormWeight(Hm)
        self.post_expert_norm = _NormWeight(Hm)
        n = self.n
        if kind == "mxfp4":
            u8 = {"dtype": torch.uint8}
            self.w13_weight = nn.Parameter(torch.empty(n, 2 * Is, Hm // 2, **u8), requires_grad=False)
            self.w13_weight_scale = nn.Parameter(torch.empty(n, 2 * Is, Hm // MXFP4_BLOCK, **u8), requires_grad=False)
            self.w2_weight = nn.Parameter(torch.empty(n, Hm, Is // 2, **u8), requires_grad=False)
            self.w2_weight_scale = nn.Parameter(torch.empty(n, Hm, Is // MXFP4_BLOCK, **u8), requires_grad=False)
        elif kind == "bf16":
            self.w13_weight = nn.Parameter(torch.empty(n, 2 * Is, Hm, dtype=torch.bfloat16), requires_grad=False)
            self.w2_weight = nn.Parameter(torch.empty(n, Hm, Is, dtype=torch.bfloat16), requires_grad=False)
        else:
            raise ValueError(f"layer {layer_id}: unknown expert kind {kind}")
        self.moe_op = None
        self._kernel_weights = None  # (w13, s13, w2, s2) when the decode kernels serve this layer

    def finalize(self) -> None:
        """After loading: fp32 gains and the op with one per-expert view of every weight and scale."""
        from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOp, VllmMixtureOfExpertsOpMXFP4

        dev = self.w13_weight.device
        self.g_pre = gain(self.pre_expert_norm.weight.data.cpu(), 1.0).to(dev)
        self.g_post = gain(self.post_expert_norm.weight.data.cpu(), 1.0).to(dev)
        lo, hi = self.e0, self.e0 + self.n - 1
        if self.kind == "mxfp4":
            op = VllmMixtureOfExpertsOpMXFP4(self.E, self.n, lo, hi, block_size=MXFP4_BLOCK, has_bias=False)
            for e in range(self.n):
                op.w13_list[e].set_weight(self.w13_weight.data[e])
                op.w13_list[e].set_scale(self.w13_weight_scale.data[e])
                op.w2_list[e].set_weight(self.w2_weight.data[e])
                op.w2_list[e].set_scale(self.w2_weight_scale.data[e])
            self.ids_dtype = torch.int32
        else:
            op = VllmMixtureOfExpertsOp(self.E, self.n, lo, hi)
            for e in range(self.n):
                op.w13_list[e].set_weight(self.w13_weight.data[e])
                op.w2_list[e].set_weight(self.w2_weight.data[e])
            self.ids_dtype = torch.int64
            if self.n < self.E:
                # The bf16 op with VLLM_MOE_CHUNK kwargs over a partial expert range returned wrong rows for the
                # last local expert inside the engine (not reproduced standalone); without them it is exact.
                op._get_extra_kwargs = lambda tokens_num: {}
        op._cache_weight_lists()
        self.moe_op = op
        if mxk.ENABLED and self.kind == "mxfp4":
            self._kernel_setup()

    def _kernel_setup(self) -> None:
        w13, s13 = self.w13_weight.data, self.w13_weight_scale.data
        w2, s2 = self.w2_weight.data, self.w2_weight_scale.data
        n, n13, half_hm = w13.shape
        Hm, Is = half_hm * 2, n13 // 2
        why = mxk.unsupported(n, Hm, Is, self.k)
        if why is None and not mxk.env_ok():
            why = "kernel library not registered before the device opened (patches/kernel_env_hook.py missing?)"
        if why is None:
            mxk.load_ops()
            self._kernel_weights = (w13, s13, w2, s2)
        if why is not None or not getattr(MuseSparkMoEHPU, "_kernel_logged", False):
            MuseSparkMoEHPU._kernel_logged = True
            logger.warning(
                "muse_spark_hpu: layer %d MXFP4 decode kernels %s (%s; %d local experts, <= %d tokens, "
                "GC_KERNEL_PATH=%s, recipe cache %s)",
                self.layer_id,
                "ON" if why is None else "OFF, stock op kept: " + why,
                mxk.lib_version(),
                n,
                mxk.MAX_TOKENS,
                os.environ.get("GC_KERNEL_PATH"),
                os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG"),
            )

    def forward(self, f: torch.Tensor, decode: bool = False) -> torch.Tensor:
        """f [T, H] bf16 -> [T, H] bf16. decode: the step is a pure decode batch ([B, 1] inputs)."""
        T, k, Hm = f.shape[0], self.k, self.Hm
        s = torch.sigmoid(router_logits(f, self.gate.weight) * self.mult)
        ids = torch.topk(s + self.gate.e_score_correction_bias, k, dim=-1).indices
        tw = s.gather(1, ids)
        tw = tw / (tw.sum(-1, keepdim=True) + 1e-15) * self.scale
        p = (rms(self.pre_expert_proj(f)[0], self.eps) * self.g_pre).to(f.dtype)
        if self._kernel_weights is not None and decode and T <= mxk.MAX_TOKENS:
            # rows of routes whose expert is on the other EP group come back as exact zeros: no mask needed
            w13, s13, w2, s2 = self._kernel_weights
            idf = ids.reshape(T * k).to(torch.int32)
            ap = torch.ops.custom_op.mxk_w13x(
                mxk.permute(p), idf, w13, s13, k, self.e0, 1.0, mxk.NO_LIMIT, 0.0, 1.0, False
            )
            z = torch.ops.custom_op.mxk_w2xr(ap, idf, w2, s2, 1, self.e0).float()
        else:
            xr = p.unsqueeze(1).expand(T, k, Hm).reshape(T * k, Hm)
            ir = ids.reshape(T * k, 1).to(self.ids_dtype)
            ones = torch.ones(T * k, 1, dtype=f.dtype, device=f.device)
            yr = self.moe_op(xr, ir, ones, permuted_weights=True, activation="silu")
            local = (ir >= self.e0) & (ir < self.e0 + self.n)
            z = torch.where(local, yr, torch.zeros_like(yr)).float()
        if self.split > 1:
            z = tensor_model_parallel_all_reduce(z)  # complete per-route rows on every rank
        z = rms(z * self.g_post, self.post_eps)
        y = (z.view(T, k, Hm) * tw.unsqueeze(-1)).sum(1)
        if self.tp > 1 and self.split == 1:
            y = tensor_model_parallel_all_reduce(y)
        return self.post_expert_proj(y.to(f.dtype))[0]
