# SPDX-License-Identifier: Apache-2.0
"""Beam feed-forward blocks on vllm-gaudi.

Dense SwiGLU (mlp_layer_types "dense", layers 0-2):
  gate_up = gate_up_proj(f)      MergedColumnParallelLinear (checkpoint mlp.gate_proj, mlp.up_proj), FP8
  m = silu(g) * u                fp32, one bf16 rounding
  y = down_proj(m)               RowParallelLinear, all-reduce

Sparse MoE ("sparse", layers 3-51): stock vLLM FusedMoEFactory, built as vLLM PR #62074 builds it:
  router  GateLinear(H, E), fp32 weight and output (checkpoint mlp.gate.weight, unquantized). The
          correction bias is close to a constant of about 28 across experts, so bf16 would lose
          the differences that pick the experts: routing must stay fp32 end to end.
  bias    gate.e_score_correction_bias, fp32 [E], used for expert selection only
  routing grouped top-k: sigmoid, + bias, group score = sum of the top 2, topk_group of n_group
          groups, top-k inside them, unbiased weights renormalised. The compressed-tensors FP8 HPU
          method routes grouped top-k with vLLM's torch select_experts in fp32
  experts the compressed-tensors FP8 per-channel MoE op, dynamic per-token activation scales
          (the cookbook fix moe-explicit-overload calls its overload directly)
  scale   the bf16 MoE output * routed_scaling_factor (2.5), then the all-reduce
"""

from __future__ import annotations

import torch
from torch import nn
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, RowParallelLinear

from . import numerics as N


class BeamHPUMLP(nn.Module):
    """Dense SwiGLU MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, quant_config, prefix: str):
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(f"beam_hpu: unsupported activation {hidden_act!r}")
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False, quant_config=quant_config, prefix=f"{prefix}.gate_up_proj"
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(N.silu_mul(self.gate_up_proj(x)[0], x.dtype))[0]


class BeamHPUSparseMoeBlock(nn.Module):
    """Grouped sigmoid noaux_tc MoE, no shared expert."""

    def __init__(self, vllm_config, prefix: str):
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config
        if (cfg.moe_router_dtype or "float32") != "float32":
            raise ValueError(f"beam_hpu: moe_router_dtype {cfg.moe_router_dtype!r} is not supported (fp32)")
        if cfg.scoring_func != "sigmoid" or cfg.topk_method != "noaux_tc":
            raise ValueError(f"beam_hpu: routing {cfg.scoring_func}/{cfg.topk_method} is not supported")
        self.gate = GateLinear(
            cfg.hidden_size,
            cfg.num_experts,
            bias=False,
            params_dtype=torch.float32,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        bias = nn.Parameter(torch.zeros(cfg.num_experts, dtype=torch.float32), requires_grad=False)
        self.gate.register_parameter("e_score_correction_bias", bias)
        self.routed_scaling_factor = float(cfg.routed_scaling_factor or 1.0)
        self.experts = FusedMoEFactory(
            num_experts=cfg.num_experts,
            top_k=cfg.num_experts_per_tok,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.moe_intermediate_size,
            renormalize=cfg.norm_topk_prob,
            quant_config=vllm_config.quant_config,
            use_grouped_topk=cfg.n_group is not None and cfg.topk_group is not None,
            num_expert_group=cfg.n_group,
            topk_group=cfg.topk_group,
            scoring_func=cfg.scoring_func,
            e_score_correction_bias=bias,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=True,
            router_logits_dtype=torch.float32,
            reduce_results=True,
            gate=self.gate,
            enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=parallel_config.eplb_config.num_redundant_experts,
            prefix=f"{prefix}.experts",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # router_logits is a placeholder: the runner applies self.gate to the hidden states
        return self.experts(hidden_states=x, router_logits=x)


__all__ = ["BeamHPUMLP", "BeamHPUSparseMoeBlock"]
