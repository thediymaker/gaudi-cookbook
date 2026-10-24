# SPDX-License-Identifier: Apache-2.0
"""Muse Spark self-attention: stock vLLM `Attention` (vllm-gaudi's HPUAttentionImpl) with the Muse pieces around it.

    a [T, H] bf16                     input of the layer (centred input norm applied by the decoder layer)
    q, k, v = qkv_proj(a)             this rank's 16 q and 2 kv heads (TP 8)
    g = gate_proj(a)                  output gate for the same q heads
    q, k = rms(q), rms(k)             weightless per-head QK norm, fp32
    q, k = rope(pos, q, k)            sliding layers only (46 of 62; the 16 global layers have no RoPE),
                                      fp32 table and fp32 q/k, then one bf16 rounding
    o = attn(q, k, v)                 scale head_dim^-0.5 * qk_scale_factor; window 2048 on sliding layers
    o = rms(o) * sigmoid(g)           weightless per-head output norm, fp32
    y = o_proj(o)                     row-parallel, bf16 all-reduce

Tensors stay 2-D [T, ...]: the model flattens at entry and HPUAttentionImpl recovers [B, S] from its metadata.
The HPU rotary kernel accepts fp32 query/cos/sin and matches an fp64 rotation to about 4e-8, so q/k stay fp32
out of the QK norm and are rounded once, as the reference does.
"""

from __future__ import annotations

import torch
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import ColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.rotary_embedding import get_rope

from .numerics import rms


def make_rope(tcfg, max_position: int):
    """The single rotary module shared by every RoPE layer (one theta in the released config)."""
    thetas = {float(t) for t in tcfg.layer_rope_theta[: tcfg.num_hidden_layers] if float(t) != 0.0}
    if len(thetas) != 1:
        raise ValueError(f"muse_spark_hpu: one RoPE theta expected, got {sorted(thetas)}")
    return get_rope(
        tcfg.head_dim,
        max_position=max_position,
        is_neox_style=True,
        rope_parameters={"rope_theta": thetas.pop(), "rope_type": "default"},
        dtype=torch.float32,
    )


class MuseSparkHPUAttention(nn.Module):
    def __init__(self, tcfg, layer_id: int, rope, cache_config, quant_config, prefix: str):
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        H, nq, nkv, d = tcfg.hidden_size, tcfg.num_attention_heads, tcfg.num_key_value_heads, tcfg.head_dim
        if nq % tp or nkv % tp:
            raise ValueError(f"muse_spark_hpu: heads {nq}/{nkv} do not divide over tp={tp}")
        self.nh, self.nkv, self.d = nq // tp, nkv // tp, d
        self.q_size, self.kv_size = self.nh * d, self.nkv * d
        self.eps = tcfg.rms_norm_eps
        sliding = tcfg.layer_types[layer_id] == "sliding_attention"
        self.use_rope = float(tcfg.layer_rope_theta[layer_id]) != 0.0
        # a tuple, so the shared module is registered once, on the text model
        self._rope = (rope if self.use_rope else None,)
        self.qkv_proj = QKVParallelLinear(
            H, d, nq, nkv, bias=False, quant_config=quant_config, prefix=f"{prefix}.qkv_proj"
        )
        self.gate_proj = ColumnParallelLinear(
            H, nq * d, bias=False, gather_output=False, quant_config=quant_config, prefix=f"{prefix}.gate_proj"
        )
        self.o_proj = RowParallelLinear(
            nq * d, H, bias=False, reduce_results=True, quant_config=quant_config, prefix=f"{prefix}.o_proj"
        )
        self.attn = Attention(
            self.nh,
            d,
            float(d**-0.5 * tcfg.qk_scale_factor),
            num_kv_heads=self.nkv,
            cache_config=cache_config,
            quant_config=None,
            per_layer_sliding_window=tcfg.sliding_window if sliding else None,
            prefix=f"{prefix}.attn",
        )

    def forward(self, a: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """a [T, H] bf16, pos [T] -> o_proj output [T, H] bf16 (all-reduced)."""
        T = a.shape[0]
        qkv = self.qkv_proj(a)[0]
        g = self.gate_proj(a)[0]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = rms(q.reshape(T, self.nh, self.d), self.eps).reshape(T, self.q_size)
        k = rms(k.reshape(T, self.nkv, self.d), self.eps).reshape(T, self.kv_size)
        if self.use_rope:
            q, k = self._rope[0](pos, q, k)
        o = self.attn(q.to(a.dtype), k.to(a.dtype), v.contiguous())
        o = rms(o.reshape(T, self.nh, self.d), self.eps) * torch.sigmoid(g.reshape(T, self.nh, self.d).float())
        return self.o_proj(o.to(a.dtype).reshape(T, self.q_size))[0]
