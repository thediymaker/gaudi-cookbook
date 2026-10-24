# SPDX-License-Identifier: Apache-2.0
"""Beam self-attention on vllm-gaudi: stock vLLM `Attention` (HPUAttentionImpl) with the Beam pieces
in torch around it.

  a [T, H] bf16 (input RMSNorm, done by the decoder layer)
  q, k, v = qkv_proj(a)          QKVParallelLinear (FP8 per-channel, compressed-tensors)
  g = output_gate_proj(a)        ColumnParallelLinear for the same q heads (checkpoint
                                 self_attn.gate_proj, renamed so it can never be taken for a
                                 gate_up_proj shard)
  q, k = rms(q) * wq, rms(k) * wk   per-head affine RMSNorm, fp32, one bf16 rounding
  sliding layers:  q, k = rope(pos, q, k)   NeoX, theta 1e4, through the ONE shared rotary module
                                 (fp32 cos/sin table): the runner refreshes cos/sin per step only for
                                 the first *RotaryEmbedding module it finds
  NoPE layers:     q = q * (1 + (pos + 1) / 2048) ** 0.1   SSMax root10, fp32, one rounding
  o = attn(q, k, v)              scale head_dim^-0.5; per_layer_sliding_window 512 on sliding layers,
                                 None on full ones (the model clears cache_config.sliding_window so it
                                 cannot leak into them)
  o = o * sigmoid(g + b_h)       per-head bias, sharded per rank
  y = o_proj(o)                  RowParallelLinear, all-reduce

Everything is 2-D [T, ...]: beam is not in vllm-gaudi's flatten_input list, so the model flattens at
entry and HPUAttentionImpl recovers [B, S] from its metadata.
"""

from __future__ import annotations

import logging
import os

import torch
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import ColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs

from . import numerics as N

logger = logging.getLogger("vllm.beam_hpu")

# FusedSDPA query tiling (install_fsdpa_query_tiling): a biased prompt-attention call whose workspace
# would exceed BEAM_FSDPA_WS_GIB GiB runs as query tiles of BEAM_FSDPA_Q_TILE rows. 0 GiB = vllm-gaudi's
# own rule (tiles only past a 2**31-byte bias plane).
FSDPA_WS_GIB = float(os.environ.get("BEAM_FSDPA_WS_GIB", "2"))
FSDPA_Q_TILE = int(os.environ.get("BEAM_FSDPA_Q_TILE", "256"))
# Device bytes FusedSDPA keeps per (query row, key column) of a biased prompt-attention call, measured
# on Gaudi 2 with fp32 softmax: about 60 B.
FSDPA_BYTES_PER_QK = 64


def install_fsdpa_query_tiling() -> None:
    """Bound the FusedSDPA workspace of long prompt chunks with cached context.

    A biased FusedSDPA prompt call (every prefill chunk with cached context: the full-attention
    layers' bias is [1, 1, q, context + q]) keeps a device workspace of about 60 B per (query row,
    key) for as long as the process lives, reused and grown by later calls. At 4096-token chunks that
    is 8 to 31 GiB per card from 32K to 128K of context, more than the cards have left after the
    weights and the KV pool: allocations fail, and the engine returned NaN logits or died. Query rows
    are independent (each tile sees all keys and its own bias rows), so vllm-gaudi's own q-tiling
    path computes the same attention; 256-row tiles keep the workspace near zero for about 3.5% more
    attention time.
    """
    if FSDPA_WS_GIB <= 0:
        return
    from vllm_gaudi.extension import ops as hpu_ops

    if getattr(hpu_ops._fsdpa_num_q_tiles, "_beam_hpu", False):
        return
    orig = hpu_ops._fsdpa_num_q_tiles

    def _fsdpa_num_q_tiles(attn_bias):
        n = orig(attn_bias)
        if attn_bias is None:
            return n
        q_len, kv_len = attn_bias.size(-2), attn_bias.size(-1)
        if q_len <= FSDPA_Q_TILE or FSDPA_BYTES_PER_QK * q_len * kv_len <= FSDPA_WS_GIB * 2**30:
            return n
        return max(n, -(-q_len // FSDPA_Q_TILE))

    _fsdpa_num_q_tiles._beam_hpu = True
    hpu_ops._fsdpa_num_q_tiles = _fsdpa_num_q_tiles
    logger.info(
        "beam_hpu: FusedSDPA prompt calls above %g GiB of workspace run as %d-row query tiles",
        FSDPA_WS_GIB,
        FSDPA_Q_TILE,
    )


def rope_theta(cfg) -> float | None:
    """The one RoPE theta of the sliding layers, or None for a model without RoPE layers."""
    rp = getattr(cfg, "rope_parameters", None) or {}
    if "rope_theta" in rp:
        return float(rp["rope_theta"])
    return float(cfg.rope_theta) if getattr(cfg, "rope_theta", None) else None


def make_rope(cfg, max_position: int):
    """The single shared NeoX rotary module (fp32 table), or None when every layer is NoPE."""
    if len(cfg.nope_layer_ids) == cfg.num_hidden_layers:
        return None
    rp = dict(getattr(cfg, "rope_parameters", None) or {})
    rope_type = rp.get("rope_type", "default")
    if rope_type != "default":
        raise ValueError(f"beam_hpu: rope_type {rope_type!r} is not supported (default only)")
    head_dim = cfg.head_dim or cfg.hidden_size // cfg.num_attention_heads
    return get_rope(
        head_dim,
        max_position=max_position,
        is_neox_style=True,
        rope_parameters={"rope_theta": rope_theta(cfg), "rope_type": "default"},
        dtype=torch.float32,
    )


class _Weight(nn.Module):
    """Holds a `.weight` parameter (norm gains; names match the checkpoint)."""

    def __init__(self, n: int, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n, dtype=dtype), requires_grad=False)


class BeamHPUAttention(nn.Module):
    def __init__(self, cfg, layer_id: int, rope, cache_config, quant_config, prefix: str):
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        H, nq, nkv = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads
        d = cfg.head_dim or H // nq
        if nq % tp:
            raise ValueError(f"beam_hpu: {nq} q heads do not divide over tp={tp}")
        if nkv >= tp:
            if nkv % tp:
                raise ValueError(f"beam_hpu: {nkv} kv heads do not divide over tp={tp}")
        elif tp % nkv:
            raise ValueError(f"beam_hpu: tp={tp} is not a multiple of {nkv} kv heads")
        if cfg.attention_bias:
            raise ValueError("beam_hpu: attention_bias=True is not supported")
        if (
            not cfg.attention_output_gate
            or cfg.attention_output_gate_floor != 0.0
            or cfg.attention_output_gate_temperature != 1.0
        ):
            raise ValueError(
                "beam_hpu: the output gate must be on with floor 0 and temperature 1 "
                f"(got {cfg.attention_output_gate}, {cfg.attention_output_gate_floor}, "
                f"{cfg.attention_output_gate_temperature})"
            )
        if cfg.use_paramfree_qk_norm:
            raise ValueError("beam_hpu: parameter-free QK norm is not supported")
        self.layer_id = layer_id
        self.nh, self.nkv, self.d = nq // tp, max(1, nkv // tp), d
        self.q_size, self.kv_size = self.nh * d, self.nkv * d
        self.eps = cfg.rms_norm_eps
        self.sliding = cfg.layer_types[layer_id] == "sliding_attention"
        self.use_rope = layer_id not in set(cfg.nope_layer_ids)
        self.use_ssmax = bool(cfg.ssmax) and not self.use_rope
        if self.use_ssmax and cfg.ssmax_scale_mode != "root10":
            raise ValueError(f"beam_hpu: ssmax_scale_mode {cfg.ssmax_scale_mode!r} is not supported (root10)")
        self.ssmax_offset = int(cfg.ssmax_offset)
        if self.use_rope and rope is None:
            raise ValueError(f"beam_hpu: layer {layer_id} uses RoPE but there is no rotary module")
        self._rope = (rope,) if self.use_rope else (None,)  # a tuple: registered once, on the model
        self.qkv_proj = QKVParallelLinear(
            H, d, nq, nkv, bias=False, quant_config=quant_config, prefix=f"{prefix}.qkv_proj"
        )
        self.output_gate_proj = ColumnParallelLinear(
            H, nq * d, bias=False, gather_output=False, quant_config=quant_config, prefix=f"{prefix}.output_gate_proj"
        )
        self.o_proj = RowParallelLinear(
            nq * d, H, bias=False, reduce_results=True, quant_config=quant_config, prefix=f"{prefix}.o_proj"
        )
        self.q_norm = _Weight(d)
        self.k_norm = _Weight(d)
        if cfg.attention_output_gate_bias:
            self.gate_bias = nn.Parameter(torch.zeros(self.nh), requires_grad=False)
            set_weight_attrs(self.gate_bias, {"weight_loader": sharded_weight_loader(0)})
        else:
            self.register_parameter("gate_bias", None)
        self.scale = float(d**-0.5)
        self.window = int(cfg.sliding_window) if self.sliding else None
        self.attn = Attention(
            self.nh,
            d,
            self.scale,
            num_kv_heads=self.nkv,
            cache_config=cache_config,
            quant_config=None,
            per_layer_sliding_window=self.window,
            prefix=f"{prefix}.attn",
        )
        if self.attn.impl.sliding_window != self.window:
            raise RuntimeError(
                f"beam_hpu: layer {layer_id} attention window {self.attn.impl.sliding_window}, expected {self.window}"
            )

    def qk(self, q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-head norm, then RoPE or SSMax. q [T, nh*d], k [T, nkv*d] -> bf16, same shapes."""
        T = q.shape[0]
        dt = q.dtype
        q = N.rms_affine(q.reshape(T, self.nh, self.d), self.q_norm.weight, self.eps, dt)
        k = N.rms_affine(k.reshape(T, self.nkv, self.d), self.k_norm.weight, self.eps, dt)
        if self.use_rope:
            qf, kf = self._rope[0](pos, N.up(q).reshape(T, self.q_size), N.up(k).reshape(T, self.kv_size))
            return N.store(qf, dt).reshape(T, self.q_size), N.store(kf, dt).reshape(T, self.kv_size)
        if self.use_ssmax:
            q = N.apply_ssmax(q, pos, self.ssmax_offset, dt)
        return q.reshape(T, self.q_size), k.reshape(T, self.kv_size)

    def forward(self, a: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """a [T, H] bf16, pos [T] -> o_proj output [T, H] bf16 (all-reduced)."""
        qkv = self.qkv_proj(a)[0]
        g = self.output_gate_proj(a)[0]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.qk(q, k, pos)
        o = self.attn(q, k, v.contiguous())
        o = N.output_gate(o, g, self.gate_bias, self.nh, self.d, a.dtype)
        return self.o_proj(o)[0]


__all__ = ["BeamHPUAttention", "install_fsdpa_query_tiling", "make_rope", "rope_theta"]
