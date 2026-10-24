# SPDX-License-Identifier: Apache-2.0
"""Inkling-Small (thinkingmachines) on Gaudi, text only, for vllm-gaudi releases/v0.30.0.

vLLM's own Inkling model is CUDA/ROCm only (Triton short-conv and QKVR kernels, FA4 relative attention) and
its layers are not Attention modules, so the HPU runner would give them no KV cache. This is a separate
implementation on the HPU attention ops. Vision, audio and the MTP draft head are not built; their
checkpoint tensors are skipped.

Model (276B total, 12B active; 42 layers: 35 sliding-window-512 + 7 global):
  * No RoPE. Attention adds a learned relative logit per head:
      logit(i, j, h) = q[i, h] . k[j, h] / head_dim + rel[i, h, i - j]   for 0 <= i - j < rel_extent
    with rel_extent 512 on local layers and 1024 on global layers. q and k are RMS-normed per head, hence
    the 1 / head_dim scale. Global layers also scale q and rel by the log-position tau (ops.py).
  * Short convs (conv_state.py) on raw K and V before the q/k norm, and on the TP-summed attention and
    MLP deltas before the residual add: h = h + sconv(sum(delta)).
  * MoE: 256 routed experts top-6 plus 2 shared "sink" experts. Selection uses sigmoid(logit) + bias on the
    routed experts; the weights are sigmoid of the raw logits of the 6 chosen + 2 shared experts,
    renormalized over those 8, times route_scale and a learned scale. Layers below dense_mlp_idx are dense.
  * Logits are divided by logits_mup_width_multiplier (16).

HPU integration:
  * Standard Attention layers own the paged KV cache. Local layers pass per_layer_sliding_window and the
    config exposes layer_types, so the runner builds the window_block_* decode metadata.
  * The rel bias goes in through InklingAttentionImpl, a copy of HPUAttentionImpl.forward with three
    insertions: prefill gets an additive FusedSDPA mask, decode a per-block position bias in the ALiBi slot.
    Each layer stashes (rel, positions) in _REL_CTX right before calling its Attention.
  * Chunked prefill: keys are [cached context blocks | current chunk] (the layout of ops._get_context) and
    the query rows are processed in sub-blocks so the materialized bias stays under a memory budget.

Switches (environment, all off by default; the recipe turns on all four):
  INKLING_MOE_SINGLE_REDUCE=1     routed and sink experts summed TP-partial and all-reduced once per layer
                                  (off: FusedMoE reduces its own output and the sinks are reduced again).
  INKLING_LOCAL_WINDOW_PREFILL=1  chunked prefill on local layers fetches and biases only the context blocks
                                  their window can reach (off: the whole context).
  INKLING_STEP_HOIST=1            per-step metadata computed once per forward instead of per layer (ops.py).
  INKLING_DECODE_WINDOW_PLUS1=1   runner patch: local layers get window // block_size + 1 decode blocks, as
                                  gpt-oss does (off: one block short, decode sees 385 to 512 of 512 keys).
  INKLING_PREFILL_BIAS_BUDGET     bytes for one prefill bias sub-block, default 1 GiB.
"""

import dataclasses
import os
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm_gaudi.attention.backends.hpu_attn import HPUAttentionImpl, _set_fetch_by_id
from vllm_gaudi.attention.ops.hpu_paged_attn import HPUPagedAttention
from vllm_gaudi.extension import ops as ghpu_ops
from vllm_gaudi.v1.attention.backends.hpu_attn import HPUAttentionBackendV1

from . import conv_state as cs
from . import ops
from .config import InklingHPUConfig

LOCAL_WINDOW_PREFILL = os.environ.get("INKLING_LOCAL_WINDOW_PREFILL", "0") == "1"
MOE_SINGLE_REDUCE = os.environ.get("INKLING_MOE_SINGLE_REDUCE", "0") == "1"
PREFILL_BIAS_BUDGET = int(os.environ.get("INKLING_PREFILL_BIAS_BUDGET", str(1024**3)))

# Per-layer (rel logits, positions), keyed by the Attention prefix: set by InklingHPUAttention.forward,
# consumed by InklingAttentionImpl.forward.
_REL_CTX: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
# Conv-state bytes per attention block per layer; the attention page is padded by this much.
_CONV_PAGE_BYTES = 0


class InklingAttentionImpl(HPUAttentionImpl):
    """HPUAttentionImpl.forward with the rel bias added (the three INKLING blocks). The rest matches the
    parent so cache writes, fetches and buckets behave the same. Not copied from the parent: the KV-sharing
    branches (unused here) and the prefill window_block_list swap, because the bias assumes context column ==
    absolute position."""

    def __init__(self, *args, **kwargs):
        self.inkling_prefix = kwargs.pop("inkling_prefix")
        self.inkling_rel_extent = kwargs.pop("inkling_rel_extent")
        self.inkling_is_local = kwargs.pop("inkling_is_local")
        self.inkling_window = kwargs.pop("inkling_window", None)
        super().__init__(*args, **kwargs)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None):
        rel_ctx = _REL_CTX.pop(self.inkling_prefix, None)

        output_shape = query.shape
        if query.dim() == 2:
            if attn_metadata.seq_lens_tensor is not None:
                batch_size = attn_metadata.seq_lens_tensor.shape[0] if not self.use_merged_prefill else 1
            else:
                assert attn_metadata.block_mapping is not None
                batch_size = attn_metadata.block_mapping.shape[1]
            num_tokens, hidden_size = query.shape
            seq_len = num_tokens // batch_size
            query = query.view(batch_size, seq_len, -1)
        else:
            batch_size, seq_len, hidden_size = query.shape

        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        slot_mapping = attn_metadata.slot_mapping.flatten() if attn_metadata.slot_mapping is not None else None
        key_cache = value_cache = k_scales = v_scales = None
        if kv_cache is not None and isinstance(kv_cache, tuple):
            key_cache, value_cache, k_scales, v_scales = HPUPagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size
            )
            if key.dtype == torch.float32 and key.dtype != key_cache.dtype:
                key = key.to(key_cache.dtype)
            if value.dtype == torch.float32 and value.dtype != value_cache.dtype:
                value = value.to(value_cache.dtype)
            if query.dtype != key.dtype:
                query = query.to(key.dtype)
            if self.kv_sharing_target_layer_name is None:
                key_cache = self.k_cache(
                    key,
                    key_cache,
                    slot_mapping,
                    scales=k_scales,
                    block_size=attn_metadata.block_size,
                    is_prompt=attn_metadata.is_prompt,
                )
                value_cache = self.v_cache(
                    value,
                    value_cache,
                    slot_mapping,
                    scales=v_scales,
                    block_size=attn_metadata.block_size,
                    is_prompt=attn_metadata.is_prompt,
                )

        if attn_metadata.is_prompt or seq_len > 1:
            if self.inkling_is_local and getattr(attn_metadata, "window_block_list", None) is not None:
                raise RuntimeError("inkling_hpu: prefill window_block_list is not supported")
            query_shape = (batch_size, seq_len, self.num_heads, self.head_size)
            kv_shape = (batch_size, -1, self.num_kv_heads, self.head_size)
            attn_bias = attn_metadata.attn_bias
            block_list = attn_metadata.block_list if attn_metadata and attn_metadata.block_list is not None else None
            _set_fetch_by_id(self.k_cache, True)
            _set_fetch_by_id(self.v_cache, True)
            common_args = self.common_attention_args(
                block_list, key_cache, value_cache, attn_metadata.block_size, k_scales, v_scales
            )
            if self.sliding_window:
                if getattr(attn_metadata, "window_attn_bias", None) is not None:
                    attn_bias = attn_metadata.window_attn_bias
                else:
                    attn_bias = None
                    common_args["window_size"] = (self.sliding_window, 0)

            if rel_ctx is not None:
                # INKLING (prefill): the rel bias carries its own causal and window mask, so it replaces the
                # native window argument.
                rel_flat, pos_flat = rel_ctx
                rel_b = rel_flat.view(batch_size, seq_len, self.num_heads, -1)
                pos_b = pos_flat.view(batch_size, seq_len)
                win = self.inkling_window if self.inkling_is_local else None
                common_args.pop("window_size", None)
                if block_list is None:
                    out = self._prefill_first_chunk(
                        query,
                        key,
                        value,
                        query_shape,
                        kv_shape,
                        rel_b,
                        pos_b,
                        win,
                        attn_bias,
                        attn_metadata,
                        common_args,
                    )
                else:
                    out = self._prefill_with_context(
                        query,
                        key,
                        value,
                        query_shape,
                        kv_shape,
                        rel_b,
                        pos_b,
                        win,
                        block_list,
                        key_cache,
                        value_cache,
                        k_scales,
                        v_scales,
                        attn_metadata,
                        common_args,
                        batch_size,
                        seq_len,
                    )
            else:  # the profiling run calls the impl without a stashed rel
                out = ghpu_ops.prompt_attention(
                    impl=self.prefill_impl,
                    query=query.view(query_shape),
                    key=key.view(kv_shape),
                    value=value.view(kv_shape),
                    is_causal=attn_bias is None,
                    attn_bias=attn_bias,
                    position_bias=None,
                    valid_seq_lengths=attn_metadata.seq_lens_tensor,
                    **common_args,
                )
            output = out.reshape(batch_size, seq_len, hidden_size)
        else:
            # Decode. Block lists hold real physical ids (non-contiguous PA), so fetch by id; the parent's
            # slice fetch (cache[:n]) assumes the contiguous identity layout.
            _set_fetch_by_id(self.k_cache, True)
            _set_fetch_by_id(self.v_cache, True)
            if self.sliding_window and attn_metadata.window_block_list is not None:
                block_list = attn_metadata.window_block_list
                block_groups = attn_metadata.window_block_groups
                block_mapping = attn_metadata.window_block_mapping
                attn_bias = attn_metadata.window_attn_bias
            else:
                block_list = attn_metadata.block_list
                block_groups = attn_metadata.block_groups
                block_mapping = attn_metadata.block_mapping
                attn_bias = attn_metadata.attn_bias

            self.position_bias = None
            if rel_ctx is not None and block_list is not None:
                # INKLING (decode): bias[blk, h, j] added to key j of block blk, in the ALiBi slot.
                rel_flat, pos_flat = rel_ctx
                rel_d = rel_flat.view(-1, self.num_heads, self.inkling_rel_extent)
                win_d = self.inkling_window if self.inkling_is_local else None
                pos_d = pos_flat.view(-1)
                index = ops.step_cached(
                    # positions are the same in every layer of a forward
                    ("drel", id(block_groups), rel_d.shape[0], self.inkling_rel_extent, win_d),
                    lambda: ops.decode_rel_index(
                        pos_d, block_groups, rel_d.shape[0], attn_metadata.block_size, self.inkling_rel_extent, win_d
                    ),
                )
                self.position_bias = ops.decode_rel_apply(rel_d, index)

            if key_cache is None:
                return torch.zeros(*output_shape, dtype=query.dtype, device=query.device)
            output = HPUPagedAttention.forward_decode(
                query=query,
                block_mapping=block_mapping,
                block_bias=attn_bias,
                block_groups=block_groups,
                position_bias=self.position_bias,
                **self.common_attention_args(
                    block_list, key_cache, value_cache, attn_metadata.block_size, k_scales, v_scales
                ),
            )
        return output.view(*output_shape)

    def _prefill_first_chunk(
        self, query, key, value, query_shape, kv_shape, rel_b, pos_b, win, attn_bias, attn_metadata, common_args
    ):
        """First (or only) chunk: the keys are the chunk itself."""
        bias = ops.prompt_rel_bias(
            rel_b, pos_b, pos_b, 0, None, self.inkling_rel_extent, win, query.dtype, attn_metadata.seq_lens_tensor
        )
        if attn_bias is not None:
            bias = bias + attn_bias.to(bias.dtype)
        return ghpu_ops.prompt_attention(
            impl=self.prefill_impl,
            query=query.view(query_shape),
            key=key.view(kv_shape),
            value=value.view(kv_shape),
            is_causal=False,
            attn_bias=bias,
            position_bias=None,
            valid_seq_lengths=None,  # FusedSDPA rejects it with an explicit mask; the bias masks pad keys
            **common_args,
        )

    def _prefill_with_context(
        self,
        query,
        key,
        value,
        query_shape,
        kv_shape,
        rel_b,
        pos_b,
        win,
        block_list,
        key_cache,
        value_cache,
        k_scales,
        v_scales,
        attn_metadata,
        common_args,
        batch_size,
        seq_len,
    ):
        """Later chunk or prefix-cache hit: prepend the cached context once, the way ops._get_context does,
        then attend in query sub-blocks so the materialized bias stays under PREFILL_BIAS_BUDGET."""
        ctx_pad = (block_list.numel() // batch_size) * attn_metadata.block_size
        fetch_blocks = block_list
        win_ctx = None
        if LOCAL_WINDOW_PREFILL and win is not None:
            fetch_blocks, ctx_kpos, ctx_valid = ops.step_cached(
                ("win", id(block_list), batch_size, win),
                lambda: ops.window_context(
                    block_list, attn_metadata.context_lens_tensor, batch_size, attn_metadata.block_size, win
                ),
            )
            ctx_pad = ctx_kpos.shape[1]
            win_ctx = (ctx_kpos, ctx_valid)
        fetch_args = {
            "key": key.view(kv_shape),
            "value": value.view(kv_shape),
            "keys_fetch_func": self.k_cache.fetch_from_cache,
            "values_fetch_func": self.v_cache.fetch_from_cache,
            "key_cache": key_cache,
            "value_cache": value_cache,
            "k_scales": k_scales,
            "v_scales": v_scales,
            "block_list": fetch_blocks,
            "block_size": attn_metadata.block_size,
        }
        ghpu_ops._get_context(fetch_args)
        full_k, full_v = fetch_args["key"], fetch_args["value"]
        common_args["block_list"] = None  # the context is already merged into the keys
        qsub = ops.query_sub_block(seq_len, batch_size, self.num_heads, ctx_pad + seq_len, PREFILL_BIAS_BUDGET)
        import habana_frameworks.torch.core as htcore

        outs = []
        for qs in range(0, seq_len, qsub):
            qe = min(qs + qsub, seq_len)
            if win_ctx is not None:
                bias = ops.prompt_rel_bias_kpos(
                    rel_b[:, qs:qe],
                    pos_b[:, qs:qe],
                    win_ctx[0],
                    win_ctx[1],
                    pos_b,
                    self.inkling_rel_extent,
                    win,
                    query.dtype,
                    attn_metadata.seq_lens_tensor,
                )
            else:
                bias = ops.prompt_rel_bias(
                    rel_b[:, qs:qe],
                    pos_b[:, qs:qe],
                    pos_b,
                    ctx_pad,
                    attn_metadata.context_lens_tensor,
                    self.inkling_rel_extent,
                    win,
                    query.dtype,
                    attn_metadata.seq_lens_tensor,
                )
            outs.append(
                ghpu_ops.prompt_attention(
                    impl=self.prefill_impl,
                    query=query.view(query_shape)[:, qs:qe],
                    key=full_k,
                    value=full_v,
                    is_causal=False,
                    attn_bias=bias,
                    position_bias=None,
                    valid_seq_lengths=None,
                    **common_args,
                )
            )
            # One graph per sub-block: in lazy mode the whole chunk (42 layers x N multi-GB bias
            # transients) otherwise accumulates into one graph and runs out of device memory near 100K tokens.
            htcore.mark_step()
        return torch.cat(outs, dim=1)


class InklingAttentionBackend(HPUAttentionBackendV1):
    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return InklingAttentionImpl


class InklingShortConvWeight(nn.Module):
    """One depthwise conv weight, checkpoint layout [C, 1, W] or [C, W]. shard=True narrows the rows to this
    rank (K/V streams); shard=False keeps the full width (the delta streams run on the all-reduced delta)."""

    def __init__(self, dim: int, kernel_size: int, shard: bool) -> None:
        super().__init__()
        self.shard = shard
        self.tp_rank = get_tensor_model_parallel_rank()
        self.weight = nn.Parameter(torch.empty(dim, kernel_size), requires_grad=False)
        self.weight.weight_loader = self._load

    def _load(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        w = loaded.squeeze(1) if loaded.dim() == 3 else loaded
        if self.shard and w.shape[0] != param.shape[0]:
            w = w.narrow(0, self.tp_rank * param.shape[0], param.shape[0])
        assert w.shape == param.shape, (w.shape, param.shape)
        param.data.copy_(w)


class InklingHPUAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        conv_state: cs.InklingConvState,
        is_local: bool,
        prefix: str,
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp_size == 0
        assert self.total_num_kv_heads % tp_size == 0, "TP > num_kv_heads is not wired for the K/V convs"
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = config.head_dim
        self.d_rel = config.d_rel
        self.is_local = is_local
        self.rel_extent = config.sliding_window_size if is_local else config.rel_extent
        self.sliding_window = config.sliding_window_size if is_local else None
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.r_size = self.num_heads * self.d_rel

        self.qkvr = MergedColumnParallelLinear(
            config.hidden_size,
            [
                self.head_dim * self.total_num_heads,
                self.head_dim * self.total_num_kv_heads,
                self.head_dim * self.total_num_kv_heads,
                self.d_rel * self.total_num_heads,
            ],
            bias=bool(getattr(config, "q_bias", False)),
            quant_config=quant_config,
            prefix=f"{prefix}.qkvr",
        )
        self.wo_ud = RowParallelLinear(
            self.head_dim * self.total_num_heads,
            config.hidden_size,
            bias=bool(getattr(config, "o_bias", False)),
            quant_config=quant_config,
            reduce_results=False,  # the layer all-reduces, then applies the delta conv
            prefix=f"{prefix}.wo_ud",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # [d_rel, rel_extent], replicated (small)
        self.rel_logits_proj = nn.Parameter(torch.empty(self.d_rel, self.rel_extent), requires_grad=False)
        self.rel_logits_proj.weight_loader = default_weight_loader

        self.conv_state = conv_state
        self.k_sconv = InklingShortConvWeight(self.kv_size, config.sconv_kernel_size, shard=True)
        self.v_sconv = InklingShortConvWeight(self.kv_size, config.sconv_kernel_size, shard=True)
        self.log_scaling_n_floor = getattr(config, "log_scaling_n_floor", None)
        self.log_scaling_alpha = getattr(config, "log_scaling_alpha", 0.1)

        self._attn_prefix = f"{prefix}.attn"
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            1.0 / self.head_dim,  # q/k are unit-RMS per head
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=self.sliding_window,
            attn_backend=InklingAttentionBackend,
            prefix=self._attn_prefix,
            # forwarded to InklingAttentionImpl
            inkling_prefix=self._attn_prefix,
            inkling_rel_extent=self.rel_extent,
            inkling_is_local=is_local,
            inkling_window=self.sliding_window,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, bsz: int, seq: int) -> torch.Tensor:
        """hidden_states [bsz * seq, H] and positions [bsz * seq], flattened once by the model."""
        qkvr, _ = self.qkvr(hidden_states)
        q, k, v, r = qkvr.split([self.q_size, self.kv_size, self.kv_size, self.r_size], dim=-1)

        meta = get_forward_context().attn_metadata
        is_prompt = bool(getattr(meta, "is_prompt", True)) if meta is not None else True
        num_tokens = hidden_states.shape[0]
        if is_prompt or seq > 1:
            k = self.conv_state.conv(cs.K_STREAM, k.reshape(bsz, seq, -1), self.k_sconv.weight, True, meta)
            v = self.conv_state.conv(cs.V_STREAM, v.reshape(bsz, seq, -1), self.v_sconv.weight, True, meta)
            k, v = k.reshape(num_tokens, -1), v.reshape(num_tokens, -1)
        else:
            k = self.conv_state.conv(cs.K_STREAM, k.contiguous(), self.k_sconv.weight, False, meta)
            v = self.conv_state.conv(cs.V_STREAM, v.contiguous(), self.v_sconv.weight, False, meta)

        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(num_tokens, -1)
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(num_tokens, -1)
        rel = torch.einsum(
            "thd,de->the", r.reshape(num_tokens, self.num_heads, self.d_rel), self.rel_logits_proj
        )  # [T, H, extent]
        if not self.is_local and self.log_scaling_n_floor is not None:
            tau = ops.log_scaling_tau(positions.view(-1), self.log_scaling_n_floor, self.log_scaling_alpha)
            q = (q.float() * tau.unsqueeze(-1)).to(q.dtype)
            rel = (rel.float() * tau.view(-1, 1, 1)).to(rel.dtype)

        _REL_CTX[self._attn_prefix] = (rel, positions)
        attn_out = self.attn(q, k, v)
        _REL_CTX.pop(self._attn_prefix, None)  # the profiling run skips the impl
        out, _ = self.wo_ud(attn_out)
        return out  # TP-partial


def _deinterleave_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[2F, D] interleaved [g0, u0, g1, u1, ...] -> (gate [F, D], up [F, D])."""
    return w[0::2].contiguous(), w[1::2].contiguous()


class InklingHPUDenseMLP(nn.Module):
    """Dense SwiGLU MLP. The checkpoint stores gate and up interleaved in one tensor (w13_dn)."""

    def __init__(self, hidden_size: int, intermediate_size: int, quant_config, prefix: str) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.down_proj",
        )
        self.global_scale = nn.Parameter(torch.empty(1), requires_grad=False)
        self.global_scale.weight_loader = default_weight_loader

    def load_fused_interleaved(self, loaded: torch.Tensor) -> None:
        gate, up = _deinterleave_rows(loaded)
        self.gate_up_proj.weight_loader(self.gate_up_proj.weight, gate, 0)
        self.gate_up_proj.weight_loader(self.gate_up_proj.weight, up, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x)[0].chunk(2, dim=-1)
        x, _ = self.down_proj(F.silu(gate) * up)
        return x * self.global_scale.to(x.dtype)


class InklingSinkExperts(nn.Module):
    """The 2 shared "sink" experts: dense SwiGLU MLPs weighted per token, sharded on the intermediate dim
    (TP-partial output)."""

    def __init__(self, n_experts: int, d_model: int, d_mlp: int) -> None:
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        assert d_mlp % tp == 0
        self.tp_rank = get_tensor_model_parallel_rank()
        self.n_experts = n_experts
        self.f_local = d_mlp // tp
        self.w13_weight = nn.Parameter(torch.empty(n_experts, 2 * self.f_local, d_model), requires_grad=False)
        self.w2_weight = nn.Parameter(torch.empty(n_experts, d_model, self.f_local), requires_grad=False)

    def load_weight(self, key: str, weight: torch.Tensor) -> None:
        f = self.f_local
        for e in range(self.n_experts):
            if key == "shared_w13_weight":  # [S, 2F, D], interleaved rows
                gate, up = _deinterleave_rows(weight[e].narrow(0, self.tp_rank * 2 * f, 2 * f))
                self.w13_weight.data[e, :f].copy_(gate)
                self.w13_weight.data[e, f:].copy_(up)
            else:
                assert key == "shared_w2_weight"  # [S, D, F]
                self.w2_weight.data[e].copy_(weight[e].narrow(1, self.tp_rank * f, f))

    def forward(self, x: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            gate, up = (x @ self.w13_weight[e].t()).chunk(2, dim=-1)
            out = out + ((F.silu(gate) * up) @ self.w2_weight[e].t()) * gammas[:, e : e + 1].to(x.dtype)
        return out


class InklingHPUMoE(nn.Module):
    def __init__(self, config: PretrainedConfig, quant_config, prefix: str) -> None:
        super().__init__()
        self.n_routed = config.n_routed_experts
        self.n_shared = config.n_shared_experts
        self.topk = config.num_experts_per_tok
        self.route_scale = float(config.route_scale)
        # fp32 gate; rows = routed + shared (+ zero padding rows in the checkpoint)
        self.gate = ReplicatedLinear(
            config.hidden_size,
            self.n_routed + self.n_shared,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.gate_bias = nn.Parameter(torch.empty(self.n_routed, dtype=torch.float32), requires_grad=False)
        self.gate_bias.weight_loader = default_weight_loader
        self.gate_global_scale = nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)
        self.gate_global_scale.weight_loader = default_weight_loader
        self._routing_weights: torch.Tensor | None = None
        self._routing_ids: torch.Tensor | None = None

        self.experts = FusedMoEFactory(
            num_experts=self.n_routed,
            top_k=self.topk,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            renormalize=False,
            custom_routing_function=self._routing,
            router_logits_dtype=torch.float32,
            quant_config=quant_config,
            activation="silu",
            prefix=f"{prefix}.experts",
            reduce_results=not MOE_SINGLE_REDUCE,
        )
        if MOE_SINGLE_REDUCE:
            # reduce_results=False is a request: FusedMoE still reduces with all2all kernels, sequence
            # parallel or zero experts, and reducing again in the layer would scale the delta by the TP size.
            granted = getattr(getattr(self.experts, "moe_config", None), "skip_final_all_reduce", None)
            if granted is not True:
                raise ValueError(f"INKLING_MOE_SINGLE_REDUCE=1 but FusedMoE skip_final_all_reduce={granted}")
        self.returns_reduced = not MOE_SINGLE_REDUCE
        self.sink_experts = InklingSinkExperts(self.n_shared, config.hidden_size, config.intermediate_size)

    def _routing(self, hidden_states, gating_output, topk, renormalize):
        assert topk == self.topk and self._routing_weights is not None
        w, ids = self._routing_weights, self._routing_ids
        self._routing_weights = self._routing_ids = None
        return w, ids

    def _select(self, logits: torch.Tensor):
        routed = logits[:, : self.n_routed]
        shared = logits[:, self.n_routed : self.n_routed + self.n_shared]
        _, ids = torch.topk(torch.sigmoid(routed) + self.gate_bias.unsqueeze(0), self.topk, dim=-1)
        s = torch.sigmoid(torch.cat([routed.gather(1, ids), shared], dim=1))  # [T, K + S] raw logits
        w = s / s.sum(dim=-1, keepdim=True) * (self.route_scale * self.gate_global_scale)
        return w[:, : self.topk].contiguous(), ids.to(torch.int32), w[:, self.topk :].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.mm(x.float(), self.gate.weight.t().float())
        self._routing_weights, self._routing_ids, gammas = self._select(logits)
        # Routing is precomputed in _routing; the MoE runner wants one logit column per routed expert.
        routed = self.experts(hidden_states=x, router_logits=logits[:, : self.n_routed])
        sink = self.sink_experts(x, gammas)
        if MOE_SINGLE_REDUCE:
            return routed + sink  # TP-partial, reduced once by the layer
        if get_tensor_model_parallel_world_size() > 1:
            sink = tensor_model_parallel_all_reduce(sink)
        return routed + sink

    def load_expert_weight(self, key: str, weight: torch.Tensor) -> None:
        experts = self.experts.routed_experts
        tp_rank = get_tensor_model_parallel_rank()
        if key == "experts.w13_weight":
            param = experts.w13_weight
            half = param.shape[1] // 2
            for e in range(self.n_routed):
                gate, up = _deinterleave_rows(weight[e].narrow(0, tp_rank * 2 * half, 2 * half))
                param.data[e, :half].copy_(gate)
                param.data[e, half:].copy_(up)
        elif key == "experts.w2_weight":
            param = experts.w2_weight
            shard = param.shape[2]
            for e in range(self.n_routed):
                param.data[e].copy_(weight[e].narrow(1, tp_rank * shard, shard))
        elif key.startswith("shared_experts."):
            self.sink_experts.load_weight(key.split(".", 1)[1], weight)
        else:
            raise ValueError(f"unexpected MoE tensor {key}")


class InklingHPUDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        layer_id = extract_layer_index(prefix)
        is_local = layer_id in set(config.local_layer_ids)
        self.tp = get_tensor_model_parallel_world_size()
        kv_dim = (config.num_key_value_heads // self.tp) * config.head_dim
        self.conv_state = cs.InklingConvState(
            vllm_config, kv_dim, config.hidden_size, config.sconv_kernel_size, prefix=f"sconv_layers.{1000 + layer_id}"
        )
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = InklingHPUAttention(
            config, vllm_config.cache_config, vllm_config.quant_config, self.conv_state, is_local, prefix=prefix
        )
        self.conv_state.register()
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if layer_id < config.dense_mlp_idx:
            self.mlp: nn.Module = InklingHPUDenseMLP(
                config.hidden_size, config.dense_intermediate_size, vllm_config.quant_config, prefix=f"{prefix}.mlp"
            )
            self.is_moe = False
        else:
            self.mlp = InklingHPUMoE(config, vllm_config.quant_config, prefix=f"{prefix}.mlp")
            self.is_moe = True
        # Full-width delta convs on every rank: a depthwise conv commutes with the channel shard, so this
        # equals the reference reduce-scatter, sharded conv, all-gather.
        self.attn_sconv = InklingShortConvWeight(config.hidden_size, config.sconv_kernel_size, shard=False)
        self.mlp_sconv = InklingShortConvWeight(config.hidden_size, config.sconv_kernel_size, shard=False)

    def _sconv_delta(self, stream, delta, weight, bsz, seq, already_reduced=False):
        if self.tp > 1 and not already_reduced:
            delta = tensor_model_parallel_all_reduce(delta)
        meta = get_forward_context().attn_metadata
        is_prompt = bool(getattr(meta, "is_prompt", True)) if meta is not None else True
        if is_prompt or seq > 1:
            return self.conv_state.conv(stream, delta.reshape(bsz, seq, -1), weight, True, meta).reshape(
                delta.shape[0], -1
            )
        return self.conv_state.conv(stream, delta, weight, False, meta)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, bsz: int, seq: int) -> torch.Tensor:
        attn_delta = self.attn(positions, self.attn_norm(hidden_states), bsz, seq)
        hidden_states = hidden_states + self._sconv_delta(cs.ATTN_STREAM, attn_delta, self.attn_sconv.weight, bsz, seq)
        mlp_delta = self.mlp(self.mlp_norm(hidden_states))
        return hidden_states + self._sconv_delta(
            cs.MLP_STREAM,
            mlp_delta,
            self.mlp_sconv.weight,
            bsz,
            seq,
            already_reduced=getattr(self.mlp, "returns_reduced", False),
        )


class InklingHPUModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        # Full-vocab embedding on every rank (about 1.6 GiB bf16) followed by the embedding RMSNorm.
        self.embed_tokens = nn.Embedding(config.padded_vocab_size, config.hidden_size)
        self.embed_tokens.weight.requires_grad = False
        self.embed_tokens.weight.weight_loader = default_weight_loader
        self.embed_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_embed_norm else None
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: InklingHPUDecoderLayer(vllm_config, prefix),
            prefix=maybe_prefix(prefix, "layers"),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        return self.embed_norm(hidden) if self.embed_norm is not None else hidden

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if get_pp_group().is_first_rank:
            # inputs_embeds already went through embed_input_ids (and the embedding norm)
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        ops.STEP_CACHE.clear()
        # The HPU runner passes padded batches as [B, L, H]; every layer works on [B * L, H] with (bsz, seq)
        # passed explicitly.
        orig_shape = None
        if hidden_states.dim() == 3:
            orig_shape = hidden_states.shape
            bsz, seq = int(orig_shape[0]), int(orig_shape[1])
            hidden_states = hidden_states.reshape(bsz * seq, -1)
        else:
            meta = get_forward_context().attn_metadata
            bsz = meta.seq_lens_tensor.shape[0] if meta is not None and meta.seq_lens_tensor is not None else 1
            seq = max(1, hidden_states.shape[0] // bsz)
        positions_flat = positions.reshape(-1)

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states = layer(positions_flat, hidden_states, bsz, seq)

        if get_pp_group().is_last_rank:
            hidden_states = self.norm(hidden_states)
        if orig_shape is not None:
            hidden_states = hidden_states.reshape(orig_shape)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        return hidden_states


def check_serving_config(vllm_config: VllmConfig) -> None:
    """Refuse, at startup, configurations the conv snapshots or the rel-bias layout cannot serve."""
    from vllm_gaudi.extension.runtime import get_config as hpu_config
    from vllm_gaudi.v1.worker.hpu_model_runner import is_interleaved

    cfg = vllm_config.model_config.hf_config
    if not isinstance(cfg, InklingHPUConfig):
        raise TypeError(f"inkling_hpu needs InklingHPUConfig, got {type(cfg).__name__} (plugin not loaded?)")
    cache = vllm_config.cache_config
    problems = []
    if cache.block_size != 128:
        problems.append(f"block_size={cache.block_size} (needs 128)")
    if cache.sliding_window is not None:
        problems.append("cache_config.sliding_window is set (it would make the global layers sliding)")
    if not is_interleaved(cfg):
        problems.append("config is not interleaved sliding-window")
    if getattr(cache, "mamba_cache_mode", "none") != "none":
        problems.append(f"mamba_cache_mode={cache.mamba_cache_mode}")
    if getattr(cache, "prefix_match_unit", None) not in (None, 128):
        problems.append(f"prefix_match_unit={cache.prefix_match_unit} (partial-block hits need block copies)")
    if str(cache.cache_dtype) not in ("auto", "bfloat16"):
        problems.append(f"kv cache_dtype={cache.cache_dtype} (the conv padding assumes a bf16 attention page)")
    if vllm_config.speculative_config is not None:
        problems.append("speculative decoding (the snapshots cannot roll back rejected drafts)")
    hpu = hpu_config()
    if hpu.merged_prefill:
        problems.append("merged prefill (conv slots need one row per request)")
    if hpu.use_contiguous_pa:
        problems.append("contiguous PA (decode slots need block_list in logical order; set VLLM_CONTIGUOUS_PA=false)")
    if os.environ.get("VLLM_PROMPT_BS_BUCKET_MAX", "1") not in ("", "1"):
        problems.append("VLLM_PROMPT_BS_BUCKET_MAX > 1 (the runner's window context mask assumes prefill batch 1)")
    if problems:
        raise ValueError("inkling_hpu: unsupported serving config: " + "; ".join(problems))


def install_kv_hooks() -> None:
    """Wrap two HPUModelRunner methods (in the worker) so the conv state rides on the attention block ids
    without being a KV-cache group.

    get_kv_cache_spec: pad every attention page by the conv bytes, so vLLM's block count (and the HBM budget)
      covers one conv row per layer per block. Refuse any MambaSpec.
    initialize_kv_cache: require one FullAttention group with the padded page and a pool that holds one
      max_model_len request, let the runner allocate the attention cache, then allocate the conv rows
      [num_blocks + 1, 3, dim] for every conv state (row = attention block id; the last row is the pad block).
    """
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
    from vllm_gaudi.v1.worker import hpu_model_runner as runner

    cls = runner.HPUModelRunner
    if getattr(cls.initialize_kv_cache, "_inkling_hooked", False):
        return
    orig_spec = cls.get_kv_cache_spec
    orig_init = cls.initialize_kv_cache

    def get_kv_cache_spec(self):
        specs = orig_spec(self)
        if not cs.CONV_STATES:
            return specs
        out = {}
        for name, spec in specs.items():
            if isinstance(spec, MambaSpec):
                raise ValueError(f"inkling_hpu: unexpected MambaSpec layer {name}; the conv state must not be KV")
            if isinstance(spec, FullAttentionSpec):
                padded = spec.real_page_size_bytes + _CONV_PAGE_BYTES
                if spec.page_size_padded not in (None, padded):
                    raise ValueError(f"inkling_hpu: {name} already padded to {spec.page_size_padded}")
                spec = dataclasses.replace(spec, page_size_padded=padded)
            out[name] = spec
        return out

    def initialize_kv_cache(self, kv_cache_config):
        if not cs.CONV_STATES:
            return orig_init(self, kv_cache_config)
        groups = kv_cache_config.kv_cache_groups
        kinds = [f"{type(g.kv_cache_spec).__name__}x{len(g.layer_names)}" for g in groups]
        print(f"[inkling-hpu] KV groups: {kinds}", flush=True)
        if len(groups) != 1 or not isinstance(groups[0].kv_cache_spec, FullAttentionSpec):
            raise ValueError(f"inkling_hpu: expected ONE FullAttention KV group, got {kinds}")
        spec = groups[0].kv_cache_spec
        if spec.page_size_bytes != spec.real_page_size_bytes + _CONV_PAGE_BYTES:
            raise ValueError("inkling_hpu: attention page is not padded by the conv-state bytes")
        max_len = self.vllm_config.model_config.max_model_len
        pool_tokens = kv_cache_config.num_blocks * spec.block_size
        print(
            f"[inkling-hpu] KV pool: {kv_cache_config.num_blocks} blocks = {pool_tokens} tokens = "
            f"{pool_tokens / max_len:.2f} x max_model_len {max_len} (attention page {spec.real_page_size_bytes} B "
            f"+ conv {_CONV_PAGE_BYTES} B)",
            flush=True,
        )
        if pool_tokens < max_len:
            raise ValueError(
                f"inkling_hpu: KV pool {pool_tokens} tokens cannot hold one max_model_len={max_len} request; "
                "lower --max-model-len or raise the KV budget"
            )
        orig_init(self, kv_cache_config)
        rows = int(self._PAD_BLOCK_ID) + 1
        if rows < kv_cache_config.num_blocks + 1:
            raise ValueError(f"inkling_hpu: runner pad block {rows - 1} < num_blocks {kv_cache_config.num_blocks}")
        total = 0
        for m in cs.CONV_STATES:
            m.kv_cache = tuple(
                torch.zeros((rows, *shape), dtype=dt, device=self.device)
                for shape, dt in zip(m.get_state_shape(), m.get_state_dtype(), strict=True)
            )
            total += sum(t.numel() * t.element_size() for t in m.kv_cache)
        print(f"[inkling-hpu] conv states: {len(cs.CONV_STATES)} layers x {rows} rows = {total / 2**30:.2f} GiB")

    get_kv_cache_spec._inkling_hooked = True
    initialize_kv_cache._inkling_hooked = True
    cls.get_kv_cache_spec = get_kv_cache_spec
    cls.initialize_kv_cache = initialize_kv_cache


class HPUInklingForCausalLM(nn.Module, SupportsPP):
    """Registered for InklingForConditionalGeneration and InklingForCausalLM (text only)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        global _CONV_PAGE_BYTES
        config = vllm_config.model_config.hf_config
        self.config = config
        check_serving_config(vllm_config)
        install_kv_hooks()
        cs.CONV_STATES.clear()
        self.model = InklingHPUModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        # The forward context must hold the 42 attentions and nothing else (the conv states are not KV).
        ctx = vllm_config.compilation_config.static_forward_context.values()
        n_attn = sum(isinstance(m, Attention) for m in ctx)
        other_kv = [m for m in ctx if not isinstance(m, Attention) and hasattr(m, "get_kv_cache_spec")]
        if other_kv or n_attn != config.num_hidden_layers or len(cs.CONV_STATES) != config.num_hidden_layers:
            raise ValueError(
                f"inkling_hpu: expected {config.num_hidden_layers} attentions + conv states, got {n_attn} / "
                f"{len(cs.CONV_STATES)} (+{len(other_kv)} other KV layers)"
            )
        _CONV_PAGE_BYTES = cs.CONV_STATES[0].state_bytes_per_block()
        self.lm_head = ParallelLMHead(
            config.padded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.padded_vocab_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        mup = getattr(config, "logits_mup_width_multiplier", None) or 1.0
        self.logits_processor = LogitsProcessor(
            config.padded_vocab_size, org_vocab_size=config.vocab_size, scale=1.0 / mup
        )
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata=None) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    # Checkpoint names (BF16 release):
    #   model.llm.{embed,embed_norm,norm,unembed}
    #   model.llm.layers.N.{attn_norm,mlp_norm,attn_sconv,mlp_sconv}
    #   model.llm.layers.N.attn.{wq_du,wk_dv,wv_dv,wr_du,wo_ud,q_norm,k_norm,k_sconv,v_sconv,rel_logits_proj.proj}
    #   model.llm.layers.N.mlp.{w13_dn,w2_md,global_scale}                 dense layers
    #   model.llm.layers.N.mlp.{gate.*,experts.*,shared_experts.*}         MoE layers
    #   model.mtp.*, model.audio.*, model.visual.*                          skipped
    _SKIP_PREFIXES = ("model.mtp.", "model.audio.", "model.visual.")
    _QKVR_SHARDS = {"wq_du.weight": 0, "wk_dv.weight": 1, "wv_dv.weight": 2, "wr_du.weight": 3}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())
        loaded: set[str] = set()

        def copy_to(name, w):
            params[name].data.copy_(w)
            loaded.add(name)

        for name, w in weights:
            if name.startswith(self._SKIP_PREFIXES):
                continue
            if name == "model.llm.embed.weight":
                p = params["model.embed_tokens.weight"]
                p.data.zero_()
                p.data[: w.shape[0]].copy_(w)
                loaded.add("model.embed_tokens.weight")
                continue
            if name == "model.llm.embed_norm.weight":
                copy_to("model.embed_norm.weight", w)
                continue
            if name == "model.llm.norm.weight":
                copy_to("model.norm.weight", w)
                continue
            if name == "model.llm.unembed.weight":
                p = params["lm_head.weight"]
                p.weight_loader(p, w)
                loaded.add("lm_head.weight")
                continue

            assert name.startswith("model.llm.layers."), name
            lid = int(name.split("layers.")[1].split(".")[0])
            layer = self.model.layers[lid]
            rest = name.split(f"layers.{lid}.", 1)[1]
            base = f"model.layers.{lid}"

            if rest in ("attn_norm.weight", "mlp_norm.weight"):
                copy_to(f"{base}.{rest}", w)
            elif rest in ("attn_sconv.weight", "mlp_sconv.weight"):
                conv = getattr(layer, rest.split(".")[0])
                conv.weight.weight_loader(conv.weight, w)
                loaded.add(f"{base}.{rest}")
            elif rest.startswith("attn."):
                self._load_attn(layer.attn, rest[len("attn.") :], w, f"{base}.attn", loaded, name)
            elif rest.startswith("mlp."):
                self._load_mlp(layer, rest[len("mlp.") :], w, f"{base}.mlp", loaded, name)
            else:
                raise ValueError(f"unmapped tensor {name}")
        return loaded

    def _load_attn(self, attn, sub, w, base, loaded, name):
        if sub in self._QKVR_SHARDS:
            attn.qkvr.weight_loader(attn.qkvr.weight, w, self._QKVR_SHARDS[sub])
            loaded.add(f"{base}.qkvr.weight")
        elif sub == "wo_ud.weight":
            attn.wo_ud.weight_loader(attn.wo_ud.weight, w)
            loaded.add(f"{base}.wo_ud.weight")
        elif sub in ("q_norm.weight", "k_norm.weight"):
            getattr(attn, sub.split(".")[0]).weight.data.copy_(w)
            loaded.add(f"{base}.{sub}")
        elif sub in ("k_sconv.weight", "v_sconv.weight"):
            conv = getattr(attn, sub.split(".")[0])
            conv.weight.weight_loader(conv.weight, w)
            loaded.add(f"{base}.{sub}")
        elif sub == "rel_logits_proj.proj":
            attn.rel_logits_proj.data.copy_(w)
            loaded.add(f"{base}.rel_logits_proj")
        else:
            raise ValueError(f"unmapped attn tensor {name}")

    def _load_mlp(self, layer, sub, w, base, loaded, name):
        mlp = layer.mlp
        if not layer.is_moe:
            if sub == "w13_dn.weight":
                mlp.load_fused_interleaved(w)
                loaded.add(f"{base}.gate_up_proj.weight")
            elif sub == "w2_md.weight":
                mlp.down_proj.weight_loader(mlp.down_proj.weight, w)
                loaded.add(f"{base}.down_proj.weight")
            elif sub == "global_scale":
                mlp.global_scale.data.copy_(w.reshape(1))
                loaded.add(f"{base}.global_scale")
            else:
                raise ValueError(f"unmapped dense-mlp tensor {name}")
            return
        targets = {
            "experts.w13_weight": "experts.routed_experts.w13_weight",
            "experts.w2_weight": "experts.routed_experts.w2_weight",
            "shared_experts.shared_w13_weight": "sink_experts.w13_weight",
            "shared_experts.shared_w2_weight": "sink_experts.w2_weight",
        }
        if sub == "gate.weight":
            mlp.gate.weight.data.zero_()
            mlp.gate.weight.data[: w.shape[0]].copy_(w.float())
            loaded.add(f"{base}.gate.weight")
        elif sub == "gate.bias":
            mlp.gate_bias.data.copy_(w.float().reshape(-1))
            loaded.add(f"{base}.gate_bias")
        elif sub == "gate.global_scale":
            mlp.gate_global_scale.data.copy_(w.float().reshape(1))
            loaded.add(f"{base}.gate_global_scale")
        elif sub in targets:
            mlp.load_expert_weight(sub, w)
            loaded.add(f"{base}.{targets[sub]}")
        else:
            raise ValueError(f"unmapped moe tensor {name}")
