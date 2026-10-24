# SPDX-License-Identifier: Apache-2.0
"""HPU attention for MiMo-V2: exact sliding window with sinks, and tiled global-attention prefill.

The classes subclass vllm-gaudi's HPUAttentionImpl and are built lazily, so this module imports on a machine without
vllm_gaudi. The model swaps them onto its layers' attention impls at construction (model.py).
"""

from __future__ import annotations

import math
import os

import torch

from .ops import ga_tile_mask, ga_tile_rows, swa_band_attention

# MIMO_GA_TILE=1: global layers prefill with context in query tiles, and the patched runner skips building its
# [chunk, context + chunk] bias (8.7 GB of bf16 at 512K context with 8192-token chunks). Off: the stock prefill.
GA_TILE = os.environ.get("MIMO_GA_TILE", "0") == "1"

# Set by swa_kv.SwaKvGroups.after_bind when window-sized KV groups are on (MIMO_SWA_KV_GROUPS=1).
_SWA_KV_ACTIVE = False


def set_swa_kv_groups_active(active: bool) -> None:
    global _SWA_KV_ACTIVE
    _SWA_KV_ACTIVE = bool(active)


def _make_swa_impl_class():
    from vllm_gaudi.attention.backends.hpu_attn import HPUAttentionImpl, _set_fetch_by_id
    from vllm_gaudi.attention.ops.hpu_paged_attn import HPUPagedAttention
    from vllm_gaudi.extension.ops import batch2block, block2batch, grouped_max
    from vllm_gaudi.extension.runtime import get_config
    from vllm_gaudi.extension.utils import get_kv_fetch_extra_args

    class MiMoSWAHPUAttentionImpl(HPUAttentionImpl):
        """Window attention with a learned sink, exact in prefill and decode.

        Decode expects the runner's exact-window lists (fixes/swa-decode-window plus patches/runner_window.py):
        window_block_list holds the last window // block_size + 1 blocks of each sequence, and the oldest block's
        usage is stored as -lbu, meaning slots [lbu, block_size) are valid. For window == block_size those are
        exactly positions (q - window, q]. The sink is counted once per sequence: each of a sequence's n blocks
        gets the sink logit minus log(n), so the n rescaled copies add up to one exp(sink - max).
        """

        def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None):
            if not isinstance(kv_cache, tuple) or self.kv_sharing_target_layer_name is not None:
                return super().forward(layer, query, key, value, kv_cache, attn_metadata, output)
            if attn_metadata.is_prompt:
                return self._forward_prefill(query, key, value, kv_cache, attn_metadata)
            if getattr(attn_metadata, "window_block_list", None) is None:
                if _SWA_KV_ACTIVE:
                    # The stock path would read the global group's block list through this layer's shared tensor,
                    # which holds another layer's K/V.
                    raise RuntimeError("mimo_v2_hpu: window-sized KV is on but a sliding decode got no window list")
                return super().forward(layer, query, key, value, kv_cache, attn_metadata, output)
            return self._forward_decode(query, key, value, kv_cache, attn_metadata)

        def _forward_decode(self, query, key, value, kv_cache, attn_metadata):
            output_shape = query.shape
            if query.dim() == 2:
                batch_size = (
                    attn_metadata.block_mapping.shape[1]
                    if attn_metadata.seq_lens_tensor is None
                    else attn_metadata.seq_lens_tensor.shape[0]
                )
                query = query.view(batch_size, query.shape[0] // batch_size, -1)

            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
            slot_mapping = attn_metadata.slot_mapping.flatten()
            key_cache, value_cache, k_scales, v_scales = HPUPagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size
            )
            if query.dtype != key.dtype:
                query = query.to(key.dtype)
            bs = attn_metadata.block_size
            key_cache = self.k_cache(key, key_cache, slot_mapping, scales=k_scales, block_size=bs, is_prompt=False)
            value_cache = self.v_cache(
                value, value_cache, slot_mapping, scales=v_scales, block_size=bs, is_prompt=False
            )
            _set_fetch_by_id(self.k_cache, False)
            _set_fetch_by_id(self.v_cache, False)

            block_mapping = attn_metadata.window_block_mapping
            usage = attn_metadata.window_block_usage
            slots = torch.arange(bs, device=usage.device, dtype=usage.dtype).unsqueeze(0)
            u = usage.unsqueeze(-1)
            masked = torch.where(u > 0, slots >= u, slots < -u)
            bias = torch.zeros(masked.shape, dtype=query.dtype, device=query.device).masked_fill_(masked, -math.inf)
            n_seq = block_mapping.sum(0)  # [B] blocks per sequence
            n_blk = torch.matmul(block_mapping, n_seq.unsqueeze(-1)).squeeze(-1).clamp(min=1.0)
            out = _flat_pa_exact_sink(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_list=attn_metadata.window_block_list,
                block_mapping=block_mapping,
                block_bias=bias,
                block_groups=attn_metadata.window_block_groups,
                block_size=bs,
                scale=self.scale,
                sinks=self.sinks,
                log_n=torch.log(n_blk.float()),
                matmul_qk_op=self.matmul_qk,
                matmul_av_op=self.matmul_av,
                batch2block_matmul_op=self.batch2block_matmul,
                block2batch_matmul_op=self.block2batch_matmul,
                keys_fetch_func=self.k_cache.fetch_from_cache,
                values_fetch_func=self.v_cache.fetch_from_cache,
                k_scales=k_scales,
                v_scales=v_scales,
            )
            return out.view(*output_shape)

        def _forward_prefill(self, query, key, value, kv_cache, attn_metadata):
            """Banded window attention with the sink (ops.swa_band_attention). FusedSDPA on Gaudi software 1.24.1
            ignores both `sinks` and `window_size`; with q.k logits near -25 and sinks near +1 the sink takes almost
            all the attention mass, so dropping it is not a small error."""
            if get_config().merged_prefill:
                raise NotImplementedError("mimo_v2_hpu: sliding-window prefill does not support merged prefill")
            output_shape = query.shape
            if query.dim() == 2:
                B = attn_metadata.seq_lens_tensor.shape[0]
                query = query.view(B, query.shape[0] // B, -1)
            B, S, _ = query.shape
            nh, kvh, d = self.num_heads, self.num_kv_heads, self.head_size

            key = key.view(-1, kvh, d)
            value = value.view(-1, kvh, d)
            key_cache, value_cache, k_scales, v_scales = HPUPagedAttention.split_kv_cache(kv_cache, kvh, d)
            if attn_metadata.slot_mapping is not None:
                slot_mapping = attn_metadata.slot_mapping.flatten()
                bs = attn_metadata.block_size
                key_cache = self.k_cache(key, key_cache, slot_mapping, scales=k_scales, block_size=bs, is_prompt=True)
                value_cache = self.v_cache(
                    value, value_cache, slot_mapping, scales=v_scales, block_size=bs, is_prompt=True
                )

            kc = vc = ctx = frame = None
            ctx_list = getattr(attn_metadata, "window_block_list", None)
            ctx_lens = getattr(attn_metadata, "context_lens_tensor", None)
            if ctx_list is not None and ctx_lens is not None:
                bs = attn_metadata.block_size
                Wc = (ctx_list.numel() // B) * bs
                # The runner pads window lists with -1 (short contexts, padding rows). Gather block 0 there: those
                # slots sit at positions >= ctx and are masked.
                ids = ctx_list.flatten().clamp(min=0)
                kc = key_cache.unflatten(0, (-1, bs)).index_select(0, ids).view(B, Wc, kvh, d)
                vc = value_cache.unflatten(0, (-1, bs)).index_select(0, ids).view(B, Wc, kvh, d)
                ctx = ctx_lens.to(torch.long).view(B)
                frame = torch.clamp((ctx + bs - 1) // bs * bs - Wc, min=0)  # absolute position of the first slot
            o = swa_band_attention(
                query.view(B, S, nh, d),
                key.view(B, S, kvh, d),
                value.view(B, S, kvh, d),
                self.sinks,
                self.scale,
                self.sliding_window,
                kc,
                vc,
                ctx,
                frame,
            )
            return o.reshape(output_shape)

    def _flat_pa_exact_sink(
        query,
        key_cache,
        value_cache,
        block_list,
        block_mapping,
        block_bias,
        block_groups,
        block_size,
        scale,
        sinks,
        log_n,
        matmul_qk_op,
        matmul_av_op,
        batch2block_matmul_op,
        block2batch_matmul_op,
        keys_fetch_func,
        values_fetch_func,
        k_scales,
        v_scales,
    ):
        """vllm-gaudi's flat_pa (unfused branch) with fp32 softmax statistics and a per-block sink."""
        batch_size, _, hidden_size = query.shape
        _, kv_heads, head_size = key_cache.shape
        q_heads = hidden_size // head_size
        k_scales_uf = k_scales.unflatten(0, (-1, block_size)) if k_scales is not None else None
        v_scales_uf = (v_scales[0].unflatten(0, (-1, block_size)), v_scales[1]) if v_scales is not None else None

        q = batch2block(scale * query, block_mapping, batch2block_matmul_op).view(-1, q_heads, 1, head_size)
        key = keys_fetch_func(
            key_cache.unflatten(0, (-1, block_size)), **get_kv_fetch_extra_args(blocks=block_list, scales=k_scales_uf)
        ).transpose(1, 2)
        value = values_fetch_func(
            value_cache.unflatten(0, (-1, block_size)),
            **get_kv_fetch_extra_args(blocks=block_list, scales=v_scales_uf),
        ).transpose(1, 2)
        n_blocks = key.size(0)
        bias = block_bias.view(n_blocks, 1, 1, -1)
        sink = sinks.float().view(1, q_heads, 1, 1) - log_n.view(-1, 1, 1, 1)  # [Nb, H, 1, 1]
        if kv_heads != q_heads:
            q = q.unflatten(1, (kv_heads, -1))
            key = key.unflatten(1, (kv_heads, 1))
            value = value.unflatten(1, (kv_heads, 1))
            bias = bias.unsqueeze(2)
            sink = sink.unflatten(1, (kv_heads, -1))
        key = key.transpose(-2, -1)

        attn = matmul_qk_op(q, key).float() + bias.float()
        block_max = torch.maximum(attn.amax(dim=-1, keepdim=True), sink)
        attn = (attn - block_max).exp()
        block_sums = attn.sum(dim=-1, keepdim=True) + (sink - block_max).exp()
        attn = matmul_av_op(attn.to(value.dtype), value)

        shape = block_max.shape
        block_max = block_max.squeeze((-1, -2))
        block_sums = block_sums.squeeze((-1, -2))
        group_max = grouped_max(block_max, batch_size, block_groups)
        block_adjustment = (block_max - group_max).exp()
        sum_adjusted = block_sums.mul(block_adjustment)
        bm32 = block_mapping.float()  # reduce the fp32 statistics with an fp32 copy of the (small) mapping
        group_sum = batch2block(block2batch(sum_adjusted, bm32), bm32)
        sum_adjusted = sum_adjusted.view(*shape)
        group_sum = torch.maximum(group_sum.view(*shape), sum_adjusted)
        attn = attn.mul(block_adjustment.view(*shape).div(group_sum).to(attn.dtype))

        attn = block2batch(attn, block_mapping, block2batch_matmul_op).squeeze(-2)
        if kv_heads != q_heads:
            attn = attn.flatten(1, 2)
        return attn

    return MiMoSWAHPUAttentionImpl


def _make_ga_impl_class():
    from vllm_gaudi.attention.backends.hpu_attn import HPUAttentionImpl
    from vllm_gaudi.attention.ops.hpu_paged_attn import HPUPagedAttention
    from vllm_gaudi.extension import ops as hpu_ops

    class MiMoGATiledHPUAttentionImpl(HPUAttentionImpl):
        """Global-attention prefill with context, in query tiles (MIMO_GA_TILE=1).

        The same keys and the same mask as the stock path (ops.ga_tile_mask rebuilds the runner's bias per tile),
        so every query row gets exactly the stock softmax; only the bias and FusedSDPA's score buffer shrink to
        tile x (context + chunk). Softmax is per query row, so the tiles need no merge. Decode and prompts without
        context take the stock path.
        """

        def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None):
            if (
                not attn_metadata.is_prompt
                or attn_metadata.block_list is None
                or not isinstance(kv_cache, tuple)
                or self.kv_sharing_target_layer_name is not None
            ):
                return super().forward(layer, query, key, value, kv_cache, attn_metadata, output)
            out_shape = query.shape
            B = attn_metadata.seq_lens_tensor.shape[0]
            if query.dim() == 2:
                query = query.view(B, query.shape[0] // B, -1)
            S = query.shape[1]
            H, kvh, d, bs = self.num_heads, self.num_kv_heads, self.head_size, attn_metadata.block_size
            key = key.view(-1, kvh, d)
            value = value.view(-1, kvh, d)
            key_cache, value_cache, k_scales, v_scales = HPUPagedAttention.split_kv_cache(kv_cache, kvh, d)
            if query.dtype != key.dtype:
                query = query.to(key.dtype)
            slot_mapping = attn_metadata.slot_mapping.flatten()
            key_cache = self.k_cache(key, key_cache, slot_mapping, scales=k_scales, block_size=bs, is_prompt=True)
            value_cache = self.v_cache(value, value_cache, slot_mapping, scales=v_scales, block_size=bs, is_prompt=True)
            ids = attn_metadata.block_list.flatten()
            past_k = key_cache.unflatten(0, (-1, bs)).index_select(0, ids).view(B, -1, kvh, d)
            past_v = value_cache.unflatten(0, (-1, bs)).index_select(0, ids).view(B, -1, kvh, d)
            P = past_k.shape[1]
            keys = torch.cat([past_k, key.view(B, S, kvh, d)], dim=1)
            vals = torch.cat([past_v, value.view(B, S, kvh, d)], dim=1)
            ctx = attn_metadata.context_lens_tensor.view(B).to(torch.int64)
            seq = attn_metadata.seq_lens_tensor.view(B).to(torch.int64)
            rows_per = ga_tile_rows(S, P + S, heads=H)
            args = self.common_attention_args(None, None, None, bs, k_scales, v_scales)
            outs = []
            for t0 in range(0, S, rows_per):
                rows = torch.arange(t0, min(S, t0 + rows_per), device=query.device)
                mask = ga_tile_mask(rows, P, S, ctx, seq)
                bias = torch.zeros(mask.shape, dtype=query.dtype, device=query.device).masked_fill_(mask, -math.inf)
                q = query[:, t0 : t0 + rows.numel()].view(B, rows.numel(), H, d)
                o = hpu_ops.prompt_attention(
                    impl=self.prefill_impl,
                    query=q,
                    key=keys,
                    value=vals,
                    is_causal=False,
                    attn_bias=bias,
                    valid_seq_lengths=None,
                    **args,
                )
                outs.append(o.reshape(B, rows.numel(), H * d))
            out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
            return out.reshape(out_shape)

    return MiMoGATiledHPUAttentionImpl


_CLASSES: dict = {}


def swa_impl_class():
    if "swa" not in _CLASSES:
        _CLASSES["swa"] = _make_swa_impl_class()
    return _CLASSES["swa"]


def ga_impl_class():
    if "ga" not in _CLASSES:
        _CLASSES["ga"] = _make_ga_impl_class()
    return _CLASSES["ga"]
