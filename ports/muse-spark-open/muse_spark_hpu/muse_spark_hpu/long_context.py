# SPDX-License-Identifier: Apache-2.0
"""Long-context prompt attention for the global layers, and the decode graph cap.

MUSE_SPARK_LONG_ATTN=1. A prefill chunk with cached context runs the 16 global (NoPE) layers through FusedSDPA
with an explicit bias. vllm-gaudi builds that bias densely, [bs, 1, q, ctx + q] in bf16 plus a bool mask of the
same shape while building it (8 GiB + 4 GiB at 1M context with 4096-token chunks), and FusedSDPA keeps a
workspace of 4 B x heads x query rows x keys for the life of the process (fp32 softmax). From MIN_KV keys on:
  * the metadata processor stores a compact key-validity row instead of the dense bias:
    keyrow[b, 0, 0, k] = -inf for past keys k >= context_len[b] (bucket padding) and for current keys
    k - ctx_max >= seq_len[b], else 0;  [bs, 1, 1, ctx_max + q] bf16;
  * prompt attention splits the queries into tiles of T rows, T the largest power of two dividing q with
    4 x heads x bs x T x keys <= WS_GIB, builds each tile's bias rows as where(k - ctx_max > i, -inf, keyrow)
    (exactly the dense bias rows) and calls FusedSDPA per tile with vllm-gaudi's own q-tiling arguments;
  * each tile is its own lazy graph, and the host waits on the HPU event of the tile INFLIGHT back, so launched
    but unrun tiles cannot pile up their bias tiles in HBM (+7 GiB at 1M without the bound).
Query rows are independent, so this is the same attention as one call over the dense bias. Sliding layers
(window bias), first chunks (no bias) and decode are untouched.

MUSE_SPARK_DECODE_GRAPH_MAX_BLOCKS=N (default 0, no cap): decode batches whose block list is longer than N run
without an HPU graph. Each captured whole-model decode graph keeps about 0.5 MiB per block of its bucket for the
life of the process, and every bucket keeps its own, so long-context decode buckets would stack several GiB.
"""

from __future__ import annotations

import logging
import math
import os

import torch

logger = logging.getLogger("vllm.muse_spark_hpu")

ENABLED = os.environ.get("MUSE_SPARK_LONG_ATTN", "0") == "1"
DECODE_GRAPH_MAX_BLOCKS = int(os.environ.get("MUSE_SPARK_DECODE_GRAPH_MAX_BLOCKS", "0"))
MIN_KV = 65536
WS_GIB = 2.0
INFLIGHT = 2
MIN_TILE = 8
_state = {"installed": False, "cap_installed": False, "logged": set()}


def compact_bias(
    context_lens_t: torch.Tensor, seq_lens_t: torch.Tensor, ctx_max: int, seq_len: int, dtype: torch.dtype, device
) -> torch.Tensor:
    """[bs, 1, 1, ctx_max + seq_len]: 0 for valid keys, -inf for padding (the key masks of
    HPUAttentionMetadataProcessor._set_attn_bias; the causal part is added per tile)."""
    bs = context_lens_t.shape[0]
    past = torch.arange(ctx_max, dtype=torch.int32, device=device).view(1, -1) >= context_lens_t.view(-1, 1)
    cur = torch.arange(seq_len, dtype=torch.int32, device=device).view(1, -1) >= seq_lens_t.view(-1, 1)
    mask = torch.cat([past, cur], dim=-1).view(bs, 1, 1, ctx_max + seq_len)
    return torch.zeros(mask.shape, dtype=dtype, device=device).masked_fill_(mask, -math.inf)


def tile_bias(keyrow: torch.Tensor, ctx_max: int, start: int, rows: int) -> torch.Tensor:
    """Rows [start, start + rows) of the dense bias: keyrow plus the chunk's causal mask
    (current key j = k - ctx_max is visible to query i iff j <= i)."""
    kv = keyrow.shape[-1]
    kidx = torch.arange(kv, dtype=torch.int32, device=keyrow.device).view(1, 1, 1, kv)
    qi = torch.arange(start, start + rows, dtype=torch.int32, device=keyrow.device).view(1, 1, rows, 1)
    ninf = torch.tensor(-math.inf, dtype=keyrow.dtype, device=keyrow.device)
    return torch.where(kidx - ctx_max > qi, ninf, keyrow)


def choose_tile(q_len: int, bs: int, heads: int, kv: int, fp32_softmax: bool = True) -> int:
    """Largest power-of-two divisor of q_len whose FusedSDPA workspace fits WS_GIB."""
    per_row = (4 if fp32_softmax else 2) * heads * bs * kv
    limit = max(MIN_TILE, int(WS_GIB * 2**30 // per_row))
    t = 1
    while t * 2 <= min(limit, q_len) and q_len % (t * 2) == 0:
        t *= 2
    return t


def _tiled_fsdpa(query, key, value, scale, fsdpa_op, is_causal, keyrow, valid_seq_lengths, fp32=True):
    """query [B, S, Hq, D], key / value [B, ctx_max + S, Hkv, D] (past included), keyrow [B, 1, 1, ctx_max + S]
    -> [B, S, Hq, D]."""
    import habana_frameworks.torch as htorch

    B, S, Hq = query.shape[0], query.shape[1], query.shape[2]
    kv = key.shape[1]
    assert keyrow.size(-1) == kv, (tuple(keyrow.shape), tuple(key.shape))
    ctx_max = kv - S
    T = choose_tile(S, B, Hq, kv, fp32)
    q, k, v = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    softmax_mode = "fp32" if fp32 else "fast"
    outs, events = [], []
    for a in range(0, S, T):
        bias_t = tile_bias(keyrow, ctx_max, a, T)
        outs.append(
            fsdpa_op(
                q[..., a : a + T, :],
                k,
                v,
                bias_t,
                0.0,
                is_causal,
                scale,
                softmax_mode,
                True,
                valid_seq_lengths,
                "right",
            )
        )
        htorch.core.mark_step()
        ev = htorch.hpu.Event()
        ev.record()
        events.append(ev)
        if len(events) > INFLIGHT:
            events.pop(0).synchronize()
    if (S, kv, T) not in _state["logged"] and len(_state["logged"]) < 64:
        _state["logged"].add((S, kv, T))
        logger.warning(
            "muse_spark long_attn: q %d x keys %d (bs %d, %d heads): %d query tiles of %d rows",
            S,
            kv,
            B,
            Hq,
            S // T,
            T,
        )
    return torch.cat(outs, dim=-2).transpose(1, 2)


def _install_decode_graph_cap() -> None:
    from vllm_gaudi.v1.worker import hpu_model_runner as hmr

    R = hmr.HPUModelRunner
    stock_use = R._use_graphs

    def _use_graphs(self, attn_metadata, batch_size):
        use = stock_use(self, attn_metadata, batch_size)
        decode = use and attn_metadata is not None and not attn_metadata.is_prompt
        if decode and attn_metadata.num_blocks() > DECODE_GRAPH_MAX_BLOCKS:
            return False
        return use

    R._use_graphs = _use_graphs
    _state["cap_installed"] = True
    logger.warning("muse_spark_hpu: decode HPU graphs only up to %d blocks per batch", DECODE_GRAPH_MAX_BLOCKS)


def install() -> None:
    if DECODE_GRAPH_MAX_BLOCKS > 0 and not _state["cap_installed"]:
        _install_decode_graph_cap()
    if _state["installed"] or not ENABLED:
        return
    from vllm_gaudi.extension import ops as hpu_ops
    from vllm_gaudi.extension.runtime import get_config
    from vllm_gaudi.v1.worker import hpu_model_runner as hmr

    P = hmr.HPUAttentionMetadataProcessor
    stock_bias = P._set_attn_bias

    def _set_attn_bias(self, attn_metadata, batch_size, seq_len, device, dtype):
        md = attn_metadata
        if (
            md is None
            or not md.is_prompt
            or not self.prefill_use_fusedsdpa
            or md.block_list is None
            or md.attn_bias is not None
        ):
            return stock_bias(self, attn_metadata, batch_size, seq_len, device, dtype)
        block_size = getattr(md, "block_size", self.block_size)
        ctx_max = (md.block_list.size(-1) // batch_size) * block_size
        if ctx_max + seq_len < MIN_KV:
            return stock_bias(self, attn_metadata, batch_size, seq_len, device, dtype)
        keyrow = compact_bias(md.context_lens_tensor, md.seq_lens_tensor, ctx_max, seq_len, dtype, device)
        return hmr.custom_tuple_replace(md, "TrimmedAttentionMetadata", attn_bias=keyrow)

    P._set_attn_bias = _set_attn_bias
    stock_fsdpa = hpu_ops._fsdpa_prompt_attention

    def _fsdpa_prompt_attention(
        query,
        key,
        value,
        scale,
        fsdpa_op,
        is_causal,
        attn_bias=None,
        valid_seq_lengths=None,
        window_size=None,
        sinks=None,
        **kwargs,
    ):
        if (
            attn_bias is None
            or attn_bias.dim() != 4
            or attn_bias.size(-2) != 1
            or query.size(1) <= 1
            or window_size is not None
            or sinks is not None
        ):
            return stock_fsdpa(
                query,
                key,
                value,
                scale,
                fsdpa_op,
                is_causal,
                attn_bias=attn_bias,
                valid_seq_lengths=valid_seq_lengths,
                window_size=window_size,
                sinks=sinks,
                **kwargs,
            )
        return _tiled_fsdpa(
            query, key, value, scale, fsdpa_op, is_causal, attn_bias, valid_seq_lengths, bool(get_config().fp32_softmax)
        )

    hpu_ops._fsdpa_prompt_attention = _fsdpa_prompt_attention
    _state["installed"] = True
    logger.warning(
        "muse_spark_hpu: long-context prompt attention on (compact bias from %d keys, tiles within a %g GiB "
        "FusedSDPA workspace, %d tiles in flight)",
        MIN_KV,
        WS_GIB,
        INFLIGHT,
    )
