"""ML4_ATTN_KERNEL=1: the exact gather kernel for the decode sparse attention . Default off. Bit-identical to ML4Attention.absorbed_mm on every gated case.

absorbed_mm reads the 2048 selected latent rows through TaggedCache.read (index math, an index_select of the int8
rows, the tag gather + compare, the i8-t128-f32 decode, a cat) and casts them to fp32. Here one TPC kernel does that
(k2_g2, in libk1_tpc.so): per key the physical row and expected owner (memoised once per forward and top-k, from the
plugin's own BatchCtx.rows), a vector tag check, [bf16(q * scale) | k_pe] as fp32 bits. Everything after the rows is
absorbed_mm's own code: the MME fp32 matmuls and the native softmax, same operands and shapes -> same bits.

Eligible: HPU, decode / verify rows (S <= ML4_DECODE_S) on the absorbed_mm path, ML4_KV_FP8=1 with the served
i8-t128-f32 codec, ML4_TAG_CHECK mask|off, no DCP, the key count a multiple of 64. Anything else: native, unchanged.
The library is the decode-indexer one (k1ix: ML4_INDEX_KERNEL_LIB); either flag registers it.
"""

from __future__ import annotations


import torch

from . import k1ix

ENABLED = k1ix.ATTN


def eligible(m, q, topk, src) -> bool:
    if not (k1ix.ATTN and k1ix._ops_loaded) or q.device.type != "hpu":
        return False
    from . import attention as A
    from . import paged as PG

    B, S, K = topk.shape
    if S > A.DECODE_S or A.BF16_POINTS or not A.decode_mm() or K % 64:
        return False
    if PG.KV_FP8 != "1" or PG.KV_CODEC != "i8-t128-f32" or PG.TAG_MODE not in ("mask", "off"):
        return False
    if not isinstance(src, PG.Fp8LatentCache) or src.fake or src.dcp is not None:
        return False
    return src.data.shape[-1] == 328 and topk.dtype != torch.bool


def _addr(ctx, topk):
    """(idx, prow, pown) int32 [B*S, K], once per forward and top-k tensor (the memo key absorbed_mm's read uses)."""

    def make():
        B, S, K = topk.shape
        x = topk.clamp(min=0)
        rows, ok, owner = ctx.rows(x, ("topk", id(topk), 0), None)
        pown = torch.where((topk >= 0) & ok, owner.to(torch.int32), torch.iinfo(torch.int32).min)
        return (
            x.reshape(B * S, K).to(torch.int32),
            rows.reshape(B * S, K).to(torch.int32),
            pown.reshape(B * S, K).to(torch.int32),
        )

    return ctx.shared(("k2_addr", id(topk)), make)


def rows(ctx, src, topk):
    """== (src.read(ctx, topk.clamp(min=0), topk >= 0)[0].float(), [1]) reshaped to [B*S, K, 576], [B*S, K]."""
    from .paged import TAG_MODE
    from . import r5k

    t32 = ctx.memo.get(("k1_t32", id(topk))) if (r5k.K2FOLD and ctx.memo is not None and r5k.ready()) else None
    if t32 is not None and ctx.block_epoch is not None and ctx.block_size == 128:
        # ML4_R5_K2FOLD: the addressing (_rows + _addr) inside the gather kernel, same rows / owners
        B, S, K = topk.shape
        bt32 = ctx.shared(("k1_bt32",), lambda: ctx.block_table.to(torch.int32))
        k1ix._KEEP.append((t32, bt32))
        r, v = torch.ops.custom_op.k2_g3(
            src.tags.view(-1), src.data, t32.reshape(B * S, K), bt32, k1ix._own(ctx), 0 if TAG_MODE == "off" else 1, S
        )
        return r.view(torch.float32), v != 0
    idx, prow, pown = _addr(ctx, topk)
    k1ix._KEEP.append((idx, prow, pown))
    r, v = torch.ops.custom_op.k2_g2(src.tags.view(-1), src.data, idx, prow, pown, 0 if TAG_MODE == "off" else 1)
    return r.view(torch.float32), v != 0


def absorbed(m, q, topk, ctx, src):
    """== m.absorbed_mm(q, topk, ctx, src, *m.w_uk_uv(), softmax_scale()) for eligible() rows: [B, S, Hl, 128]."""
    from .attention import KV_RANK, V_DIM, NEG_FLOOR, softmax_scale

    B, S = ctx.positions.shape
    H, P = m.Hl, m.P
    W_UK, W_UV = m.w_uk_uv()
    r32, ok = rows(ctx, src, topk)
    k = r32.shape[1]
    cf = r32[..., :KV_RANK]
    qf = q.float().reshape(B * S, H, q.shape[-1])
    ql = torch.matmul(qf[..., :P].transpose(0, 1), W_UK.float()).transpose(0, 1)  # [BS, H, 512]
    lg = torch.matmul(ql, cf.transpose(1, 2))  # [BS, H, k]
    if m.role.use_rope:
        lg = lg + torch.matmul(qf[..., P:], r32[..., KV_RANK:].transpose(1, 2))
    lg = (lg * softmax_scale()).masked_fill(~ok.reshape(B * S, 1, k), float("-inf"))
    mx = lg.amax(-1, keepdim=True).clamp_min(NEG_FLOOR)
    e = torch.exp(lg - mx)
    pr = e / e.sum(-1, keepdim=True).clamp_min(1e-30)
    ol = torch.matmul(pr, cf)  # [BS, H, 512]
    o = torch.matmul(ol.transpose(0, 1), W_UV.float().transpose(1, 2)).transpose(0, 1)  # [BS, H, 128]
    return o.reshape(B, S, H, V_DIM).to(q.dtype)
