# SPDX-License-Identifier: Apache-2.0
"""Pure torch pieces of the MiMo-V2 port: weight decoding, attention math and the RoPE table.

Nothing here imports vLLM or the Gaudi stack, so the CPU tests exercise exactly the code the server runs.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

FP8_BLOCK = 128  # block size of the checkpoint's FP8 block scales
MXFP4_BLOCK = 32  # MXFP4 elements per E8M0 scale


# --------------------------------------------------------------------------------------------------------------------
# FP8 block weights (fused qkv_proj, the dense MLP of layer 0)
# --------------------------------------------------------------------------------------------------------------------
def dequant_fp8_block(w: torch.Tensor, s: torch.Tensor, block: int = FP8_BLOCK) -> torch.Tensor:
    """[R, C] e4m3 times [ceil(R/b), ceil(C/b)] fp32 scale_inv, as [R, C] bf16."""
    rows, cols = w.shape
    s_full = s.float().repeat_interleave(block, 0)[:rows].repeat_interleave(block, 1)[:, :cols]
    return (w.float() * s_full).to(torch.bfloat16)


def qkv_rank_rows(
    num_heads: int, num_kv_heads: int, head_dim: int, v_head_dim: int, ckpt_tp: int, tp_rank: int, tp_size: int
) -> torch.Tensor:
    """Row indices of one rank's [Q | K | V] slice inside the checkpoint's fused qkv_proj.

    The checkpoint stores qkv_proj in `ckpt_tp` chunks, each [Q_c | K_c | V_c] for its share of heads (the layout
    upstream vLLM reads since vllm-project/vllm#57508), not as one flat [Q | K | V]. QKVParallelLinear wants, per
    rank, its own contiguous Q heads, then its KV heads (replicated when tp_size > num_kv_heads), then the same KV
    heads' V rows.
    """
    if num_heads % ckpt_tp or num_kv_heads % ckpt_tp:
        raise ValueError(f"qkv_proj chunking {ckpt_tp} does not divide heads {num_heads}/{num_kv_heads}")
    if num_heads % tp_size:
        raise ValueError(f"num_heads={num_heads} not divisible by tp_size={tp_size}")
    q_per_chunk = (num_heads // ckpt_tp) * head_dim
    k_per_chunk = (num_kv_heads // ckpt_tp) * head_dim
    v_per_chunk = (num_kv_heads // ckpt_tp) * v_head_dim
    rows_per_chunk = q_per_chunk + k_per_chunk + v_per_chunk
    qh_per_chunk = num_heads // ckpt_tp
    kvh_per_chunk = num_kv_heads // ckpt_tp

    q_ids = range(tp_rank * (num_heads // tp_size), (tp_rank + 1) * (num_heads // tp_size))
    if tp_size <= num_kv_heads:
        if num_kv_heads % tp_size:
            raise ValueError(f"num_kv_heads={num_kv_heads} not divisible by tp_size={tp_size}")
        kv_ids = list(range(tp_rank * (num_kv_heads // tp_size), (tp_rank + 1) * (num_kv_heads // tp_size)))
    else:
        if tp_size % num_kv_heads:
            raise ValueError(f"tp_size={tp_size} not a multiple of num_kv_heads={num_kv_heads}")
        kv_ids = [tp_rank // (tp_size // num_kv_heads)]

    hd = torch.arange(head_dim)
    vd = torch.arange(v_head_dim)
    idx = []
    for h in q_ids:
        c = h // qh_per_chunk
        idx.append(c * rows_per_chunk + (h % qh_per_chunk) * head_dim + hd)
    for h in kv_ids:
        c = h // kvh_per_chunk
        idx.append(c * rows_per_chunk + q_per_chunk + (h % kvh_per_chunk) * head_dim + hd)
    for h in kv_ids:
        c = h // kvh_per_chunk
        idx.append(c * rows_per_chunk + q_per_chunk + k_per_chunk + (h % kvh_per_chunk) * v_head_dim + vd)
    return torch.cat(idx)


def qkv_scale_index(total_rows: int, scale_rows: int, ckpt_tp: int, block: int = FP8_BLOCK) -> torch.Tensor:
    """Scale row of every qkv_proj weight row.

    Scales are tiled per chunk (each chunk padded to whole blocks) or, when chunks are block aligned, one continuous
    grid. Anything else raises.
    """
    rows_per_chunk = total_rows // ckpt_tp
    chunk_scale_rows = -(-rows_per_chunk // block)
    r = torch.arange(total_rows)
    if scale_rows == ckpt_tp * chunk_scale_rows:
        return (r // rows_per_chunk) * chunk_scale_rows + (r % rows_per_chunk) // block
    if scale_rows == -(-total_rows // block):
        return r // block
    raise ValueError(
        f"qkv scale has {scale_rows} rows; expected {ckpt_tp * chunk_scale_rows} (per chunk) "
        f"or {-(-total_rows // block)} (continuous)"
    )


def dequant_qkv(w: torch.Tensor, s: torch.Tensor, ckpt_tp: int) -> torch.Tensor:
    """The fused qkv_proj in bf16: every row gets its own chunk's scale row first."""
    s_rows = s.float()[qkv_scale_index(w.shape[0], s.shape[0], ckpt_tp)]
    return (w.float() * s_rows.repeat_interleave(FP8_BLOCK, 1)[:, : w.shape[1]]).to(torch.bfloat16)


# --------------------------------------------------------------------------------------------------------------------
# Routed experts: exact MXFP4 to per-channel FP8
# --------------------------------------------------------------------------------------------------------------------
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_E2M1_EXP = (-1000, -1, 0, 0, 1, 1, 2, 2, -1000, -1, 0, 0, 1, 1, 2, 2)  # floor(log2|v|); zeros get -1000
# Exponent window of the emitted e4m3 values: top binade 2^7 (|q| <= 192, below Gaudi 2's e4m3 maximum of 240) and
# bottom 2^-6, the smallest e4m3 normal. The bytes go straight to the MoE op (vLLM's FP8 loader, which halves Gaudi 2
# weights and would need one more binade of headroom, is not involved).
FP8_EXP_MAX, FP8_EXP_MIN = 7, -6
_P2_LO, _P2_HI = -60, 60
_TABLES: dict = {}


def mxfp4_to_fp8(w: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed MXFP4 [rows, cols/2] (low nibble first) plus E8M0 [rows, cols/32], as (e4m3 [rows, cols], fp32 [rows]).

    Each output row gets a power-of-two scale 2^k with k = (largest exponent in the row) - 7. Every MXFP4 value is
    e2m1 * 2^m with integer m, so the scaled value is exact in fp32, and the e4m3 cast is exact whenever it is
    representable. The round trip below proves it on every tensor: any element that would need rounding raises,
    so a checkpoint the format cannot hold exactly never loads. -0 becomes +0.
    """
    dev = w.device
    if dev not in _TABLES:
        _TABLES[dev] = (
            torch.tensor(_E2M1, dtype=torch.float32, device=dev),
            torch.tensor(_E2M1_EXP, dtype=torch.int32, device=dev),
            torch.tensor([math.ldexp(1.0, e) for e in range(_P2_LO, _P2_HI + 1)], dtype=torch.float32, device=dev),
        )
    e2m1, e2m1_exp, pow2 = _TABLES[dev]
    rows, _ = w.shape
    nb = s.shape[1]
    nib = torch.stack((w & 0x0F, w >> 4), dim=-1).reshape(rows, nb, MXFP4_BLOCK).to(torch.int32)
    blk = s.to(torch.int32) - 127  # [rows, nb] block exponents
    emax = (e2m1_exp[nib].amax(-1) + blk).amax(-1)  # [rows]
    k = torch.where(emax < -500, torch.zeros_like(emax), emax - FP8_EXP_MAX)  # all-zero rows keep k = 0
    shift = (blk - k[:, None]).clamp(_P2_LO, _P2_HI) - _P2_LO
    q = (e2m1[nib] * pow2[shift][..., None]).reshape(rows, nb * MXFP4_BLOCK) + 0.0
    q8 = q.to(torch.float8_e4m3fn)
    nz = q != 0
    lowest = torch.where(nz, torch.floor(torch.log2(q.abs().clamp(min=1e-30))), torch.full_like(q, 99.0)).amin()
    n_bad = int((q8.to(torch.float32) != q).sum())
    q_max = float(q.abs().amax())
    if n_bad or float(lowest) < FP8_EXP_MIN or q_max > 2.0**FP8_EXP_MAX * 1.875:
        raise RuntimeError(
            f"MXFP4 to FP8 re-encode is not exact: {n_bad} elements changed, lowest exponent {float(lowest)}, "
            f"max {q_max}, row exponents {int(k.min())}..{int(k.max())}"
        )
    return q8, pow2[(k - _P2_LO).clamp(0, _P2_HI - _P2_LO)].contiguous()


def mxfp4_reference(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """fp64 decode of packed MXFP4 plus E8M0, for tests."""
    lut = torch.tensor(_E2M1, dtype=torch.float64)
    rows = w.shape[0]
    nib = torch.stack((w & 0x0F, w >> 4), dim=-1).reshape(rows, -1).long()
    scale = torch.pow(2.0, s.double() - 127).repeat_interleave(MXFP4_BLOCK, 1)
    return lut[nib] * scale


# --------------------------------------------------------------------------------------------------------------------
# Sliding-window attention with a sink, prefill
# --------------------------------------------------------------------------------------------------------------------
def swa_band_attention(q, k, v, sinks, scale, window, kc=None, vc=None, ctx=None, frame=None):
    """Causal window attention with a per-head sink logit, for one prefill chunk.

    q [B, S, nh, d]; k, v [B, S, kvh, d] are the chunk's tokens at absolute positions ctx + i; sinks [nh]. Query i
    sees keys (i - window, i] plus the sink, as the reference model does (kv_idx > q_idx - window). Optional
    context: kc, vc [B, Wc, kvh, d] hold absolute positions frame + j; only those below ctx and inside the window
    are visible. Returns [B, S, nh * d].

    Queries are tiled by the window: tile t sees chunk keys [(t - 1) * window, (t + 1) * window), and only tile 0
    can see context, so memory is O(S * window). The softmax runs in fp32 with the sink as one extra logit. Every
    product is a batched matmul on contiguous [..., group, rows, d] tensors: a 6-D einsum with a shared KV-group dim
    returned the first group's values for every group on Gaudi (software 1.24.1).
    """
    B, S, nh, d = q.shape
    kvh = k.shape[2]
    rep = nh // kvh
    W = window
    pad = (-S) % W
    if pad:
        q = F.pad(q, (0, 0, 0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
    nt = (S + pad) // W
    dev = q.device
    # [B, nt, g, r*W, d], rows ordered (r, query)
    q_t = q.view(B, nt, W, kvh, rep, d).permute(0, 1, 3, 4, 2, 5).contiguous().view(B, nt, kvh, rep * W, d)
    k_cur = k.view(B, nt, W, kvh, d).permute(0, 1, 3, 2, 4)  # [B, nt, g, W, d]
    v_cur = v.view(B, nt, W, kvh, d).permute(0, 1, 3, 2, 4)
    k_t = torch.cat([torch.cat([torch.zeros_like(k_cur[:, :1]), k_cur[:, :-1]], dim=1), k_cur], dim=3).contiguous()
    v_t = torch.cat([torch.cat([torch.zeros_like(v_cur[:, :1]), v_cur[:, :-1]], dim=1), v_cur], dim=3).contiguous()
    sc = torch.matmul(q_t, k_t.transpose(-1, -2)).float() * scale  # [B, nt, g, r*W, 2W]
    sc = sc.view(B, nt, kvh, rep, W, 2 * W)
    a = torch.arange(W, device=dev)
    t = torch.arange(nt, device=dev)[:, None, None]
    kidx = (t - 1) * W + torch.arange(2 * W, device=dev)[None, None, :]  # [nt, 1, 2W]
    rel = t * W + a[None, :, None] - kidx  # query index minus key index
    ok_c = (rel >= 0) & (rel < W) & (kidx >= 0)  # [nt, W, 2W]
    neg = torch.tensor(float("-inf"), device=dev)
    zero = torch.tensor(0.0, device=dev)
    sc = sc + torch.where(ok_c, zero, neg)[None, :, None, None]
    sink = sinks.float().view(kvh, rep, 1, 1)

    s = torch.cat([sc, sink.expand(B, nt, kvh, rep, W, 1)], dim=-1)
    p = torch.softmax(s, dim=-1)[..., : 2 * W].to(v.dtype).reshape(B, nt, kvh, rep * W, 2 * W)
    o = torch.matmul(p, v_t).view(B, nt, kvh, rep, W, d)

    if kc is not None:
        Wc = kc.shape[1]
        kp = frame[:, None] + torch.arange(Wc, device=dev)[None]  # [B, Wc] absolute key positions
        ok_x = (kp[:, None, :] < ctx[:, None, None]) & (kp[:, None, :] > ctx[:, None, None] + a[None, :, None] - W)
        kc_ = kc.permute(0, 2, 1, 3).contiguous()  # [B, g, Wc, d]
        vc_ = vc.permute(0, 2, 1, 3).contiguous()
        sx = torch.matmul(q_t[:, 0], kc_.transpose(-1, -2)).float() * scale  # [B, g, r*W, Wc]
        sx = sx.view(B, kvh, rep, W, Wc) + torch.where(ok_x, zero, neg)[:, None, None]
        s0 = torch.cat([sx, sc[:, 0], sink.expand(B, kvh, rep, W, 1)], dim=-1)
        p0 = torch.softmax(s0, dim=-1)
        px = p0[..., :Wc].to(vc.dtype).reshape(B, kvh, rep * W, Wc)
        pc = p0[..., Wc : Wc + 2 * W].to(v.dtype).reshape(B, kvh, rep * W, 2 * W)
        o0 = (torch.matmul(px, vc_) + torch.matmul(pc, v_t[:, 0])).view(B, kvh, rep, W, d)
        o = torch.cat([o0.unsqueeze(1), o[:, 1:]], dim=1)
    # [B, nt, g, r, W, d] -> [B, nt, W, g, r, d] -> [B, S, nh*d]
    return o.permute(0, 1, 4, 2, 3, 5).contiguous().view(B, S + pad, nh * d)[:, :S]


def swa_reference(q, k, v, sinks, scale, window, kc=None, vc=None, ctx=None, frame=None):
    """Brute-force fp32 twin of swa_band_attention, for tests."""
    B, S, nh, d = q.shape
    rep = nh // k.shape[2]
    outs = []
    for bi in range(B):
        c = int(ctx[bi]) if ctx is not None else 0
        keys, vals, pos = k[bi].float(), v[bi].float(), torch.arange(S) + c
        if kc is not None:
            kpos = int(frame[bi]) + torch.arange(kc.shape[1])
            keep = kpos < c
            keys = torch.cat([kc[bi].float()[keep], keys])
            vals = torch.cat([vc[bi].float()[keep], vals])
            pos = torch.cat([kpos[keep], pos])
        qp = torch.arange(S) + c
        allowed = (pos[None] <= qp[:, None]) & (pos[None] > qp[:, None] - window)
        kk = keys.repeat_interleave(rep, 1)
        vv = vals.repeat_interleave(rep, 1)
        s = torch.einsum("qhd,khd->hqk", q[bi].float(), kk) * scale
        s = s.masked_fill(~allowed[None], float("-inf"))
        s = torch.cat([s, sinks.float().view(nh, 1, 1).expand(nh, S, 1)], -1)
        p = torch.softmax(s, -1)[..., :-1]
        outs.append(torch.einsum("hqk,khd->qhd", p, vv).reshape(S, nh * d))
    return torch.stack(outs)


# --------------------------------------------------------------------------------------------------------------------
# Global attention prefill with context, in query tiles
# --------------------------------------------------------------------------------------------------------------------
GA_TILE_ELEMS = 1 << 30  # score elements per tile, heads included: about 2 GiB of bf16 scores


def ga_tile_rows(S: int, L: int, heads: int = 1, budget: int = GA_TILE_ELEMS) -> int:
    """Query rows per tile: the largest power of two (at least 128) with rows * L * heads <= budget.

    Heads count: budgeting rows * keys alone asked for 8.6 GiB tiles at 400K context with 8 heads per rank.
    """
    rows = 128
    while rows * 2 <= S and rows * 2 * L * heads <= budget:
        rows *= 2
    return S if S * L * heads <= budget else rows


def ga_tile_mask(rows: torch.Tensor, P: int, S: int, ctx: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
    """The runner's prefix-prefill mask (_set_attn_bias), for query rows `rows` only. True means masked.

    Keys are [P past slots | S chunk slots]. Past slot j is masked when j >= ctx[b]; chunk slot j when j > row
    (causal) or j >= seq[b] (padding). Returns bool [B, 1, len(rows), P + S].
    """
    B = ctx.shape[0]
    jp = torch.arange(P, device=rows.device)
    jc = torch.arange(S, device=rows.device)
    past = (jp.view(1, 1, -1) >= ctx.view(-1, 1, 1)).expand(B, rows.numel(), P)
    cur = (jc.view(1, 1, -1) > rows.view(1, -1, 1)) | (jc.view(1, 1, -1) >= seq.view(-1, 1, 1))
    return torch.cat([past, cur], dim=-1).view(B, 1, rows.numel(), P + S)


# --------------------------------------------------------------------------------------------------------------------
# RoPE table
# --------------------------------------------------------------------------------------------------------------------
def exact_cos_sin(n: int, rotary_dim: int, theta: float) -> torch.Tensor:
    """[n, rotary_dim] cos | sin table, computed on the CPU in float64.

    Gaudi fp32 cos and sin are wrong for arguments past about 1.5e5 radians, and the fastest rotary frequency's
    angle is the position itself, so a table built on the device is wrong from about 196K positions on.
    """
    inv = 1.0 / (float(theta) ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim))
    ang = torch.outer(torch.arange(n, dtype=torch.float64), inv)
    return torch.cat([ang.cos(), ang.sin()], dim=-1)
