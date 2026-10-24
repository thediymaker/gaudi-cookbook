# SPDX-License-Identifier: Apache-2.0
"""Rel-bias and window helpers for Inkling attention. Plain torch, no vLLM imports, so the CPU tests can
run them directly.

Inkling has no RoPE. Each head adds a learned relative-position logit to its attention scores:

    logit(i, j, h) = q[i, h] . k[j, h] / head_dim + rel[i, h, i - j]   for 0 <= i - j < rel_extent

`rel` comes from the R branch of the QKVR projection (see model.py). On HPU the term is passed to the
attention kernels as an additive bias: a [B, H, Lq, KV] mask for FusedSDPA in prefill, and a per-block
[NB, H, block_size] bias in the ALiBi slot of paged decode.
"""

import os

import torch

# INKLING_STEP_HOIST=1: tensors that depend only on the step's metadata (conv-state slots, decode rel-bias
# indices, the prefill window context) are computed once per forward instead of once per layer. Off
# recomputes them in every layer; the results are identical.
STEP_HOIST = os.environ.get("INKLING_STEP_HOIST", "0") == "1"
STEP_CACHE: dict = {}


def step_cached(key, fn):
    """fn() once per model forward when STEP_HOIST is on; the model clears STEP_CACHE at each forward."""
    if not STEP_HOIST:
        return fn()
    if key not in STEP_CACHE:
        STEP_CACHE[key] = fn()
    return STEP_CACHE[key]


def log_scaling_tau(positions: torch.Tensor, n_floor: int, alpha: float) -> torch.Tensor:
    """Global layers scale q and rel by 1 + alpha * log(max((pos + 1) / n_floor, 1))."""
    effective_n = (positions + 1).to(torch.float32)
    return 1.0 + alpha * torch.log(torch.clamp(effective_n / float(n_floor), min=1.0))


def _bias_from_kpos(rel, qpos, kpos, rel_extent, window, dtype, masked_extra):
    """Shared core: gather rel at distance i - j, zero it out of range, mask future, out-of-window and
    `masked_extra` keys with the dtype's minimum."""
    n_heads = rel.shape[2]
    dist = qpos.to(torch.long).unsqueeze(-1) - kpos.unsqueeze(-2)  # [B, Lq, KV] = i - j
    in_rng = (dist >= 0) & (dist < rel_extent)
    idx = dist.clamp(0, rel_extent - 1)
    bias = rel.permute(0, 2, 1, 3).gather(3, idx.unsqueeze(1).expand(-1, n_heads, -1, -1))  # [B, H, Lq, KV]
    bias = torch.where(in_rng.unsqueeze(1), bias, torch.zeros_like(bias))
    masked = dist < 0
    if window is not None:
        masked = masked | (dist > (window - 1))
    masked = masked | masked_extra
    return bias.to(dtype).masked_fill(masked.unsqueeze(1), torch.finfo(dtype).min)


def prompt_rel_bias(
    rel: torch.Tensor,  # [B, Lq, H, E] rel logits of the query rows (tau applied)
    qpos: torch.Tensor,  # [B, Lq] absolute positions of the query rows
    kpos_chunk: torch.Tensor,  # [B, L] absolute positions of the whole chunk
    ctx_pad: int,  # padded context width, 0 when there are no context blocks
    context_lens: torch.Tensor | None,  # [B] valid context length
    rel_extent: int,
    window: int | None,
    dtype: torch.dtype,
    seq_lens: torch.Tensor | None = None,  # [B] valid chunk length
) -> torch.Tensor:
    """Additive FusedSDPA bias [B, H, Lq, ctx_pad + L].

    Keys follow the layout of vllm_gaudi's ops._get_context: `ctx_pad` context keys first (absolute
    position = column index, valid below context_lens), then the chunk's keys (valid below seq_lens). The
    bias carries the causal, window and pad masks, so it replaces valid_seq_lengths, which FusedSDPA
    rejects with an explicit mask. `rel`/`qpos` may be a sub-block of the chunk's rows."""
    bsz = rel.shape[0]
    device = rel.device
    if ctx_pad:
        ctx_pos = torch.arange(ctx_pad, device=device).view(1, ctx_pad).expand(bsz, -1)
        kpos = torch.cat([ctx_pos, kpos_chunk.to(torch.long)], dim=1)
    else:
        kpos = kpos_chunk.to(torch.long)
    kv_len = kpos.shape[1]
    # Pad keys are masked by column index: positions repeat inside a padded bucket.
    col = torch.arange(kv_len, device=device).view(1, 1, kv_len)
    extra = torch.zeros(1, 1, kv_len, dtype=torch.bool, device=device)
    if seq_lens is not None:
        chunk_col = col - ctx_pad
        extra = extra | ((chunk_col >= 0) & (chunk_col >= seq_lens.view(bsz, 1, 1)))
    if ctx_pad:
        assert context_lens is not None
        extra = extra | ((col < ctx_pad) & (col >= context_lens.view(bsz, 1, 1)))
    return _bias_from_kpos(rel, qpos, kpos, rel_extent, window, dtype, extra)


def window_context(block_list: torch.Tensor, context_lens: torch.Tensor, bsz: int, block_size: int, window: int):
    """The last context blocks a window-`window` query at the chunk start can reach:
    ceil(window / block_size) + 1 of them (the +1 covers an unaligned chunk start).

    -> (block ids [B * W], absolute key positions [B, W * bs], valid [B, W * bs]). Columns before the start
    of the context fetch block 0 (vLLM's null block) and are masked."""
    nblk = block_list.numel() // bsz
    n_win = min(nblk, -(-window // block_size) + 1)
    dev = block_list.device
    bl = block_list.view(bsz, nblk).to(torch.long)
    ctx = context_lens[:bsz].to(torch.long)
    n_ctx = torch.div(ctx + block_size - 1, block_size, rounding_mode="floor")
    cols = n_ctx.view(bsz, 1) - n_win + torch.arange(n_win, device=dev).view(1, n_win)  # logical block index
    ids = bl.gather(1, cols.clamp(0, nblk - 1))
    ids = torch.where(cols >= 0, ids, torch.zeros_like(ids))
    kpos = (cols.unsqueeze(-1) * block_size + torch.arange(block_size, device=dev).view(1, 1, block_size)).view(bsz, -1)
    valid = (kpos >= 0) & (kpos < ctx.view(bsz, 1))
    return ids.view(-1).to(block_list.dtype), kpos, valid


def prompt_rel_bias_kpos(
    rel: torch.Tensor,  # [B, Lq, H, E]
    qpos: torch.Tensor,  # [B, Lq]
    ctx_kpos: torch.Tensor,  # [B, C] absolute positions of the context keys
    ctx_valid: torch.Tensor,  # [B, C]
    kpos_chunk: torch.Tensor,  # [B, L]
    rel_extent: int,
    window: int | None,
    dtype: torch.dtype,
    seq_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """prompt_rel_bias for an explicit context key set (the window context): keys are
    [context keys at ctx_kpos | chunk], same bias and masks."""
    bsz = rel.shape[0]
    n_ctx = ctx_kpos.shape[1]
    kpos = torch.cat([ctx_kpos.to(torch.long), kpos_chunk.to(torch.long)], dim=1)
    kv_len = kpos.shape[1]
    col = torch.arange(kv_len, device=rel.device).view(1, 1, kv_len)
    extra = torch.cat([~ctx_valid, torch.zeros_like(kpos_chunk, dtype=torch.bool)], dim=1).unsqueeze(1)
    if seq_lens is not None:
        chunk_col = col - n_ctx
        extra = extra | ((chunk_col >= 0) & (chunk_col >= seq_lens.view(bsz, 1, 1)))
    return _bias_from_kpos(rel, qpos, kpos, rel_extent, window, dtype, extra)


def decode_rel_index(
    positions: torch.Tensor,  # [B] absolute position of each decode token
    block_groups: torch.Tensor,  # [NB] block -> sequence index (pad blocks: -1)
    bsz: int,
    block_size: int,
    rel_extent: int,
    window: int | None,
):
    """The metadata part of the decode bias, the same for every layer of a forward.

    Key positions come from each block's ordinal in its sequence's (possibly window-truncated) block list:
    base(blk) = (total_blocks(seq) - blocks_in_list(seq) + ordinal) * block_size, which is ordinal *
    block_size for full lists and the tail window otherwise. Per-sequence first index and count come from a
    [B, NB] membership mask (scatter_reduce amin has no HPU kernel). Membership tests the raw group ids so
    pad blocks (-1) match no sequence."""
    nb = block_groups.shape[0]
    device = block_groups.device
    grp = block_groups.clamp(min=0).to(torch.long)
    ar = torch.arange(nb, device=device)
    member = block_groups.view(1, nb).to(torch.long) == torch.arange(bsz, device=device).view(bsz, 1)
    group_start = torch.where(member, ar.view(1, nb), torch.full_like(ar, nb).view(1, nb)).amin(dim=1)
    ordinal = ar - group_start[grp]
    counts = member.sum(dim=1).to(torch.long)
    seq_pos = positions.to(torch.long)
    total_blocks = torch.div(seq_pos + block_size, block_size, rounding_mode="floor")  # ceil((pos + 1) / bs)
    kbase = (total_blocks[grp] - counts[grp] + ordinal) * block_size
    j = torch.arange(block_size, device=device)
    dist = seq_pos[grp].unsqueeze(-1) - (kbase.unsqueeze(-1) + j)  # [NB, bs]
    in_rng = (dist >= 0) & (dist < rel_extent)
    idx = dist.clamp(0, rel_extent - 1)
    out_of_window = (dist >= window).unsqueeze(1) if window is not None else None
    return grp, idx, in_rng, out_of_window


def decode_rel_apply(rel: torch.Tensor, index) -> torch.Tensor:
    """The per-layer part: paged-decode position bias [NB, H, bs] from this layer's rel [B, H, E]. Keys
    outside the window get -1e4 (future and pad keys are already masked by the runner's block bias)."""
    grp, idx, in_rng, out_of_window = index
    n_heads = rel.shape[1]
    out = rel[grp].gather(2, idx.unsqueeze(1).expand(-1, n_heads, -1))
    out = torch.where(in_rng.unsqueeze(1), out, torch.zeros_like(out))
    if out_of_window is not None:
        out = torch.where(out_of_window, torch.full_like(out, -1e4), out)
    return out


def query_sub_block(seq_len: int, batch_size: int, n_heads: int, kv_len: int, budget_bytes: int) -> int:
    """Query rows per prefill sub-block so the bf16 bias [B, H, rows, kv] stays under budget_bytes: the
    largest power of two that fits, at least 32, at most seq_len."""
    per_row = batch_size * n_heads * kv_len * 2
    return max(32, min(seq_len, 1 << max(5, (budget_bytes // max(per_row, 1)).bit_length() - 1)))
