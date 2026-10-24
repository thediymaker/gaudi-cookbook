# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the rel-bias helpers (ops.py). Need torch; skip without it.

- prompt_rel_bias against a direct per-element reference.
- Local-window prefill (window_context + prompt_rel_bias_kpos) gives the same attention output as the full
  context, which is what makes INKLING_LOCAL_WINDOW_PREFILL safe.
- The decode bias split into decode_rel_index + decode_rel_apply (cached per forward under
  INKLING_STEP_HOIST) is bit-exact with the one-pass computation.
"""

import pytest

torch = pytest.importorskip("torch")

from inkling_hpu import ops  # noqa: E402


def _reference_prompt_bias(rel, qpos, kpos, kvalid, rel_extent, window):
    """bias[b, h, i, j] by definition: rel[b, i, h, qpos - kpos] in range, -inf for future, out-of-window
    or invalid keys."""
    bsz, lq, n_heads, _ = rel.shape
    out = torch.zeros(bsz, n_heads, lq, kpos.shape[1])
    neg = torch.finfo(torch.float32).min
    for b in range(bsz):
        for i in range(lq):
            for j in range(kpos.shape[1]):
                d = int(qpos[b, i] - kpos[b, j])
                if d < 0 or not kvalid[b, j] or (window is not None and d > window - 1):
                    out[b, :, i, j] = neg
                elif d < rel_extent:
                    out[b, :, i, j] = rel[b, i, :, d]
    return out


def test_prompt_rel_bias_matches_definition():
    torch.manual_seed(0)
    bsz, n_heads, ext, bs = 2, 3, 12, 4
    for ctx_blocks, window in ((0, None), (0, 6), (2, None), (2, 6)):
        ctx_pad = ctx_blocks * bs
        lq = 7
        ctx = torch.tensor([ctx_pad, max(0, ctx_pad - 3)])
        seq_lens = torch.tensor([lq, lq - 2])
        pos = ctx.view(bsz, 1) + torch.arange(lq).view(1, lq)
        rel = torch.randn(bsz, lq, n_heads, ext)
        got = ops.prompt_rel_bias(rel, pos, pos, ctx_pad, ctx, ext, window, torch.float32, seq_lens)
        kpos = torch.cat([torch.arange(ctx_pad).view(1, -1).expand(bsz, -1), pos], dim=1)
        kvalid = torch.cat(
            [
                torch.arange(ctx_pad).view(1, -1) < ctx.view(bsz, 1),
                torch.arange(lq).view(1, -1) < seq_lens.view(bsz, 1),
            ],
            dim=1,
        )
        want = _reference_prompt_bias(rel, pos, kpos, kvalid, ext, window)
        assert torch.equal(got, want), (ctx_blocks, window)


def _attend(q, k, v, bias):  # q [B, Lq, H, D], k/v [B, KV, H, D], bias [B, H, Lq, KV]
    s = torch.einsum("bqhd,bkhd->bhqk", q, k) / q.shape[-1] + bias
    return torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, -1), v)


def test_local_window_prefill_equals_full_context():
    torch.manual_seed(0)
    for trial in range(40):
        bsz, n_heads, dim, bs = 3, 2, 8, 8
        window = [16, 32, 24][trial % 3]
        n_pool = 200
        kc, vc = torch.randn(n_pool + 1, bs, n_heads, dim), torch.randn(n_pool + 1, bs, n_heads, dim)
        kc[0] = 0  # null block
        vc[0] = 0
        ctx = torch.tensor([int(torch.randint(0, 90, ())) for _ in range(bsz)])
        if trial % 4 == 0:
            ctx[-1] = 0
        lq = 20
        qlen = torch.tensor([lq, int(torch.randint(1, lq + 1, ())), 0 if trial % 5 == 0 else lq])
        nblk = max(1, (int(ctx.max()) + bs - 1) // bs) + trial % 3  # bucket padding
        bl = torch.full((bsz, nblk), -1, dtype=torch.long)
        perm = torch.randperm(n_pool)[: bsz * nblk] + 1
        for b in range(bsz):
            n = (int(ctx[b]) + bs - 1) // bs
            bl[b, :n] = perm[b * nblk : b * nblk + n]
        q, kch, vch = (torch.randn(bsz, lq, n_heads, dim) for _ in range(3))
        pos = ctx.view(bsz, 1) + torch.arange(lq).view(1, lq)
        rel = torch.randn(bsz, lq, n_heads, window)

        ids_full = bl.clone()
        ids_full[ids_full < 0] = n_pool  # the runner's pad block
        kf, vf = kc[ids_full].view(bsz, -1, n_heads, dim), vc[ids_full].view(bsz, -1, n_heads, dim)
        bias_f = ops.prompt_rel_bias(rel, pos, pos, nblk * bs, ctx, window, window, torch.float32, qlen)
        out_f = _attend(q, torch.cat([kf, kch], 1), torch.cat([vf, vch], 1), bias_f)

        ids, kpos, valid = ops.window_context(bl.flatten().to(torch.int32), ctx, bsz, bs, window)
        kw = kc[ids.long()].view(bsz, -1, n_heads, dim)
        vw = vc[ids.long()].view(bsz, -1, n_heads, dim)
        bias_w = ops.prompt_rel_bias_kpos(rel, pos, kpos, valid, pos, window, window, torch.float32, qlen)
        out_w = _attend(q, torch.cat([kw, kch], 1), torch.cat([vw, vch], 1), bias_w)
        for b in range(bsz):
            n = int(qlen[b])
            if n:
                assert (out_f[b, :n] - out_w[b, :n]).abs().max() <= 1e-5, (trial, b)


def _one_pass_decode_bias(rel, positions, block_groups, block_size, rel_extent, window):
    """The decode bias computed in one pass, per block and slot, from the definition."""
    nb = block_groups.shape[0]
    n_heads = rel.shape[1]
    out = torch.zeros(nb, n_heads, block_size)
    for seq in range(rel.shape[0]):
        mine = [i for i in range(nb) if int(block_groups[i]) == seq]
        total = (int(positions[seq]) + block_size) // block_size
        for ordinal, blk in enumerate(mine):
            base = (total - len(mine) + ordinal) * block_size
            for j in range(block_size):
                d = int(positions[seq]) - (base + j)
                if window is not None and d >= window:
                    out[blk, :, j] = -1e4
                elif 0 <= d < rel_extent:
                    out[blk, :, j] = rel[seq, :, d]
    return out


def test_decode_bias_split_matches_one_pass():
    torch.manual_seed(0)
    for t in range(60):
        bsz = [1, 3, 5][t % 3]
        n_heads, bs = 4, 8
        ext = [16, 32][t % 2]
        win = [None, 16, 12][t % 3]
        pos = torch.randint(0, 120, (bsz,))
        groups = []
        for b in range(bsz):
            n = int((pos[b] + bs) // bs)
            groups += [b] * (n if t % 2 else min(n, 3))  # full list or window-truncated tail
        groups += [-1] * (t % 4)  # bucket padding
        bg = torch.tensor(groups)
        rel = torch.randn(bsz, n_heads, ext)
        got = ops.decode_rel_apply(rel, ops.decode_rel_index(pos, bg, bsz, bs, ext, win))
        real = bg >= 0
        assert torch.equal(got[real], _one_pass_decode_bias(rel, pos, bg, bs, ext, win)[real]), t


def test_query_sub_block_stays_under_budget():
    for seq_len, kv_len in ((2048, 2048), (2048, 131072), (2048, 262144), (64, 300000)):
        rows = ops.query_sub_block(seq_len, 1, 4, kv_len, 805306368)
        assert 32 <= rows <= seq_len and rows & (rows - 1) == 0
        assert rows == 32 or rows * 4 * kv_len * 2 <= 805306368


def test_log_scaling_tau():
    tau = ops.log_scaling_tau(torch.tensor([0, 127999, 255999]), 128000, 0.1)
    assert tau[0] == 1.0 and tau[1] == 1.0
    assert abs(float(tau[2]) - (1 + 0.1 * torch.log(torch.tensor(2.0)))) < 1e-6
