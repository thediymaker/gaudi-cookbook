# SPDX-License-Identifier: Apache-2.0
"""long_context.py on CPU: the compact key row plus the per-tile causal mask reproduce vllm-gaudi's dense prompt
bias (HPUAttentionMetadataProcessor._set_attn_bias) row for row and bit for bit, for random bucketed contexts,
padded batch rows and every tile size; choose_tile keeps the FusedSDPA workspace inside its budget."""

from __future__ import annotations

import math
import random

import pytest

torch = pytest.importorskip("torch")

from muse_spark_hpu import long_context as LC  # noqa: E402


def dense_bias(context_lens_t, seq_lens_t, ctx_max, seq_len, dtype):
    """vllm-gaudi v0.30.0's dense prompt bias with cached context, as _set_attn_bias builds it."""
    bs = context_lens_t.shape[0]
    past = torch.arange(ctx_max, dtype=torch.int32).view(1, -1).expand(bs, -1).ge(context_lens_t.view(-1, 1))
    past = past.view(bs, 1, -1).expand(bs, seq_len, -1).view(bs, 1, seq_len, -1)
    pad = torch.arange(seq_len, dtype=torch.int32).view(1, seq_len).ge(seq_lens_t.unsqueeze(-1)).view(bs, 1, 1, seq_len)
    causal = torch.triu(torch.ones((bs, 1, seq_len, seq_len), dtype=torch.bool), diagonal=1)
    mask = torch.cat((past, causal.logical_or(pad)), dim=-1)
    return torch.zeros_like(mask, dtype=dtype).masked_fill_(mask, -math.inf)


def same(a, b):
    return (
        a.shape == b.shape
        and torch.equal(torch.isinf(a), torch.isinf(b))
        and torch.equal(a.nan_to_num(0, 0, 0), b.nan_to_num(0, 0, 0))
    )


def test_tiles_match_dense_bias():
    rng = random.Random(0)
    checked = 0
    for _ in range(60):
        bs = rng.choice([1, 1, 2, 3])
        ctx_max = rng.choice([1, 2, 7, 33, 64]) * 128
        seq_len = rng.choice([128, 256, 512, 1024])
        ctx = torch.tensor([rng.randint(0, ctx_max) for _ in range(bs)], dtype=torch.int32)
        seql = torch.tensor([rng.randint(1, seq_len) for _ in range(bs)], dtype=torch.int32)
        dense = dense_bias(ctx, seql, ctx_max, seq_len, torch.bfloat16)
        keyrow = LC.compact_bias(ctx, seql, ctx_max, seq_len, torch.bfloat16, "cpu")
        assert keyrow.shape == (bs, 1, 1, ctx_max + seq_len)
        for tile in (8, 16, 64, seq_len):
            rows = torch.cat([LC.tile_bias(keyrow, ctx_max, a, tile) for a in range(0, seq_len, tile)], dim=-2)
            assert same(rows, dense), (bs, ctx_max, seq_len, tile)
            checked += 1
    assert checked == 240


@pytest.mark.parametrize("q", [128, 1024, 4096])
@pytest.mark.parametrize("kv", [8192, 135168, 266240, 528384, 1052672])
def test_choose_tile_budget(q, kv):
    t = LC.choose_tile(q, 1, 16, kv, True)
    workspace = 4 * 16 * t * kv
    assert q % t == 0 and t & (t - 1) == 0
    assert workspace <= LC.WS_GIB * 2**30 or t <= LC.MIN_TILE


def test_choose_tile_served_shapes():
    # 16 query heads per card, 4096-token chunks: 128 rows at 128K keys, 16 at 1M
    assert LC.choose_tile(4096, 1, 16, 135168) == 128
    assert LC.choose_tile(4096, 1, 16, 1052672) == 16
