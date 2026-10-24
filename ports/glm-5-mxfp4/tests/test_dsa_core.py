# SPDX-License-Identifier: Apache-2.0
"""glm5_hpu.dsa core on CPU against brute force: top-k selection, bisection thresholds, bit packing, compaction."""

import pytest

torch = pytest.importorskip("torch")

from glm5_hpu import dsa  # noqa: E402


def test_two_pass_topk_is_exact():
    g = torch.Generator().manual_seed(0)
    s = torch.randn(3, 70000, generator=g)
    k = 2048
    ref = torch.topk(s, k, dim=-1)
    assert torch.equal(dsa.kth_largest(s, k), ref.values[:, -1:])
    v, i = dsa.topk_indices(s, k)
    assert torch.equal(v, ref.values)
    assert torch.equal(torch.sort(i, dim=-1).values, torch.sort(ref.indices, dim=-1).values)


def test_bisect_threshold_selects_at_least_k_and_keeps_short_rows():
    g = torch.Generator().manual_seed(1)
    acc = torch.randn(4, 5000, generator=g)
    ok = torch.ones_like(acc, dtype=torch.bool)
    ok[3, 100:] = False  # row 3 has 100 valid keys: keep them all
    acc = torch.where(ok, acc, -float("inf"))
    k = 256
    thr = dsa.bisect_threshold(acc, ok, k, iters=40)
    sel = (acc >= thr) & ok
    for r in range(3):
        assert int(sel[r].sum()) == k  # 40 halvings separate distinct random floats
        assert torch.equal(sel[r], acc[r] >= torch.topk(acc[r], k).values[-1])
    assert int(sel[3].sum()) == 100


def test_prefill_visible_layout():
    vis = dsa.prefill_visible(0, 4, kv=8, cb=4, ctx_len=3, seq_len=4, device="cpu")
    # context slots 0..2 valid (3 is padding), then the chunk causally
    want = torch.tensor(
        [
            [1, 1, 1, 0, 1, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 0, 0],
            [1, 1, 1, 0, 1, 1, 1, 0],
            [1, 1, 1, 0, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(vis, want)


def test_pack_unpack_bits_roundtrip():
    g = torch.Generator().manual_seed(2)
    sel = torch.rand(5, 4096, generator=g) > 0.7
    assert torch.equal(dsa.unpack_bits(dsa.pack_bits(sel), 4096), sel)


def test_union_bucket():
    assert dsa.union_bucket(1, 2048) == 2048
    assert dsa.union_bucket(3000, 2048) == 3072
    assert dsa.union_bucket(8192, 2048) == 8192
    assert dsa.union_bucket(8193, 2048) == 16384


def test_compact_gathers_union_in_key_order():
    sel = torch.zeros(4, 16, dtype=torch.bool)
    sel[0, [1, 5]] = True
    sel[1, [5, 9]] = True
    sel[2, [0]] = True
    sel[3, [15]] = True
    u = sel.view(2, 2, 16).any(1)
    cl, bias = dsa.compact(sel, u, rr=2, ucap=4, dtype=torch.float32)
    assert cl[0, :3].tolist() == [1, 5, 9] and cl[1, :2].tolist() == [0, 15]
    keep = bias == 0
    assert keep[0].tolist() == [True, True, False, False]
    assert keep[1].tolist() == [False, True, True, False]
    assert keep[2].tolist() == [True, False, False, False]
    assert keep[3].tolist() == [False, True, False, False]


def test_decode_mask_bias_keeps_topk_per_sequence():
    g = torch.Generator().manual_seed(3)
    bs, hi, d, t = 2, 4, 8, 16
    nb = 8
    q = torch.randn(bs, hi, d, generator=g)
    w = torch.rand(bs, hi, generator=g)
    kb = torch.randn(nb, t, d, generator=g)
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, -1])
    bias = torch.zeros(nb, t)
    out = dsa.decode_mask_bias(q, w, kb, groups, bias, topk=20)
    kept = out == 0
    assert int(kept[:4].sum()) == 20 and int(kept[4:7].sum()) == 20
    assert bool(kept[7].all())  # padding block keeps its bias
