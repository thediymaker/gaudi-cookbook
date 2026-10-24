# SPDX-License-Identifier: Apache-2.0
"""CPU checks of mimo_v2_hpu.swa_kv: the per-group slot and window builders against brute force."""

import random
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mimo_v2_hpu"))
from mimo_v2_hpu import swa_kv  # noqa: E402

BS, WIN = 128, 2  # block 128, window 128: window // block + 1 blocks


def cdiv(a, b):
    return -(-a // b)


def ref_prefill(rows, row_idx, ctx, q, target_bs, target_seq):
    slots = [[-1] * target_seq for _ in range(target_bs)]
    win = [[-1] * WIN for _ in range(target_bs)]
    for i, (r, c, n) in enumerate(zip(row_idx, ctx, q, strict=True)):
        blocks = [0] * cdiv(c + n, BS) if r is None else [int(x) for x in rows[r][: cdiv(c + n, BS)]]
        for j in range(n):
            slots[i][j] = blocks[(c + j) // BS] * BS + (c + j) % BS
        w = blocks[: cdiv(c, BS)][-WIN:]
        win[i][: len(w)] = w
    return slots, win


def test_builders_match_brute_force():
    rng = random.Random(1)
    R, M = 8, 40
    for _ in range(300):
        rows = torch.tensor([[rng.randrange(1, 5000) for _ in range(M)] for _ in range(R)], dtype=torch.int32)
        n = rng.randrange(1, 4)
        target_bs = n + rng.randrange(0, 3)
        ctx, q, row_idx = [], [], []
        for _ in range(n):
            c = rng.choice([0, 1, 127, 128, 129, 255, 256, 257, rng.randrange(0, 3000)])
            qq = rng.choice([1, 2, 127, 128, 129, rng.randrange(1, 900)])
            if cdiv(c + qq, BS) > M:
                c, qq = 100, 50
            ctx.append(c)
            q.append(qq)
            row_idx.append(None if rng.random() < 0.1 else rng.randrange(R))
        target_seq = cdiv(max(q), BS) * BS + rng.choice([0, 128])
        slots, win = swa_kv.prefill_group_tensors(rows, row_idx, ctx, q, target_bs, target_seq, BS, WIN, True)
        rs, rw = ref_prefill(rows.tolist(), row_idx, ctx, q, target_bs, target_seq)
        assert slots.dtype == torch.int64 and slots.tolist() == rs
        assert win.dtype == torch.int32 and win.tolist() == rw

        nd = rng.randrange(1, R + 1)
        padded = nd + rng.randrange(0, 4)
        pos = [rng.choice([0, 1, 127, 128, 129, 255, 256, rng.randrange(0, M * BS - 1)]) for _ in range(nd)]
        index = torch.tensor([[p] for p in pos], dtype=torch.int64)
        base = torch.tensor([[rng.randrange(-200, -100)] for _ in range(padded)], dtype=torch.int64)
        nblk = [cdiv(p + 1, BS) for p in pos]
        sm, tabs = swa_kv.decode_group_tensors(rows, nd, index, base, nblk, BS, WIN)
        want = [[int(rows[i][p // BS]) * BS + p % BS] for i, p in enumerate(pos)] + base.tolist()[nd:]
        assert sm.tolist() == want
        assert tabs == [[int(x) for x in rows[i][:n]][-WIN:] for i, n in enumerate(nblk)]


def test_window_must_be_whole_blocks():
    with pytest.raises(ValueError):
        swa_kv.SwaKvGroups(window=100, block_size=128)
    assert swa_kv.SwaKvGroups(window=128, block_size=128).win_blocks == 2


def test_spec_conversion():
    kv = pytest.importorskip("vllm.v1.kv_cache_interface")
    spec = kv.FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=192, dtype=torch.bfloat16)
    specs = {f"l{i}": spec for i in range(6)}
    windows = {f"l{i}": (None if i % 3 == 0 else 128) for i in range(6)}
    out = swa_kv.convert_spec_dict(specs, windows)
    assert [type(s).__name__ for s in out.values()].count("SlidingWindowSpec") == 4
    with pytest.raises(ValueError):
        swa_kv.convert_spec_dict(specs, dict.fromkeys(specs, 128))
