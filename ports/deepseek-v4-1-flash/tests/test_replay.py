# SPDX-License-Identifier: Apache-2.0
"""CPU test of decoder replay (replay.py, DSV41_REPLAY_KEEP) on a tiny random-weight model.

Mask mode runs every token through every layer with the same decoder window floors, so a kept token's result must
not depend on whether the others were dropped: compact must equal mask (fp32; in bf16 the compact batch's different
GEMM shapes round differently and the tiny model amplifies that). Scenarios: three chunks whose last one is shorter
than KEEP (its floor reaches into the previous chunk's tail, which that chunk replayed itself), and a prefix-cache
hit (B reuses A's first blocks; the hit ends at least KEEP tokens before B's end). Every compact run reads 0 masked
rows (tag check in count mode): no read below a floor, no read of a row nobody wrote. Control: without the floors,
compact mode reads window rows it never wrote.
"""

import pytest

torch = pytest.importorskip("torch")

import tiny  # noqa: E402
from dsv41_hpu import csa, paged, replay  # noqa: E402

BS = 16
KEEP = 24  # > the tiny model's 16-token window, as 140 > 128 in the real one


@pytest.fixture(scope="module")
def model():
    cfg = tiny.config()
    st, head, hasher = tiny.build(cfg, seed=4)
    return cfg, st.float(), head, hasher


@pytest.fixture(autouse=True)
def _modes(monkeypatch):
    monkeypatch.setattr(csa, "DET_TOPK", True)
    monkeypatch.setattr(paged, "TAG_MODE", "count")


def _seq(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, tiny.VOCAB, (n,), generator=g).tolist()


def _pool(cfg, seed=5):
    return tiny.Pool(cfg, n_blocks=96, block_size=BS, dtype=torch.float32, seed=seed, max_blocks_per_req=40)


@pytest.mark.parametrize("chunks", [[(0, 100), (100, 230), (230, 240)], [(0, 181)]])
def test_compact_equals_mask(model, chunks):
    cfg, st, head, hasher = model
    seq, dec = _seq(chunks[-1][1], 1), _seq(5, 2)
    res = {}
    for compact in (False, True):
        paged.reset_tag_violations()
        r = tiny.Runner(st, head, hasher, cfg, _pool(cfg), replay_keep=KEEP, compact=compact, pad=16)
        res[compact] = r.request("a", seq, chunks, dec)
        if compact:
            assert paged.tag_violations()["rows"] == 0
    assert float((res[True] - res[False]).abs().max()) < 1e-3 * float(res[False].abs().max())


def test_prefix_hit_matches_lone_run(model):
    cfg, st, head, hasher = model
    k = 4 * BS
    a = _seq(150, 3)
    b = a[:k] + _seq(170 - k, 4)
    da, db = _seq(4, 5), _seq(4, 6)
    assert k <= len(b) - KEEP, "the hit must respect the replay cap"
    lone_b = tiny.Runner(st, head, hasher, cfg, _pool(cfg, 7), replay_keep=KEEP).request("b", b, [(0, k), (k, 170)], db)
    paged.reset_tag_violations()
    r = tiny.Runner(st, head, hasher, cfg, _pool(cfg, 8), replay_keep=KEEP)
    r.request("a", a, [(0, k), (k, 150)], da)
    r.pool.admit("b")
    r.pool.share("b", "a", 4)
    shared_b = r.request("b", b, [(k, 170)], db, admit=False)
    assert paged.tag_violations()["rows"] == 0
    assert float((shared_b - lone_b[1:]).abs().max()) < 1e-3 * float(lone_b.abs().max())


def test_control_without_floors_reads_unwritten_rows(model, monkeypatch):
    cfg, st, head, hasher = model
    orig = replay._floored_cls

    def no_floor():
        _, read_only = orig()
        return type("NoFloor", (), {}), read_only

    monkeypatch.setattr(replay, "_floored_cls", no_floor)
    paged.reset_tag_violations()
    r = tiny.Runner(st, head, hasher, cfg, _pool(cfg), replay_keep=KEEP, pad=16)
    r.request("a", _seq(240, 1), [(0, 100), (100, 230), (230, 240)], _seq(3, 2))
    assert paged.tag_violations()["rows"] > 0
