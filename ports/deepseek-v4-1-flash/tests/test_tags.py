# SPDX-License-Identifier: Apache-2.0
"""CPU test of the cache ownership tags (paged.py) on a tiny random-weight model.

1. Block reuse: request A fills blocks and finishes; request B gets them back as fresh allocations and computes
   exactly what it computes on an untouched pool, reading 0 masked rows (in normal operation the tags never fire).
2. Prefix-cache hit: B receives A's first blocks as hits; their allocation numbers are kept, so B reads A's rows
   (0 masked) and computes what it computes alone.
3. Control: the same hit with the blocks numbered as fresh allocations (what a stale block table or a read before
   the write would look like): every row of the shared blocks is refused, the count is non-zero and B's result
   changes. A gate that cannot fail proves nothing.
"""

import pytest

torch = pytest.importorskip("torch")

import tiny  # noqa: E402
from dsv41_hpu import csa, paged  # noqa: E402

BS = 16


@pytest.fixture(scope="module")
def model():
    cfg = tiny.config()
    st, head, hasher = tiny.build(cfg, seed=6)
    return cfg, st.float(), head, hasher


@pytest.fixture(autouse=True)
def _modes(monkeypatch):
    monkeypatch.setattr(csa, "DET_TOPK", True)
    monkeypatch.setattr(paged, "TAG_MODE", "count")


def _seq(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, tiny.VOCAB, (n,), generator=g).tolist()


def _pool(cfg, seed=2):
    return tiny.Pool(cfg, n_blocks=24, block_size=BS, dtype=torch.float32, seed=seed, max_blocks_per_req=24)


def test_block_reuse(model):
    cfg, st, head, hasher = model
    b, db = _seq(120, 2), _seq(3, 3)
    ref = tiny.Runner(st, head, hasher, cfg, _pool(cfg)).request("b", b, [(0, 60), (60, 120)], db)
    pool = _pool(cfg)
    r = tiny.Runner(st, head, hasher, cfg, pool)
    r.request("a", _seq(200, 1), [(0, 200)], [])
    pool.release("a")
    pool.free.sort(reverse=True)  # B takes A's blocks, in another order than A had them
    paged.reset_tag_violations()
    got = r.request("b", b, [(0, 60), (60, 120)], db)
    assert paged.tag_violations()["rows"] == 0
    assert torch.equal(ref, got)


def _shared(model, fresh):
    cfg, st, head, hasher = model
    a = _seq(100, 4)
    b = a[:64] + _seq(56, 5)
    db = _seq(3, 6)
    lone = tiny.Runner(st, head, hasher, cfg, _pool(cfg, 3)).request("b", b, [(0, 64), (64, 120)], db)[1:]
    pool = _pool(cfg, 4)
    r = tiny.Runner(st, head, hasher, cfg, pool)
    r.request("a", a, [(0, 64), (64, 100)], [])  # the shared blocks computed in the same chunk shape as lone B
    pool.admit("b")
    pool.share("b", "a", 4)
    if fresh:
        for blk in pool.tables["b"]:
            pool.epoch[blk] = pool.next_epoch
            pool.next_epoch += 1
    paged.reset_tag_violations()
    got = r.request("b", b, [(64, 120)], db, admit=False)
    return lone, got, paged.tag_violations()["rows"]


def test_prefix_hit_reads_sharer_rows(model):
    lone, got, masked = _shared(model, fresh=False)
    assert masked == 0
    assert float((lone - got).abs().max()) < 1e-3 * float(lone.abs().max())


def test_control_fresh_numbers_refuse_shared_rows(model):
    lone, got, masked = _shared(model, fresh=True)
    assert masked > 0
    assert float((lone - got).abs().max()) > 1e-2 * float(lone.abs().max())
