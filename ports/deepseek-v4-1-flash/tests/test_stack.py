# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the decoder stack on a tiny random-weight model.

1. Chunking: a prompt prefilled in several chunks and then decoded gives the same logits as one chunk (every token
   writes its rows, every query reads what it may see, whatever the chunk boundaries).
2. Switches: every speed-up switch the recipe turns on computes the same function as its reference path (within
   fp32 summation-order noise). DSV41_TOPK_SELECT=thresh is checked in test_topk.py: it relies on bf16 scores.
The model runs in fp32: in bf16 the tiny random model amplifies last-bit GEMM differences between chunk shapes into
top-k flips, which says nothing about the code.
"""

import pytest

torch = pytest.importorskip("torch")

import tiny  # noqa: E402
from dsv41_hpu import csa, mhc, numerics, paged  # noqa: E402

BS = 16


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = tiny.config()
    st, head, hasher = tiny.build(cfg, seed=3)
    return cfg, st.float(), head, hasher


def _run(model, chunks, n_dec=4, seed=5, pad=0):
    cfg, st, head, hasher = model
    g = torch.Generator().manual_seed(seed)
    seq = torch.randint(3, tiny.VOCAB, (chunks[-1][1],), generator=g).tolist()
    dec = torch.randint(3, tiny.VOCAB, (n_dec,), generator=g).tolist()
    pool = tiny.Pool(cfg, n_blocks=64, block_size=BS, dtype=torch.float32, seed=1, max_blocks_per_req=32)
    paged.reset_tag_violations()
    out = tiny.Runner(st, head, hasher, cfg, pool, pad=pad).request("r", seq, chunks, dec)
    return out[len(chunks) - 1 :]  # the last chunk's sampled row, then the decode steps


def test_chunking_invariance(model, monkeypatch):
    monkeypatch.setattr(csa, "DET_TOPK", True)
    one = _run(model, [(0, 200)])
    many = _run(model, [(0, 37), (37, 130), (130, 200)], pad=16)
    assert torch.equal(one.argmax(-1), many.argmax(-1))
    assert float((one - many).abs().max()) < 2e-3 * float(one.abs().max())


@pytest.mark.parametrize(
    "switches",
    [
        {"numerics.LEAN": True, "paged.LEAN": True, "csa.LEAN": True},
        {"numerics.FASTBITS": True},
        {"numerics.ROPE_FUSED": True},
        {"mhc.SINKHORN_SLICED": True},
        {"mhc.MHC_TPC_DOT": True},
        {"csa.PREFILL_ATTN": True},
        {"csa.INDEX_BF16_DOT": True},
        {"csa.INDEX_DECODE": True},
        {"csa.WOA_LINEAR": True},
    ],
    ids=lambda d: ",".join(d),
)
def test_switch_matches_reference(model, monkeypatch, switches):
    monkeypatch.setattr(csa, "DET_TOPK", True)
    chunks = [(0, 90), (90, 170)]
    ref = _run(model, chunks)
    mods = {"numerics": numerics, "paged": paged, "csa": csa, "mhc": mhc}
    for k, v in switches.items():
        m, name = k.split(".")
        monkeypatch.setattr(mods[m], name, v)
    got = _run(model, chunks)
    assert torch.equal(ref.argmax(-1), got.argmax(-1))
    # DSV41_PREFILL_ATTN changes the fp32 summation order of the attention, which the fake-FP4 caches of the tiny
    # random model amplify; the others leave the logits within rounding noise
    tol = 0.15 if "csa.PREFILL_ATTN" in switches else 2e-3
    assert float((ref - got).abs().max()) < tol * float(ref.abs().max())


@pytest.mark.parametrize("hoist", [False, True])
def test_decode_groups_match_stack(model, monkeypatch, hoist):
    """Decode through the layer-group steps the HPU graphs wrap (DSV41_LAYER_GROUP, DSV41_DECODE_HOIST) equals the
    plain stack forward."""
    from dsv41_hpu.core import BlockStep, GroupStep, run_groups
    from dsv41_hpu.mhc import hc_pre, identity_pre
    from dsv41_hpu.numerics import rms_norm

    monkeypatch.setattr(csa, "DET_TOPK", True)
    monkeypatch.setattr(paged, "HOIST", hoist)
    import dsv41_hpu.core as core

    monkeypatch.setattr(core, "HOIST", hoist)
    cfg, st, head, hasher = model
    g = torch.Generator().manual_seed(9)
    seq = torch.randint(3, tiny.VOCAB, (150,), generator=g).tolist()
    outs = []
    for grouped in (False, True):
        pool = tiny.Pool(cfg, n_blocks=64, block_size=BS, dtype=torch.float32, seed=1, max_blocks_per_req=32)
        r = tiny.Runner(st, head, hasher, cfg, pool)
        r.request("r", seq[:140], [(0, 140)], [])
        pos = torch.arange(140, 150).view(1, -1)
        step_out = []
        for i in range(10):
            p = pos[:, i : i + 1]
            pool.ensure("r", int(p) + 1)
            ctx = pool.ctx(["r"], p)
            rows = tiny.engram_rows(hasher, cfg, [seq[: int(p) + 1]], p)
            ids = torch.tensor([[seq[int(p)]]])
            with torch.inference_mode():
                if grouped:
                    steps = [BlockStep(layer, pool.caches, BS) for layer in st.layers]
                    src = [rl.layer_id for rl in st.roles if rl.is_index_source]
                    groups = [GroupStep(steps[j : j + 10], src, cfg.engram_layer_ids) for j in range(0, 40, 10)]
                    e = st.embed(ids)
                    h = e.unsqueeze(-2).repeat(1, 1, cfg.hc_mult, 1)
                    h, pre = run_groups(groups, h, identity_pre(h, cfg.hc_mult), ctx, rows)
                    out = rms_norm(hc_pre(h, pre), st.norm, cfg.rms_norm_eps)
                else:
                    out = st(ids, ctx, pool.caches, rows)
            step_out.append(out[0, -1].float() @ head.t())
        outs.append(torch.stack(step_out))
    assert torch.equal(outs[0].argmax(-1), outs[1].argmax(-1))
    assert float((outs[0] - outs[1]).abs().max()) < 2e-3 * float(outs[0].abs().max())
