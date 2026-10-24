# SPDX-License-Identifier: Apache-2.0
"""glm5_hpu.moe_layout on CPU: the EP x TP group layout, the TP slice loader and the placement slot mapping."""

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from glm5_hpu import moe_layout as ml  # noqa: E402


def test_group_layout(monkeypatch):
    monkeypatch.setattr(ml, "TP", 2)
    assert [ml.group_layout(8, r, 1, 1) for r in range(4)] == [(2, 0, 4, 0), (2, 1, 4, 0), (2, 0, 4, 1), (2, 1, 4, 1)]
    assert ml.group_layout(8, 0, 2, 1) is None  # data parallel: vLLM's layout
    monkeypatch.setattr(ml, "TP", 1)
    assert ml.group_layout(8, 3, 1, 1) is None


def _layer(tp, tp_rank, ep, ep_rank, name="model.layers.5.mlp.experts"):
    pc = SimpleNamespace(tp_size=tp, tp_rank=tp_rank, use_ep=True, ep_size=ep)
    return SimpleNamespace(moe_config=SimpleNamespace(moe_parallel_config=pc, ep_rank=ep_rank), layer_name=name)


@pytest.mark.parametrize("w13", [True, False])
def test_tp_slice_loader(monkeypatch, w13):
    monkeypatch.setattr(ml, "PLACEMENT", "")
    e, inter, k = 8, 8, 6  # experts, intermediate rows per projection, columns
    full = torch.arange(e * 2 * inter * k, dtype=torch.int32).view(e, 2 * inter, k) if w13 else None
    if not w13:
        full = torch.arange(e * k * inter, dtype=torch.int32).view(e, k, inter)
    t, ep = 2, 2
    for ep_rank in range(ep):
        for r in range(t):
            n_local = e // ep
            shape = (n_local, inter, k) if w13 else (n_local, k, inter // t)
            p = torch.zeros(shape, dtype=torch.int32)
            assert ml.load_fused(_layer(t, r, ep, ep_rank), p, full)
            ids = list(range(ep_rank * n_local, (ep_rank + 1) * n_local))
            if w13:
                h = inter // t
                want = torch.cat([full[ids, r * h : (r + 1) * h], full[ids, inter + r * h : inter + (r + 1) * h]], 1)
            else:
                want = full[ids][:, :, r * (inter // t) : (r + 1) * (inter // t)]
            assert torch.equal(p, want)


def test_placement_loader_and_route_mapping(monkeypatch, tmp_path):
    perm = [3, 0, 2, 1, 7, 6, 5, 4]
    f = tmp_path / "placement.json"
    f.write_text(json.dumps({"n_experts": 8, "n_ranks": 2, "layers": {"5": perm}}))
    monkeypatch.setattr(ml, "PLACEMENT", str(f))
    ml._cache.clear()
    full = torch.arange(8 * 2 * 3, dtype=torch.int32).view(8, 2, 3)
    p = torch.zeros(4, 2, 3, dtype=torch.int32)
    assert ml.load_fused(_layer(1, 0, 2, 1), p, full)
    assert torch.equal(p, full[perm[4:]])
    slot_of = torch.tensor([perm.index(x) for x in range(8)], dtype=torch.int32)
    layer = SimpleNamespace(_glm_slot_of=slot_of)
    ids = torch.tensor([[0, 4], [7, 3]], dtype=torch.int32)
    assert ml.map_ids(layer, ids).tolist() == [[1, 7], [4, 0]]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"n_experts": 8, "n_ranks": 2, "layers": {"5": [0] * 8}}))
    monkeypatch.setattr(ml, "PLACEMENT", str(bad))
    ml._cache.clear()
    with pytest.raises(ValueError):
        ml.placement()
    ml._cache.clear()


def test_shipped_placement_is_valid(monkeypatch):
    import os

    path = os.path.join(os.path.dirname(__file__), "..", "serving", "glm53-ep4-placement.json")
    monkeypatch.setattr(ml, "PLACEMENT", path)
    ml._cache.clear()
    layers = ml.placement()
    assert sorted(layers) == list(range(3, 79))
    ml._cache.clear()
