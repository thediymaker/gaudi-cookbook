# SPDX-License-Identifier: Apache-2.0
"""glm5_hpu.decode_lean on CPU: the fast router equals the grouped top-k with one group, and the device rejection
walk equals the host walk it replaces."""

import pytest

torch = pytest.importorskip("torch")

from glm5_hpu import decode_lean as dl  # noqa: E402

PH = -1


def _grouped_topk_ref(scores, topk, renormalize, scale, bias, n_group, topk_group):
    """vLLM's torch grouped top-k (fused_moe/router/cpu_router.py) after the scoring function."""
    n = scores.shape[0]
    original = scores
    scores = scores + bias.unsqueeze(0)
    group_scores = scores.view(n, n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
    smask = mask.unsqueeze(-1).expand(n, n_group, scores.shape[-1] // n_group).reshape(n, -1)
    tmp = scores.masked_fill(~smask.bool(), float("-inf"))
    ids = torch.topk(tmp, k=topk, dim=-1, sorted=False)[1]
    w = original.gather(1, ids)
    if renormalize:
        w = w / w.sum(dim=-1, keepdim=True)
    return w * scale, ids.to(torch.int32)


def test_router_fast_equals_grouped_topk_with_one_group():
    g = torch.Generator().manual_seed(0)
    scores = torch.randn(16, 256, generator=g).sigmoid()
    bias = torch.randn(256, generator=g) * 0.1
    ref = _grouped_topk_ref(scores, 8, True, 2.5, bias, 1, 1)
    out = dl.grouped_topk_all_groups(scores, 8, True, 2.5, bias)
    assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])


def _host_walk(n_list, k, accept, tok, bonus):
    out = torch.full((len(n_list), k + 1), PH, dtype=torch.int32)
    for i, n in enumerate(n_list):
        ok = True
        for j in range(n):
            out[i, j] = int(tok[i, j])
            if not bool(accept[i, j]):
                ok = False
                break
        if ok:
            out[i, n] = int(bonus[i])
    return out


@pytest.mark.parametrize("k", [1, 2, 3])
def test_accept_grid_equals_host_walk(k):
    g = torch.Generator().manual_seed(k)
    for _ in range(50):
        n_req = int(torch.randint(1, 6, (1,), generator=g))
        n_list = torch.randint(0, k + 1, (n_req,), generator=g).tolist()
        accept = torch.rand(n_req, k, generator=g) > 0.4
        tok = torch.randint(0, 1000, (n_req, k), generator=g, dtype=torch.int32)
        bonus = torch.randint(0, 1000, (n_req,), generator=g, dtype=torch.int32)
        got = dl._accept_grid(n_list, k, accept, tok, bonus, "cpu", PH)
        assert torch.equal(got, _host_walk(n_list, k, accept, tok, bonus))
