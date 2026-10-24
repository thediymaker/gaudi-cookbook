# SPDX-License-Identifier: Apache-2.0
"""CPU test: DSV41_TOPK_SELECT=thresh picks exactly the set of the stable descending sort (ties to the lower index)."""

import pytest

torch = pytest.importorskip("torch")

from dsv41_hpu.csa import select_topk_thresh  # noqa: E402


@pytest.mark.parametrize("L,k", [(512, 64), (1000, 512), (70000, 512)])
def test_thresh_equals_stable_sort(L, k):
    g = torch.Generator().manual_seed(L)
    # few distinct values, so ties are everywhere, plus -inf (masked) entries
    score = torch.randint(-8, 8, (3, 2, L), generator=g).float().to(torch.bfloat16) / 4
    score[..., ::7] = float("-inf")
    score[0, 0, :] = 0.0  # a row of all ties
    v, i = select_topk_thresh(score, k)
    ref = torch.sort(score, dim=-1, descending=True, stable=True).indices[..., :k]
    assert torch.equal(torch.sort(i, -1).values, torch.sort(ref, -1).values)
    assert torch.equal(v, torch.gather(score, -1, i))
