# SPDX-License-Identifier: Apache-2.0
"""CPU test for laguna-moe-sigmoid-routing: the routing helpers the fix inserts.

Needs torch (CPU build is enough). Run with: pytest fixes/laguna-moe-sigmoid-routing
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

HERE = Path(__file__).resolve().parent


def _helpers():
    spec = importlib.util.spec_from_file_location("moe_routing_apply", HERE / "apply.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ns = {"torch": torch}
    exec(mod.HELPERS, ns)
    return ns


def _layer(num_experts=16, top_k=4, **overrides):
    attrs = dict(
        top_k=top_k,
        use_grouped_topk=False,
        custom_routing_function=None,
        scoring_func="softmax",
        e_score_correction_bias=None,
        renormalize=True,
        routed_scaling_factor=1.0,
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (dict(), True),
        (dict(scoring_func="sigmoid"), False),
        (dict(e_score_correction_bias=torch.zeros(16)), False),
        (dict(renormalize=False), False),
        (dict(use_grouped_topk=True), False),
        (dict(custom_routing_function=lambda **kwargs: None), False),
    ],
)
def test_is_plain_softmax_topk(overrides, expected):
    assert _helpers()["is_plain_softmax_topk"](_layer(**overrides)) is expected


def test_sigmoid_bias_affects_selection_only():
    ns = _helpers()
    num_experts, top_k, tokens = 16, 4, 8
    logits = torch.randn(tokens, num_experts, dtype=torch.bfloat16)
    bias = torch.zeros(num_experts)
    bias[-top_k:] = 10.0  # forces the last experts regardless of the logits
    layer = _layer(num_experts, top_k, scoring_func="sigmoid", e_score_correction_bias=bias, routed_scaling_factor=2.5)
    weights, ids = ns["_topk_routing"](layer, logits)
    assert ids.dtype == torch.int32
    expected_ids = torch.arange(num_experts - top_k, num_experts).expand(tokens, -1)
    torch.testing.assert_close(ids.to(torch.int64).sort(dim=-1).values, expected_ids)
    scores = logits.float().sigmoid().gather(1, ids.to(torch.int64))
    torch.testing.assert_close(weights, scores / scores.sum(dim=-1, keepdim=True) * 2.5)


def test_softmax_without_renormalize_keeps_raw_probabilities():
    ns = _helpers()
    logits = torch.randn(8, 16)
    weights, ids = ns["_topk_routing"](_layer(renormalize=False), logits)
    ref_weights, ref_ids = torch.topk(torch.softmax(logits, dim=-1), 4, dim=-1)
    torch.testing.assert_close(ids.to(torch.int64), ref_ids)
    torch.testing.assert_close(weights, ref_weights)
    assert bool((weights.sum(dim=-1) < 1.0).all())


def test_plain_softmax_matches_stock_inline_branch():
    ns = _helpers()
    logits = torch.randn(8, 16)
    weights, ids = ns["_topk_routing"](_layer(), logits)
    stock = torch.softmax(logits, dim=1, dtype=torch.float32)
    stock_w, stock_ids = torch.topk(stock, 4, dim=-1)
    stock_w /= stock_w.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(ids.to(torch.int64), stock_ids)
    torch.testing.assert_close(weights, stock_w)


def test_unknown_scoring_func_is_rejected():
    with pytest.raises(ValueError):
        _helpers()["_topk_routing"](_layer(scoring_func="sqrtsoftplus"), torch.randn(2, 16))
