# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Qwen3-TTS engine's pure parts (no model, no device): `python3 -m pytest servers/qwen3-tts`."""

import pytest

torch = pytest.importorskip("torch")
from server_modules import load  # noqa: E402

(engine,) = load(__file__, "engine")


def test_gumbel_topk_is_an_exact_draw():
    """Sampled frequencies match softmax(top-k(logits / T)); total variation within sampling noise."""
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(1, 64, generator=g) * 2
    k, temp, n = 8, 0.9, 200_000
    u = torch.rand(n, k, generator=g)
    draws = engine.gumbel_topk(logits.expand(n, -1), k, temp, u)
    vals, idx = torch.topk(logits[0] / temp, k)
    want = torch.zeros(64)
    want[idx] = torch.softmax(vals, -1)
    got = torch.bincount(draws, minlength=64).float() / n
    assert set(draws.unique().tolist()) <= set(idx.tolist())  # never outside the top k
    assert 0.5 * (got - want).abs().sum() < 0.01


def test_bucket():
    assert engine.bucket(1, (4, 16)) == 4 and engine.bucket(16, (4, 16)) == 16
    with pytest.raises(ValueError):
        engine.bucket(17, (4, 16))


def test_captured_without_graphs_reruns(monkeypatch):
    monkeypatch.setattr(engine, "htcore", None)  # no lazy-mode flush on a box without a card
    calls = []
    c = engine.Captured(lambda: calls.append(1) or len(calls), use_graph=False)
    assert c.out == 1 and c() == 2 and c() == 3


def test_frame_cap():
    pytest.importorskip("fastapi")
    pytest.importorskip("prometheus_client")
    pytest.importorskip("uvicorn")
    (server,) = load(__file__, "server")

    assert server.frame_cap("hello", 2048) == max(16, 2 * 5 + 40)
    assert server.frame_cap("你好世界", 2048) == 6 * 4 + 40
    assert server.frame_cap("a" * 5000, 2048) == 2048
