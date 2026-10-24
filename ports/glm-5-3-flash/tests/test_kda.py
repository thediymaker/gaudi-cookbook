# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the KDA kernels against a float64 per-token recurrence (needs torch; skips without it).

The reference is the delta rule written out one token at a time, in float64:

    S = S * exp(g_t);  S = S + beta_t * k_t (v_t - k_t^T S);  o_t = q_t^T S      (q, k l2-normalised, q / sqrt(K))

Inputs use the model's gate range (lower bound -5, so within-chunk decay reaches about -300) to exercise the gauge,
and lengths that are not multiples of the 64-token chunk.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_KDA = Path(__file__).resolve().parents[1] / "glm5_next_hpu" / "glm5_next_hpu" / "kda.py"
_spec = importlib.util.spec_from_file_location("glm5_next_kda", _KDA)
kda = importlib.util.module_from_spec(_spec)
sys.modules["glm5_next_kda"] = kda
_spec.loader.exec_module(kda)


def reference(q, k, v, g, beta, state):
    """float64 per-token recurrence. q, k, v, g: [B, T, H, D]; beta: [B, T, H]; state [B, H, K, V]."""
    q, k, v, g, beta, s = (x.double() for x in (q, k, v, g, beta, state))
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) * q.shape[-1] ** -0.5
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    outs = []
    for t in range(q.shape[1]):
        s = s * g[:, t].exp().unsqueeze(-1)
        kv = (s * k[:, t].unsqueeze(-1)).sum(-2)
        s = s + k[:, t].unsqueeze(-1) * ((v[:, t] - kv) * beta[:, t].unsqueeze(-1)).unsqueeze(-2)
        outs.append((s * q[:, t].unsqueeze(-1)).sum(-2))
    return torch.stack(outs, 1), s


def inputs(b, t, h=2, d=16, lower=-5.0, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(b, t, h, d, generator=gen) for _ in range(3))
    raw = torch.randn(b, t, h * d, generator=gen)
    a_log = torch.randn(h, generator=gen) * 0.5
    g = kda.hpu_kda_gate(raw, a_log, d, g_bias=None, lower_bound=lower)
    beta = torch.sigmoid(torch.randn(b, t, h, generator=gen))
    state = torch.randn(b, h, d, d, generator=gen) * 0.1
    return q, k, v, g, beta, state


@pytest.mark.parametrize("t", [1, 37, 64, 200, 513])
@pytest.mark.parametrize("lower", [-5.0, -1.24])
def test_chunk_kernels_match_reference(t, lower):
    q, k, v, g, beta, state = inputs(2, t, lower=lower)
    ref_o, ref_s = reference(q, k, v, g, beta, state)
    for fn in (kda.hpu_chunk_kda, kda.hpu_chunk_kda_scan):
        o, s = fn(q, k, v, g, beta, initial_state=state)
        assert (o.double() - ref_o).abs().max() < 2e-4, fn.__name__
        assert (s.double() - ref_s).abs().max() < 2e-4, fn.__name__


def test_scan_matches_loop():
    q, k, v, g, beta, state = inputs(3, 1000, seed=3)
    o1, s1 = kda.hpu_chunk_kda(q, k, v, g, beta, initial_state=state)
    o2, s2 = kda.hpu_chunk_kda_scan(q, k, v, g, beta, initial_state=state)
    assert (o1 - o2).abs().max() < 1e-5
    assert (s1 - s2).abs().max() < 1e-5


def test_recurrent_matches_reference():
    q, k, v, g, beta, state = inputs(4, 1, seed=5)
    ref_o, ref_s = reference(q, k, v, g, beta, state)
    o, s = kda.hpu_recurrent_kda(q, k, v, g, beta, initial_state=state)
    assert (o.double() - ref_o).abs().max() < 1e-5
    assert (s.double() - ref_s).abs().max() < 1e-5


def test_prefill_then_decode_continues_state():
    q, k, v, g, beta, state = inputs(1, 130, seed=7)
    ref_o, _ = reference(q, k, v, g, beta, state)
    _, s = kda.hpu_chunk_kda_scan(q[:, :129], k[:, :129], v[:, :129], g[:, :129], beta[:, :129], initial_state=state)
    o, _ = kda.hpu_recurrent_kda(q[:, 129:], k[:, 129:], v[:, 129:], g[:, 129:], beta[:, 129:], initial_state=s)
    assert (o.double() - ref_o[:, 129:]).abs().max() < 2e-4


def test_ut_inverse_matches_exact():
    """(I - A)^-1 for a strictly lower-triangular A with signed entries up to 0.9, as the real model's."""
    gen = torch.Generator().manual_seed(11)
    a = (torch.rand(8, 64, 64, generator=gen) * 1.8 - 0.9).tril(-1)
    eye = torch.eye(64)
    exact = torch.linalg.inv(eye.double() - a.double())
    blocked = kda._ut_inverse(a, eye)
    assert ((blocked.double() - exact).abs().max() / exact.abs().max()) < 1e-4


def test_scan_does_not_mutate_inputs():
    q, k, v, g, beta, state = inputs(1, 100, seed=13)
    copies = [x.clone() for x in (q, k, v, g, beta, state)]
    kda.hpu_chunk_kda_scan(q, k, v, g, beta, initial_state=state)
    assert all(torch.equal(a, b) for a, b in zip(copies, (q, k, v, g, beta, state), strict=False))
