# SPDX-License-Identifier: Apache-2.0
"""CPU test: the fake quantizers and their DSV41_FQ_FASTBITS forms against independent references."""

import pytest

torch = pytest.importorskip("torch")

from dsv41_hpu import numerics as N  # noqa: E402

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _values(n=200_000, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, generator=g) * torch.exp2(torch.randint(-12, 12, (n,), generator=g).float())
    return torch.cat([v, torch.tensor([0.0, -0.0, 2.0**-130, 448.0, -448.0, 6.0, 0.25, 0.75, 2.5, 5.0])])


def _e2m1_nearest_even(x):
    """Nearest E2M1 level by exhaustive search; ties to the level with the even code."""
    a = x.abs().clamp(max=6.0)
    d = (a.unsqueeze(-1) - E2M1).abs()
    best = d.min(-1, keepdim=True).values
    cand = d == best
    idx = torch.where(cand.sum(-1) > 1, (cand * (torch.arange(8) % 2 == 0)).float().argmax(-1), d.argmin(-1))
    return torch.copysign(E2M1[idx], x)


def test_round_e4m3fn_matches_torch_cast():
    v = _values().clamp(-448, 448)
    ref = v.to(torch.float8_e4m3fn).float()
    assert torch.equal(N.round_e4m3fn(v), ref)


def test_round_e2m1_matches_search():
    v = _values().clamp(-6, 6)
    assert torch.equal(N.round_e2m1(v), _e2m1_nearest_even(v))


def test_ratio_rounding_is_exact():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(4096, 16, generator=g) * 3
    amax = x.abs().amax(-1, keepdim=True)
    s = N.round_e4m3fn_ratio(amax, 6.0)
    ref_s = (amax.double() / 6.0).float().to(torch.float8_e4m3fn).float()
    assert torch.equal(s, ref_s)
    q = N.round_e2m1_ratio(x, s)
    ref_q = _e2m1_nearest_even((x.double() / s.double()).clamp(-6, 6).float())
    assert torch.equal(q, ref_q)


def test_fake_fp8_block_scales_are_powers_of_two():
    x = torch.randn(8, 512) * 37
    y = N.fake_fp8_ue8m0(x)
    xb = x.view(8, 16, 32)
    s = N.pow2_ceil(xb.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.0)
    assert torch.equal(y.view(8, 16, 32), (xb / s).to(torch.float8_e4m3fn).float() * s)


@pytest.mark.parametrize("fn", ["fake_fp8_ue8m0", "fake_fp4_ue8m0", "fake_fp4_e4m3"])
def test_fastbits_is_bit_identical(monkeypatch, fn):
    x = torch.randn(64, 512) * torch.exp2(torch.randint(-20, 20, (64, 1)).float())
    monkeypatch.setattr(N, "FASTBITS", False)
    a = getattr(N, fn)(x)
    monkeypatch.setattr(N, "FASTBITS", True)
    b = getattr(N, fn)(x)
    assert torch.equal(a, b)


def test_lean_and_fused_rope_match(monkeypatch):
    x = torch.randn(3, 5, 4, 128, dtype=torch.bfloat16)
    inv = N.rope_inv_freq(32, 160000.0, 65536, 16, 32, 1)
    cos, sin = N.rope_tables(1000, inv)
    p = torch.tensor([0, 7, 99, 500, 999])
    c, s = cos[p].view(1, 5, 1, -1), sin[p].view(1, 5, 1, -1)
    for inverse in (False, True):
        monkeypatch.setattr(N, "ROPE_FUSED", False)
        a = N.rope_tail(x, c, s, 32, inverse)
        monkeypatch.setattr(N, "ROPE_FUSED", True)
        b = N.rope_tail(x, c, s, 32, inverse)
        assert torch.equal(a, b)
    w = torch.randn(128)
    monkeypatch.setattr(N, "LEAN", False)
    a = N.rms_norm(x, w, 1e-20)
    monkeypatch.setattr(N, "LEAN", True)
    b = N.rms_norm(x, w, 1e-20)
    assert (a.float() - b.float()).abs().max() <= 2**-6
