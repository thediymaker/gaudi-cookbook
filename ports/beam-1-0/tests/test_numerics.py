# SPDX-License-Identifier: Apache-2.0
"""beam_hpu.numerics on CPU: the integer bf16 rounding used on HPU equals torch's, and the formulas hold."""

import math

import pytest

torch = pytest.importorskip("torch")

from beam_hpu import numerics as N  # noqa: E402


def test_integer_rounding_matches_torch_bf16():
    g = torch.Generator().manual_seed(0)
    parts = [torch.randn(200000, generator=g) * s for s in (1e-30, 1e-6, 1.0, 1e4, 1e30)]
    bits = torch.randint(0, 2**31 - 1, (200000,), generator=g, dtype=torch.int64).to(torch.int32)
    parts.append(bits.view(torch.float32))  # arbitrary bit patterns, ties included
    edges = [0.0, -0.0, 1.0, 1.00390625, 1.01171875, 3.0e38, -3.0e38, 1e-40, float("inf"), float("-inf")]
    parts.append(torch.tensor(edges))
    x = torch.cat(parts)
    x = x[torch.isfinite(x) | torch.isinf(x)]
    x = x[~torch.isnan(x)]
    got = N.rne_bf16_bits(x)
    want = x.to(torch.bfloat16).to(torch.float32)
    assert torch.equal(got.view(torch.int32), want.view(torch.int32))


def test_cpu_paths_are_plain_casts():
    x = torch.randn(64, 32)
    assert torch.equal(N.rnd(x), x.to(torch.bfloat16).float())
    assert torch.equal(N.store(x), x.to(torch.bfloat16))
    assert N.store(x, None) is x


def test_depth_coeffs():
    for layer in range(52):
        for attn in (True, False):
            xs, rs = N.depth_coeffs(layer, attn)
            n = 2 * (layer + 1) + (0 if attn else 1)
            assert math.isclose(xs, 1 / math.sqrt(n)) and math.isclose(xs**2 + rs**2, 1.0)


def test_ssmax_scale():
    pos = torch.tensor([0, 2047, 4095])
    want = torch.tensor([(1 + 1 / 2048) ** 0.1, 2**0.1, 3**0.1])
    assert torch.allclose(N.ssmax_scale(pos), want, rtol=1e-6)


def test_output_gate_roundings():
    T, H, D = 5, 4, 8
    o, g = torch.randn(T, H * D).to(torch.bfloat16), torch.randn(T, H * D).to(torch.bfloat16)
    b = torch.randn(H)
    out = N.output_gate(o, g, b, H, D)
    gb = (g.float().reshape(T, H, D) + b[None, :, None]).to(torch.bfloat16).float()
    s = torch.sigmoid(gb).to(torch.bfloat16).float()
    want = (o.float().reshape(T, H, D) * s).to(torch.bfloat16).reshape(T, H * D)
    assert torch.equal(out, want)


def test_boundary_is_sandwich_norm_plus_scaled_residual():
    x, r = torch.randn(3, 16).to(torch.bfloat16), torch.randn(3, 16).to(torch.bfloat16)
    w = torch.rand(16) + 0.5
    xs, rs = N.depth_coeffs(4, False)
    p = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w).to(torch.bfloat16)
    a = (p.float() * xs).to(torch.bfloat16).float()
    c = (r.float() * rs).to(torch.bfloat16).float()
    assert torch.equal(N.boundary(x, r, w, 1e-6, (xs, rs)), (a + c).to(torch.bfloat16))


def test_softcap():
    z = torch.tensor([0.0, 30.0, -300.0]).to(torch.bfloat16)
    out = N.softcap(z, 30.0)
    assert out.dtype == torch.float32 and abs(float(out[2]) + 30.0) < 1e-3 and float(out[0]) == 0.0
