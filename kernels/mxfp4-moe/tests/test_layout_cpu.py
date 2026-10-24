# SPDX-License-Identifier: Apache-2.0
"""CPU test of the host-side layout code (no Gaudi device needed): ``python3 -m pytest tests/test_layout_cpu.py``.

The kernels read the packed weights in their served order and expect the host to provide the activation row
permuted per 512-k chunk and the biases re-laid in the kernels' column order (``permute_x``, ``bias13``, ``bias2`` in
python/mxfp4_tpc_moe_rt.py). This test emulates the kernels' lane arithmetic in torch from the raw bytes:

* w13: 128 u16 lanes per 512-k chunk; lane i of the low-nibble plane of byte pair (2i, 2i+1) gives k = 4i (low byte)
  and k = 4i + 2 (high byte), the high-nibble plane gives k = 4i + 1 and 4i + 3; lane i of chunk c takes scale byte
  16c + i // 8; output column c of row pair (gate, up) is intermediate perm(c); bias planes indexed by c % 2, c // 2;
* w2: the same dot on the permuted activation row; output row n takes bias plane n % 2 at n // 2; per-token sum of
  the routing-weighted local routes in ascending local-expert order;

and checks the result against the plain fp32 reference. Shapes are small and not multiples of 512 to cover padding.
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

import mxfp4_tpc_moe_rt as rt  # noqa: E402
import mxk_harness as H  # noqa: E402

LUT = torch.tensor(H.E2M1)


def lane_dot(wrow, srow, xp):
    """Kernel dot of one packed weight row (bytes, natural order) with a permuted activation row (len Kp)."""
    kp = xp.numel()
    nch = kp // 512
    b = torch.zeros(nch * 256, dtype=torch.uint8)
    b[: wrow.numel()] = wrow  # bytes past the row read as zero
    s = torch.full((nch * 16,), 127, dtype=torch.uint8)
    s[: srow.numel()] = srow
    b = b.view(nch, 128, 2).long()
    planes = [LUT[b[..., 0] & 15], LUT[b[..., 0] >> 4], LUT[b[..., 1] & 15], LUT[b[..., 1] >> 4]]  # p = 0..3
    w = torch.stack(planes, -1)  # [c, i, p]: value k = 512c + 4i + p
    lane = torch.arange(128)
    sc = torch.pow(2.0, s.view(nch, 16)[:, lane // 8].float() - 127.0)  # [c, i]
    x = xp.float().view(nch, 4, 128).transpose(1, 2)  # [c, i, p] = xp[512c + 128p + i]
    return (w * sc[..., None] * x).sum()


def emulate(d, x, ids, tw, e_min, e_local, alpha=H.ALPHA, limit=H.LIMIT):
    k, inter, topk = d["k"], d["inter"], d["topk"]
    kp, ip = rt.round512(k), rt.round512(inter)
    b13k, b2k = rt.bias13(d["b13"], inter), rt.bias2(d["b2"])
    assert b13k.shape == (e_local, 4, ip // 2) and b2k.shape == (e_local, 2, k // 2)
    perm = rt.perm_index(ip)
    xp = rt.permute_x(x, kp)
    tokens = x.shape[0]
    y = torch.zeros(tokens, k)
    twb = tw.to(torch.bfloat16).float()
    for t in range(tokens):
        routes = sorted(
            (int(ids[t, j]) - e_min, j) for j in range(topk) if 0 <= int(ids[t, j]) - e_min < e_local
        )  # ascending local expert, as the kernel sums
        for e, j in routes:
            ap = torch.zeros(ip)
            for c in range(ip):
                g = int(perm[c])
                if g >= inter:
                    continue  # padding column: exact zero
                gs = lane_dot(d["w13"][e, 2 * g], d["s13"][e, 2 * g], xp[t]) + b13k[e, c % 2, c // 2]
                us = lane_dot(d["w13"][e, 2 * g + 1], d["s13"][e, 2 * g + 1], xp[t]) + b13k[e, 2 + c % 2, c // 2]
                gs = gs.clamp(max=limit)
                us = us.clamp(-limit, limit)
                ap[c] = (gs * torch.sigmoid(alpha * gs) * (us + 1.0)).to(torch.bfloat16).float()
            for n in range(k):
                o = lane_dot(d["w2"][e, n], d["s2"][e, n], ap) + b2k[e, n % 2, n // 2]
                y[t, n] += twb[t, j] * o
    return y


def test_perm_index_is_x_permutation():
    x = torch.arange(1024, dtype=torch.float32).view(1, 1024)  # fp32: exact integers
    xp = rt.permute_x(x, 1024)
    assert torch.equal(xp[0], rt.perm_index(1024).float())


def test_permute_x_pads_with_zeros():
    x = torch.ones(2, 600, dtype=torch.bfloat16)
    xp = rt.permute_x(x, 1024)
    assert xp.shape == (2, 1024)
    assert int((xp != 0).sum()) == 2 * 600


def test_bias_layouts():
    e, inter, k = 3, 576, 608
    b13 = torch.randn(e, 2 * inter).to(torch.bfloat16)
    b2 = torch.randn(e, k).to(torch.bfloat16)
    b13k, b2k = rt.bias13(b13, inter), rt.bias2(b2)
    perm = rt.perm_index(rt.round512(inter))
    for c in range(rt.round512(inter)):
        g = int(perm[c])
        want_g = b13[:, 2 * g].float() if g < inter else torch.zeros(e)
        want_u = b13[:, 2 * g + 1].float() if g < inter else torch.zeros(e)
        assert torch.equal(b13k[:, c % 2, c // 2], want_g)
        assert torch.equal(b13k[:, 2 + c % 2, c // 2], want_u)
    for n in range(k):
        assert torch.equal(b2k[:, n % 2, n // 2], b2[:, n].float())


def test_kernel_emulation_matches_reference():
    e_local, e_min = 4, 2
    d = H.random_layer(e_local, k=576, inter=544, e_global=8, topk=2, seed=1)
    # token 0: two local routes, token 1: one local route, token 2: none
    ids = torch.tensor([[3, 5], [0, 4], [6, 7]], dtype=torch.int32)
    tw = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.5, 0.5]])
    x = (torch.randn(3, 576, generator=torch.Generator().manual_seed(2)) * 2).to(torch.bfloat16)
    y_ref, local = H.reference(d, x, ids, tw, e_min, e_local)
    assert local == 3
    y_emu = emulate(d, x, ids, tw, e_min, e_local)
    assert torch.equal(y_emu[2], torch.zeros(576))
    # same math, other fp32 summation order: only rare bf16 rounding flips of the intermediate differ. A layout
    # error (wrong permutation, nibble order, scale group or bias plane) gives rel_l2 of order 1.
    assert H.rel_l2(y_emu, y_ref) < 1e-3
