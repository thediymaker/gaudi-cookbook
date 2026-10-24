# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the host-side layout code of every kernel variant (no Gaudi device needed):
``python3 -m pytest tests/``. Skipped without torch.

The kernels read the packed weights in their served order and expect the host to provide the activation row
permuted per 512-k chunk (``permute_x``) and, for gpt-oss, the biases re-laid in the kernels' column order
(``bias13``, ``bias2``). These tests emulate the kernels' lane arithmetic in torch from the raw bytes:

* w13: 128 u16 lanes per 512-k chunk; lane i of the low-nibble plane of byte pair (2i, 2i+1) gives k = 4i (low
  byte) and k = 4i + 2 (high byte), the high-nibble plane gives k = 4i + 1 and 4i + 3; lane i of chunk c takes scale
  byte 16c + i // 8. Output column c of ap is intermediate index perm(c); its gate / up rows are (2g, 2g + 1) for
  interleaved rows and (g, I + g) for [gate; up] rows;
* w2: the same dot on the permuted activation row; per token, the routing-weighted local routes in ascending
  local-expert order (mxk_w2x, mxk_w2xb), or one row per route (mxk_w2xr);

and check the result against the plain fp32 reference (mxk_harness.Reference) for each variant's row layout,
activation, bias and output mode. Shapes are small; gpt-oss shapes are not multiples of 512 to cover padding.
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


def emulate(spec, d, x, ids, tw, e_min, e_local, scales13=None):
    """The kernels' arithmetic on the raw bytes. scales13 overrides the w13 scales (the served +k shift)."""
    k, inter, topk = d["k"], d["inter"], d["topk"]
    kp, ip = rt.round512(k), rt.round512(inter)
    alpha, limit, beta, ascale = H.kernel_act_params(spec) if scales13 is not None else (*spec["act"], 1.0)
    s13 = d["s13"] if scales13 is None else scales13
    bias = "b13" in d
    if bias:
        b13k, b2k = rt.bias13(d["b13"], inter), rt.bias2(d["b2"])
    perm = rt.perm_index(ip)
    xp = rt.permute_x(x, kp)
    tokens = x.shape[0]
    rows = torch.zeros(tokens, topk, k)
    for t in range(tokens):
        for e, j in sorted(
            (int(ids[t, j]) - e_min, j) for j in range(topk) if 0 <= int(ids[t, j]) - e_min < e_local
        ):  # ascending local expert, as the kernels walk the routes
            ap = torch.zeros(ip)
            for c in range(ip):
                g = int(perm[c])
                if g >= inter:
                    continue  # padding column: exact zero
                rg, ru = (2 * g, 2 * g + 1) if spec["layout"] == "il" else (g, inter + g)
                gs = lane_dot(d["w13"][e, rg], s13[e, rg], xp[t])
                us = lane_dot(d["w13"][e, ru], s13[e, ru], xp[t])
                if bias:
                    gs, us = gs + b13k[e, c % 2, c // 2], us + b13k[e, 2 + c % 2, c // 2]
                gs, us = gs.clamp(max=limit), us.clamp(-limit, limit)
                ap[c] = (gs * torch.sigmoid(alpha * gs) * ((us + beta) * ascale)).to(torch.bfloat16).float()
            for n in range(k):
                o = lane_dot(d["w2"][e, n], d["s2"][e, n], ap)
                rows[t, j, n] = o + (b2k[e, n % 2, n // 2] if bias else 0.0)
    if spec["w2"] == "route":
        return rows.view(tokens * topk, k)
    return (rows * tw.to(torch.bfloat16).float().unsqueeze(-1)).sum(1)


def small(variant, k, inter, e_local=4, topk=2, e_global=8):
    spec = dict(H.VARIANTS[variant])
    spec.update(topk=topk, e_global=e_global)
    d = H.random_layer(variant, e_local, seed=1, k=k, inter=inter, topk=topk, e_global=e_global)
    return spec, d


# token 0: two local routes, token 1: one, token 2: none (e_min 2, 4 local experts of 8)
IDS = torch.tensor([[3, 5], [0, 4], [6, 7]], dtype=torch.int32)
TW = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.5, 0.5]])


@pytest.mark.parametrize(
    "variant,k,inter",
    [("gptoss", 576, 544), ("glm", 1024, 512), ("dsv41", 1024, 256), ("muse", 512, 512), ("kimi", 1536, 512)],
)
def test_kernel_emulation_matches_reference(variant, k, inter):
    spec, d = small(variant, k, inter)
    x = (torch.randn(3, k, generator=torch.Generator().manual_seed(2)) * 2).to(torch.bfloat16)
    y_ref, local = H.Reference(spec, d, 2, 4)(x, IDS, TW)
    assert local == 3
    y_emu = emulate(spec, d, x, IDS, TW, 2, 4)
    if spec["w2"] == "route":
        assert y_emu.shape == (6, k)
        assert torch.equal(y_emu[[2, 4, 5]], torch.zeros(3, k))  # routes to experts on other ranks: zero rows
    else:
        assert torch.equal(y_emu[2], torch.zeros(k))
    # same math, other fp32 summation order: only rare bf16 rounding flips of the intermediate differ. A layout
    # error (wrong permutation, nibble order, scale group, row pairing or bias plane) gives rel_l2 of order 1.
    assert H.rel_l2(y_emu, y_ref) < 1e-3


def test_shifted_scales_are_exact():
    """DeepSeek serves w13 scales +7: kernel_act_params must give bit for bit the unshifted result."""
    spec, d = small("dsv41", 512, 256)
    x = (torch.randn(3, 512, generator=torch.Generator().manual_seed(3)) * 6).to(torch.bfloat16)  # clamps active
    y_raw = emulate(spec, d, x, IDS, TW, 2, 4)
    y_shift = emulate(spec, d, x, IDS, TW, 2, 4, scales13=d["s13"] + spec["scale_shift"])
    assert torch.equal(y_raw, y_shift)
    alpha, limit, beta, ascale = H.kernel_act_params(spec)
    assert (alpha, limit, beta, ascale) == (2.0**-7, 10.0 * 2**7, 0.0, 2.0**-14)


def test_activation_layouts():
    h = torch.arange(8, dtype=torch.float32).view(1, 8) - 3.0
    silu = H.VARIANTS["glm"]
    g, u = h[:, :4], h[:, 4:]
    assert torch.allclose(H.activation(silu, h, 4), torch.nn.functional.silu(g) * u)
    il = H.VARIANTS["dsv41"]
    g, u = h[:, 0::2], h[:, 1::2]
    assert torch.allclose(H.activation(il, h, 4), torch.nn.functional.silu(g.clamp(max=10)) * u.clamp(-10, 10))


def test_permute_x_matches_runtime():
    x = torch.randn(3, 1024).to(torch.bfloat16)
    assert torch.equal(H.permute_x(x), rt.permute_x(x, 1024))


def test_perm_index_is_x_permutation():
    x = torch.arange(1024, dtype=torch.float32).view(1, 1024)  # fp32: exact integers
    assert torch.equal(rt.permute_x(x, 1024)[0], rt.perm_index(1024).float())


def test_permute_x_pads_with_zeros():
    xp = rt.permute_x(torch.ones(2, 600, dtype=torch.bfloat16), 1024)
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


def test_glm_checkpoint_tp_slice(tmp_path):
    """A full GLM served-layout checkpoint (I = 2 x the per-rank I) is cut to this rank's half of every expert."""
    st = pytest.importorskip("safetensors.torch")
    import json

    k, i_full, e = 512, 2048, 3
    g = torch.Generator().manual_seed(4)
    full = {
        "w13_weight": torch.randint(0, 256, (e, 2 * i_full, k // 2), generator=g, dtype=torch.int32).to(torch.uint8),
        "w13_weight_scale": torch.randint(0, 256, (e, 2 * i_full, k // 32), generator=g, dtype=torch.int32).to(
            torch.uint8
        ),
        "w2_weight": torch.randint(0, 256, (e, k, i_full // 2), generator=g, dtype=torch.int32).to(torch.uint8),
        "w2_weight_scale": torch.randint(0, 256, (e, k, i_full // 32), generator=g, dtype=torch.int32).to(torch.uint8),
    }
    pre = H.SERVED_PREFIX["glm"].format(layer=5)
    st.save_file({pre + n: v for n, v in full.items()}, str(tmp_path / "a.safetensors"))
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": {pre + n: "a.safetensors" for n in full}}, f)
    for r in (0, 1):
        d = H.load_layer("glm", str(tmp_path), 5, e, 0, tp_rank=r)
        lo, hi = r * 1024, (r + 1) * 1024
        assert d["inter"] == 1024 and d["k"] == k
        assert torch.equal(d["w13"][:, :1024], full["w13_weight"][:, lo:hi])
        assert torch.equal(d["w13"][:, 1024:], full["w13_weight"][:, i_full + lo : i_full + hi])
        assert torch.equal(d["w2"], full["w2_weight"][..., lo // 2 : hi // 2])
        assert torch.equal(d["s2"], full["w2_weight_scale"][..., lo // 32 : hi // 32])


def test_decode_cases_respect_caps():
    for name, spec in H.VARIANTS.items():
        cases = H.decode_cases(spec)
        assert cases, name
        assert all(t <= spec["max_tokens"] and t * spec["topk"] <= 128 for t, _ in cases), name
