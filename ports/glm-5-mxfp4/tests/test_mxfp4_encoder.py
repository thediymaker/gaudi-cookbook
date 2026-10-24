# SPDX-License-Identifier: Apache-2.0
"""convert/mxfp4.py on CPU: layout, the Quark shared-exponent rule, ties to even, and the floor encoder."""

import pytest

torch = pytest.importorskip("torch")

import mxfp4  # noqa: E402


def _encode_block(values, enc=mxfp4.quantize_mxfp4):
    w = torch.tensor([values], dtype=torch.float32)
    packed, scale = enc(w)
    return mxfp4.unpack_nibbles(packed)[0].tolist(), int(scale[0, 0])


def test_layout_low_nibble_first_and_scale_bias():
    vals = [0.5, 1.0] + [0.0] * 30
    packed, scale = mxfp4.quantize_mxfp4(torch.tensor([vals]))
    assert packed.shape == (1, 16) and scale.shape == (1, 1)
    # amax 1.0: X = 0 - 2 = -2, codes 0.5 * 4 = 2.0 -> code 4, 1.0 * 4 = 4.0 -> code 6
    assert int(scale[0, 0]) == 127 - 2
    assert int(packed[0, 0]) == 4 | (6 << 4)


def test_roundtrip_exact_on_grid():
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 16, (64, 256), generator=g)
    exps = torch.randint(-20, 20, (64, 8), generator=g).float()
    w = mxfp4.E2M1_VALUES[codes] * torch.pow(2.0, exps).repeat_interleave(32, dim=1)
    w[:, ::32] = 6.0 * torch.pow(2.0, exps)  # amax = 6 * 2^X: the floor rule, no bump
    p, s = mxfp4.quantize_mxfp4(w)
    assert torch.equal(mxfp4.dequant_mxfp4(p, s), w)


def test_quark_bump_at_ratio_seven():
    # amax 7.0: floor rule X = 0 (ratio 7 -> clamped to 6); Quark bumps to X = 1 (7 / 2 = 3.5 -> 4 by ties to even)
    codes, scale = _encode_block([7.0] + [0.0] * 31)
    assert scale == 128 and codes[0] == 6
    codes, scale = _encode_block([6.9] + [0.0] * 31)
    assert scale == 127 and codes[0] == 7
    codes, scale = _encode_block([7.0] + [0.0] * 31, mxfp4.quantize_mxfp4_floor)
    assert scale == 127 and codes[0] == 7


def test_ties_to_even_code_and_no_negative_zero():
    # amax 4 -> X = 0. 0.25 (tie 0/0.5) -> code 0; 0.75 (0.5/1.0) -> code 2; 1.25 -> 2; 1.75 -> 4; 2.5 -> 4;
    # 3.5 -> 6; 5.0 (4/6) -> 6
    vals = [4.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.1] + [0.0] * 23
    codes, scale = _encode_block(vals)
    assert scale == 127
    assert codes[1:9] == [0, 2, 2, 4, 4, 6, 6, 0]


def test_floor_encoder_rounds_half_down_and_keeps_sign():
    vals = [4.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.1] + [0.0] * 23
    codes, scale = _encode_block(vals, mxfp4.quantize_mxfp4_floor)
    assert scale == 127
    assert codes[1:9] == [0, 1, 2, 3, 4, 5, 6, 8]


def test_zero_block():
    codes, scale = _encode_block([0.0] * 32)
    assert scale == 127 and set(codes) == {0}


def test_error_quark_not_worse_than_floor():
    g = torch.Generator().manual_seed(1)
    w = torch.randn(256, 1024, generator=g) * 0.02
    errs = []
    for enc in (mxfp4.quantize_mxfp4, mxfp4.quantize_mxfp4_floor):
        p, s = enc(w)
        errs.append(float((mxfp4.dequant_mxfp4(p, s) - w).norm() / w.norm()))
    assert errs[0] < errs[1]
