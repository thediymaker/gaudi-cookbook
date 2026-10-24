# SPDX-License-Identifier: Apache-2.0
"""Pure-Python tests of the MXFP4 to FP8 re-encoding in convert/mxfp4_to_fp8.py (no torch, no numpy).

Covered:
  - the e4m3fn decoder against known codes, and encode(decode(b)) == b for every Gaudi 2-finite normal code;
  - the (nibble, shift) LUT, exhaustively, against exact Fractions; no 0x80, no exponent-15 code, and with the
    default -5 floor the Gaudi 2 x0.5 load stays a normal;
  - the row encoder on random and adversarial rows (E8M0 spread 9 passes, 10 needs min_exp -6, 11+ fails),
    zero and -0 rows, a NaN scale;
  - the compressed-tensors target/ignore regexes under vLLM's matching rules, and the EP weight filter's
    expert-id regex;
  - an independent oracle (e2m1 from its bit fields, e4m3fn through an fp32 bit pattern, no code shared with the
    converter) against the converter's tables;
  - adversarial channels: widest exact rows, rows whose E8M0 spread exceeds 9 but whose element span still fits,
    the top element coming from a lower-scale block, -0 and +0 in the largest-scale block, the 2^-5/2^-6/2^-7
    subnormal edge, E8M0 0 and 254, a single nonzero element, Gaudi 2's x0.5 load replayed row by row, and a
    property fuzz (the encoder passes iff the element span fits);
  - real rows read raw from a converted MXFP4 checkpoint, only when KIMI_MXFP4_DIR points at one (the output of
    convert/repack_quark_mxfp4.py); skipped otherwise.

The torch path is the vectorised twin of reference_encode_row; `mxfp4_to_fp8.py --self-test` compares the two
byte for byte where torch is installed.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import re
import struct
from fractions import Fraction
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "convert" / "mxfp4_to_fp8.py"
_SPEC = importlib.util.spec_from_file_location("kimi_mxfp4_to_fp8", _PATH)
R = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(R)


def _row_exact(nib, scales, q, k, min_exp):
    """Float64 re-check (exact at these magnitudes): FP8 * 2^k == E2M1 * 2^(s-127)."""
    sk = math.ldexp(1.0, k)
    exps = []
    for b, s in enumerate(scales):
        ss = math.ldexp(1.0, s - R.E8M0_BIAS)
        for j in range(R.BLOCK):
            i = b * R.BLOCK + j
            qb = q[i]
            assert qb != 0x80 and (qb >> 3) & 0xF != 0xF, f"forbidden code {qb:#x}"
            assert R.FP8_DECODE[qb] * sk == R.E2M1[nib[i]] * ss, (i, nib[i], s, qb, k)
            if qb & 0x7F:
                exps.append(((qb >> 3) & 0xF) - 7)
    if exps:
        assert max(exps) == R.FP8_EXP_MAX, f"top element not in the top binade: {max(exps)}"
        assert min(exps) >= min_exp, (min(exps), min_exp)
    else:
        assert k == 0 and set(q) == {0}
    return (min(exps), max(exps)) if exps else (None, None)


# ------------------------------------------------------------------ decoder
def test_e4m3fn_known_codes():
    d = R.e4m3fn_decode
    assert d(0x00) == 0 and d(0x80) == 0
    assert d(0x38) == 1  # exponent field 7, mantissa 0
    assert d(0x08) == Fraction(1, 64)  # smallest normal 2^-6
    assert d(0x01) == Fraction(1, 512)  # smallest subnormal 2^-9
    assert d(0x77) == 240  # Gaudi 2 max finite
    assert d(0x7E) == 448  # e4m3fn max (not finite on Gaudi 2)
    assert d(0x7F) is None and d(0xFF) is None
    assert d(0xF4) == -192 and d(0x74) == 192


def test_encode_decode_roundtrip_all_codes():
    for b in range(256):
        e = (b >> 3) & 0xF
        v = R.e4m3fn_decode(b)
        enc = R.e4m3fn_encode_normal(v, R.FP8_EXP_MIN_NORMAL) if v is not None else None
        if 1 <= e <= 14:
            assert enc == b, (hex(b), v, enc)
        elif e == 0 and b & 0x7F:
            assert enc is None, f"subnormal {b:#x} must not be emitted"
        elif e == 15:
            assert enc is None, f"exponent-15 code {b:#x} is not finite on Gaudi 2"


# ---------------------------------------------------------------------- LUT
def _check_lut(min_exp):
    lut_b, lut_ok = R.build_lut(min_exp)
    assert len(lut_b) == len(lut_ok) == 16 * R.ND
    n_ok = 0
    for n in range(16):
        for d in range(R.D_MIN, R.D_MAX + 1):
            i = n * R.ND + (d - R.D_MIN)
            v = Fraction(R.E2M1[n]) * Fraction(2) ** d
            if v == 0:
                assert lut_ok[i] and lut_b[i] == 0x00, (n, d)
                continue
            e = R.floor_log2(abs(v))
            want = min_exp <= e <= R.FP8_EXP_MAX
            assert bool(lut_ok[i]) == want, (n, d, e, min_exp)
            if want:
                n_ok += 1
                b = lut_b[i]
                assert R.e4m3fn_decode(b) == v, (n, d, hex(b))
                assert b != 0x80 and (b >> 3) & 0xF != 0xF
                assert abs(v) <= 192
                if min_exp >= -5:
                    # Gaudi 2 scale adjustment: (w.float() * 0.5).to(e4m3fn) must stay a normal.
                    assert R.e4m3fn_encode_normal(v / 2, R.FP8_EXP_MIN_NORMAL) is not None
    # each nonzero E2M1 value has exactly (7 - min_exp + 1) shifts into the window
    assert n_ok == 14 * (R.FP8_EXP_MAX - min_exp + 1), n_ok


def test_lut_exhaustive_default():
    _check_lut(R.DEFAULT_MIN_EXP)


def test_lut_exhaustive_normals():
    _check_lut(R.FP8_EXP_MIN_NORMAL)


# --------------------------------------------------------------------- rows
def test_pack_unpack_low_nibble_first():
    nib = list(range(16)) * 4
    packed = R.pack_row(nib)
    assert packed[0] == 0x10 and packed[1] == 0x32  # element 0 in the low nibble
    assert R.unpack_row(packed) == nib


def test_random_rows():
    rng = random.Random(7)
    lut = R.build_lut(R.DEFAULT_MIN_EXP)
    seen = set()
    for cols, count in ((7168, 60), (2048, 160)):
        for _ in range(count):
            spread = rng.randint(0, 9)
            nib, sc = R.synthetic_row(rng, cols, spread)
            q, k = R.reference_encode_row(nib, sc, R.DEFAULT_MIN_EXP, lut)
            _row_exact(nib, sc, q, k, R.DEFAULT_MIN_EXP)
            seen.add(spread)
    assert seen == set(range(10))


def test_adversarial_spreads():
    for cols in (7168, 2048):
        nib, sc = R.adversarial_row(cols, 9)
        q, k = R.reference_encode_row(nib, sc, -5)
        assert _row_exact(nib, sc, q, k, -5) == (-5, 7)
        nib, sc = R.adversarial_row(cols, 10)
        with pytest.raises(R.InexactError):  # spread 10 must fail at min_exp -5
            R.reference_encode_row(nib, sc, -5)
        q, k = R.reference_encode_row(nib, sc, -6)
        assert _row_exact(nib, sc, q, k, -6) == (-6, 7)
        for spread in (11, 12, 20):
            nib, sc = R.adversarial_row(cols, spread)
            with pytest.raises(R.InexactError):  # needs subnormals
                R.reference_encode_row(nib, sc, -6)


def test_zero_rows_and_blocks():
    cols, nb = 2048, 64
    q, k = R.reference_encode_row([0] * cols, [127] * nb)
    assert k == 0 and set(q) == {0}
    q, k = R.reference_encode_row([8] * cols, [0] * nb)  # all -0, odd scales
    assert k == 0 and set(q) == {0}, "-0 must become 0x00"
    # The largest-scale block is all zero: k comes from the nonzero blocks only.
    nib, sc = [0] * cols, [110] * nb
    sc[0] = 200
    nib[R.BLOCK : 2 * R.BLOCK] = [7] * R.BLOCK
    q, k = R.reference_encode_row(nib, sc)
    assert k == (2 + 110 - 127) - 7
    _row_exact(nib, sc, q, k, R.DEFAULT_MIN_EXP)


def test_nan_scale_rejected():
    with pytest.raises(R.InexactError):
        R.reference_encode_row([0] * 64, [127, 255])


# ------------------------------------------------------------- config/names
def _expert_names(pfx="language_model.model", layers=(1, 2, 11, 60), experts=(0, 1, 11, 111, 383)):
    return [
        f"{pfx}.layers.{L}.mlp.experts.{e}.{p}.{s}"
        for L in layers
        for e in experts
        for p in ("gate_proj", "up_proj", "down_proj")
        for s in ("weight", "weight_scale")
    ]


_REST = [
    "language_model.lm_head.weight",
    "language_model.model.embed_tokens.weight",
    "language_model.model.norm.weight",
    "language_model.model.layers.0.mlp.gate_proj.weight",
    "language_model.model.layers.0.mlp.up_proj.weight",
    "language_model.model.layers.0.mlp.down_proj.weight",
    "language_model.model.layers.3.mlp.gate.weight",
    "language_model.model.layers.3.mlp.gate.e_score_correction_bias",
    "language_model.model.layers.3.mlp.shared_experts.gate_proj.weight",
    "language_model.model.layers.3.mlp.shared_experts.up_proj.weight",
    "language_model.model.layers.3.mlp.shared_experts.down_proj.weight",
    "language_model.model.layers.3.self_attn.q_a_proj.weight",
    "language_model.model.layers.3.self_attn.q_b_proj.weight",
    "language_model.model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
    "language_model.model.layers.3.self_attn.kv_b_proj.weight",
    "language_model.model.layers.3.self_attn.o_proj.weight",
    "language_model.model.layers.3.input_layernorm.weight",
    "mm_projector.proj.0.weight",
    "mm_projector.proj.2.weight",
    "vision_tower.encoder.blocks.0.wqkv.weight",
    "vision_tower.encoder.blocks.0.wo.weight",
    "vision_tower.encoder.blocks.0.mlp.fc0.weight",
    "vision_tower.patch_embed.proj.weight",
]

# vLLM-side module names the compressed-tensors config is consulted with: fused Linears are checked shard by shard
# (packed_modules_mapping), RoutedExperts via layer_name + ".0.<proj>".
_VLLM_UNQUANT = [
    "language_model.lm_head",
    "language_model.model.layers.3.self_attn.q_a_proj",
    "language_model.model.layers.3.self_attn.kv_a_proj_with_mqa",
    "language_model.model.layers.3.self_attn.q_b_proj",
    "language_model.model.layers.3.self_attn.kv_b_proj",
    "language_model.model.layers.3.self_attn.o_proj",
    "language_model.model.layers.3.mlp.shared_experts.gate_proj",
    "language_model.model.layers.3.mlp.shared_experts.up_proj",
    "language_model.model.layers.3.mlp.shared_experts.down_proj",
    "language_model.model.layers.0.mlp.gate_proj",
    "language_model.model.layers.0.mlp.up_proj",
    "language_model.model.layers.0.mlp.down_proj",
    "language_model.model.layers.3.mlp.gate",
    "vision_tower.encoder.blocks.0.wqkv",
    "vision_tower.encoder.blocks.0.mlp.fc0",
    "mm_projector.linear_1",
    "mm_projector.linear_2",
]


def test_ct_targets_and_ignore(monkeypatch):
    n_exp, n_other = R.check_targets(_expert_names() + _REST)
    assert n_exp == 4 * 5 * 3 and n_other == len([n for n in _REST if n.endswith(".weight")])
    for L in (1, 60):
        for p in ("gate_proj", "up_proj", "down_proj"):
            mod = f"language_model.model.layers.{L}.mlp.experts.0.{p}"
            assert R.ct_match(mod, R.EXPERT_TARGET)
            assert not any(R.ct_match(mod, t) for t in R.IGNORE)
    for mod in _VLLM_UNQUANT:
        assert not R.ct_match(mod, R.EXPERT_TARGET), mod
        assert any(R.ct_match(mod, t) for t in R.IGNORE), mod
    # The gate must trip on a target that misses an expert projection or reaches past the experts.
    for bad, names in (
        (r"re:.*\.mlp\.experts\.\d+\.(gate_proj|up_proj)$", _expert_names()),
        (r"re:.*_proj$", _REST),
    ):
        monkeypatch.setattr(R, "EXPERT_TARGET", bad)
        with pytest.raises(SystemExit):
            R.check_targets(names)


def test_ep_weight_filter_regex():
    # vllm/model_executor/model_loader/ep_weight_filter.py:_EXPERT_ID_RE
    rx = re.compile(r"\.experts\.(\d+)\.")
    for n in _expert_names():
        assert rx.search(n), n
    for n in _REST:
        assert not rx.search(n), n


def test_quant_config_shape():
    cfg = json.loads(json.dumps(R.build_quant_config()))
    assert cfg["quant_method"] == "compressed-tensors" and cfg["format"] == "float-quantized"
    assert cfg["fp8_dtype_flavor"] == "float8_e4m3fnuz"
    (grp,) = cfg["config_groups"].values()
    assert grp["weights"] == {
        "num_bits": 8,
        "type": "float",
        "symmetric": True,
        "strategy": "channel",
        "dynamic": False,
    }
    assert grp["input_activations"]["strategy"] == "token" and grp["input_activations"]["dynamic"]
    assert grp["targets"] == [R.EXPERT_TARGET]


# ------------------------------------------------------ independent oracle
# Nothing below calls the converter's decoders or encoders: values come from bit fields.
def _o_e2m1(n: int) -> Fraction:
    """OCP MX e2m1: sign bit 3, exponent bits 2:1 (bias 1), mantissa bit 0."""
    s, e, m = n >> 3, (n >> 1) & 3, n & 1
    v = Fraction(m, 2) if e == 0 else (1 + Fraction(m, 2)) * Fraction(2) ** (e - 1)
    return -v if s else v


def _o_e4m3fn(b: int) -> float | None:
    """e4m3fn byte -> float through an IEEE fp32 bit pattern (normals) or m * 2^-9."""
    s, e, m = b >> 7, (b >> 3) & 0xF, b & 7
    if e == 0xF and m == 7:
        return None
    if e == 0:
        return (-1.0 if s else 1.0) * m * 2.0**-9
    return struct.unpack("<f", struct.pack("<I", (s << 31) | ((e - 7 + 127) << 23) | (m << 20)))[0]


# exact value -> byte, Gaudi 2-finite normal codes only (exponent field 1..14).
_O_NORMAL = {Fraction(_o_e4m3fn(b)): b for b in range(256) if 1 <= (b >> 3) & 0xF <= 14}


def _o_encode(v: Fraction) -> int | None:
    """Unique Gaudi 2-finite normal e4m3fn byte for v (0 -> 0x00), or None."""
    return 0x00 if v == 0 else _O_NORMAL.get(v)


def _o_check_row(nib, scales, q, k, min_exp):
    """Oracle verdict on an encoded row: exact, Gaudi 2-finite, normal, floor kept, top element in the 2^7 binade,
    scale an fp32 normal. Returns the exponents."""
    assert -126 <= k <= 127, k
    assert struct.unpack("<f", struct.pack("<f", 2.0**k))[0] == 2.0**k
    sk = Fraction(2) ** k
    exps = []
    for i, n in enumerate(nib):
        b = q[i]
        f = _o_e4m3fn(b)
        assert f is not None and abs(f) <= 240, f"non-finite on Gaudi 2: {b:#x}"
        assert b != 0x80, "-0 emitted"
        e = (b >> 3) & 0xF
        assert e != 0 or b == 0, f"subnormal {b:#x} emitted"
        src = _o_e2m1(n) * Fraction(2) ** (scales[i // R.BLOCK] - 127)
        assert Fraction(f) * sk == src, (i, n, scales[i // R.BLOCK], hex(b), k)
        if b:
            exps.append(e - 7)
    if exps:
        assert max(exps) == 7 and min(exps) >= min_exp, (min(exps), max(exps), min_exp)
    return (min(exps), max(exps)) if exps else (None, None)


def _o_span(nib, scales):
    """Exponent span (max - min) of a row's nonzero elements, from the oracle."""
    es = [math.floor(math.log2(abs(_o_e2m1(n)))) + scales[i // R.BLOCK] - 127 for i, n in enumerate(nib) if n & 7]
    return (max(es) - min(es), max(es)) if es else (None, None)


def test_oracle_matches_module_tables():
    for n in range(16):
        assert Fraction(R.E2M1[n]) == _o_e2m1(n), n
        assert math.copysign(1.0, R.E2M1[n]) == (-1.0 if n & 8 else 1.0), n
    for b in range(256):
        o, r = _o_e4m3fn(b), R.FP8_DECODE[b]
        assert (o is None and math.isnan(r)) or o == r, (hex(b), o, r)
    # Gaudi 2 range: 0x77 = 240 is the largest finite code; exponent field 15 is not finite.
    assert _o_e4m3fn(0x77) == 240.0 and max(abs(Fraction(v)) for v in _O_NORMAL) == 240
    lut_b, lut_ok = R.build_lut(R.DEFAULT_MIN_EXP)
    for n in range(16):
        for d in range(R.D_MIN, R.D_MAX + 1):
            i = n * R.ND + (d - R.D_MIN)
            if lut_ok[i]:
                assert _o_encode(_o_e2m1(n) * Fraction(2) ** d) == lut_b[i], (n, d)


def _row(cols=256, base=120, fill=0):
    nb = cols // R.BLOCK
    return [fill] * cols, [base] * nb


def _put(nib, scales, block, nibble, scale, count=R.BLOCK):
    scales[block] = scale
    nib[block * R.BLOCK : block * R.BLOCK + count] = [nibble] * count


def _expect(nib, scales, ok_at):
    """Encode at min_exp -5 and -6; ok_at is the set of floors that must pass. Whatever passes is re-checked by the
    oracle; whatever fails must be a row the oracle also calls too wide (or with a non-fp32 scale)."""
    span, top = _o_span(nib, scales)
    for me in (-5, -6):
        try:
            q, k = R.reference_encode_row(nib, scales, me)
        except R.InexactError:
            assert me not in ok_at, f"floor {me} must pass (span {span})"
            assert span is None or span > 7 - me or not -126 <= top - 7 <= 127, (span, top, me)
            continue
        assert me in ok_at, f"floor {me} must fail (span {span})"
        _o_check_row(nib, scales, q, k, me)
    return span


def test_adversarial_channels():
    # widest exact row: 6.0 at s=129 down to 0.5 at s=120 (spread 9, span 12)
    nib, sc = _row()
    _put(nib, sc, 0, 7, 129)
    _put(nib, sc, 1, 9, 120)
    assert _expect(nib, sc, {-5, -6}) == 12
    # E8M0 spread 10 is still exact when the top block holds at most 3.0
    nib, sc = _row()
    _put(nib, sc, 0, 13, 130)  # -3.0
    _put(nib, sc, 1, 1, 120)  # 0.5
    assert _expect(nib, sc, {-5, -6}) == 12
    # ... and not when it holds a 4.0: span 13 needs the -6 floor
    nib, sc = _row()
    _put(nib, sc, 0, 6, 130)
    _put(nib, sc, 1, 1, 120)
    assert _expect(nib, sc, {-6}) == 13
    # spread 12, all 0.5: span 12, exact at -5
    nib, sc = _row()
    _put(nib, sc, 0, 1, 132)
    _put(nib, sc, 1, 9, 120)
    assert _expect(nib, sc, {-5, -6}) == 12
    # span 14 (2^-7 after scaling): needs subnormals, fails at both floors
    nib, sc = _row()
    _put(nib, sc, 0, 7, 131)
    _put(nib, sc, 1, 1, 120)
    assert _expect(nib, sc, set()) == 14


def test_top_element_from_lower_scale_block():
    # block 0: s=130 but only 0.5 (2^2); block 1: s=128 with 6.0 (1.5 * 2^3)
    nib, sc = _row()
    _put(nib, sc, 0, 1, 130)
    _put(nib, sc, 1, 7, 128)
    q, k = R.reference_encode_row(nib, sc)
    assert k == 3 - 7, k
    _o_check_row(nib, sc, q, k, R.DEFAULT_MIN_EXP)


def test_zeros_in_largest_scale_block():
    for zero in (0, 8):  # +0 and -0
        nib, sc = _row()
        _put(nib, sc, 0, zero, 254)  # would be 2^127 if it counted
        _put(nib, sc, 1, 15, 120)  # -6.0
        q, k = R.reference_encode_row(nib, sc)
        assert k == (2 + 120 - 127) - 7, (zero, k)
        assert set(q[: R.BLOCK]) == {0x00}
        assert set(q[R.BLOCK : 2 * R.BLOCK]) == {0xF4}  # -192
        _o_check_row(nib, sc, q, k, R.DEFAULT_MIN_EXP)


def test_subnormal_edge():
    # bottom element exactly at 2^-5, 2^-6, 2^-7 after the row scale
    k_row = (1 + 130 - 127) - 7  # 2.0 at s=130 is the top: k = -3
    for bottom, ok_at in ((-5, {-5, -6}), (-6, {-6}), (-7, set())):
        nib, sc = _row()
        _put(nib, sc, 0, 4, 130)  # 2.0
        _put(nib, sc, 1, 3, bottom + k_row + 127)  # 1.5 landing at 1.5 * 2^bottom
        assert _expect(nib, sc, ok_at) == 7 - bottom
        if -6 in ok_at:
            q, k = R.reference_encode_row(nib, sc, -6)
            low = min(b for b in q if b) & 0x7F
            assert low >> 3 == bottom + 7 and low & 7 == 4, hex(low)  # 1.5 * 2^bottom


def test_e8m0_extremes():
    nib, sc = _row()
    _put(nib, sc, 0, 7, 0)  # 6 * 2^-127: row scale 2^-132, not an fp32 normal
    with pytest.raises(R.InexactError):
        R.reference_encode_row(nib, sc)
    nib, sc = _row()
    _put(nib, sc, 0, 7, 254)  # 6 * 2^127: row scale 2^122
    q, k = R.reference_encode_row(nib, sc)
    assert k == 122
    _o_check_row(nib, sc, q, k, R.DEFAULT_MIN_EXP)
    nib, sc = _row()
    _put(nib, sc, 0, 0, 0)  # an all-zero block may carry any finite scale
    _put(nib, sc, 1, 2, 120)
    _o_check_row(nib, sc, *R.reference_encode_row(nib, sc), R.DEFAULT_MIN_EXP)
    nib, sc = _row()
    _put(nib, sc, 0, 0, 255)  # NaN scale, even on an all-zero block
    with pytest.raises(R.InexactError):
        R.reference_encode_row(nib, sc)


def test_single_nonzero_element():
    nib, sc = _row()
    nib[37] = 9  # -0.5 at s=120, everything else +0
    q, k = R.reference_encode_row(nib, sc)
    assert q[37] == 0xF0 and sum(1 for b in q if b) == 1, hex(q[37])  # -128
    _o_check_row(nib, sc, q, k, R.DEFAULT_MIN_EXP)


def test_gaudi2_half_load_replay():
    """Replay gaudi_weight_wrapper with scale adjustment on: q' = e4m3fn(q * 0.5), scale' = 2 * scale.

    Every row encoded at the -5 floor survives exactly and stays normal. At -6 the halved smallest elements become
    e4m3 subnormals (exact on the CPU, since e2m1 only yields mantissas 0 and 4), and Gaudi 2's handling of FP8
    subnormals is unverified, so -6 is only for scale adjustment off."""
    rng = random.Random(11)
    for _ in range(200):
        nib, sc = R.synthetic_row(rng, 512, rng.randint(0, 9))
        q, k = R.reference_encode_row(nib, sc)
        for i, b in enumerate(q):
            h = _o_encode(Fraction(_o_e4m3fn(b)) / 2)
            assert h is not None, (i, hex(b))
            assert Fraction(_o_e4m3fn(h)) * Fraction(2) ** (k + 1) == _o_e2m1(nib[i]) * Fraction(2) ** (
                sc[i // R.BLOCK] - 127
            )
    nib, sc = _row()
    _put(nib, sc, 0, 7, 130)  # 6.0: top exponent 2+3 = 5, k = -2
    _put(nib, sc, 1, 3, 119)  # 1.5 at 2^-8 -> 1.5 * 2^-6 after the row scale
    q, k = R.reference_encode_row(nib, sc, -6)
    assert k == -2
    lows = [b for b in q if b and (b & 0x7F) >> 3 == 1]
    assert lows and all(_o_encode(Fraction(_o_e4m3fn(b)) / 2) is None for b in lows)
    # ... it is an exact e4m3fn subnormal (6 * 2^-9), not a rounding
    assert all(Fraction(_o_e4m3fn(b)) / 2 == Fraction(6, 512) for b in lows)
    # The default floor must refuse that row, not emit it.
    assert R.DEFAULT_MIN_EXP == -5
    with pytest.raises(R.InexactError):
        R.reference_encode_row(nib, sc)


def test_property_fuzz_span_rule():
    """The encoder passes iff the oracle's element span fits the window (7 - floor)."""
    rng = random.Random(7)
    passed = failed = 0
    for _ in range(1500):
        cols = rng.choice((64, 128, 256))
        nb = cols // R.BLOCK
        base = rng.randint(100, 140)
        sc = [base + rng.randint(0, rng.randint(0, 14)) for _ in range(nb)]
        nib = [rng.randrange(16) if rng.random() < 0.6 else rng.choice((0, 8)) for _ in range(cols)]
        span, top = _o_span(nib, sc)
        me = rng.choice((-5, -6))
        want = span is None or span <= 7 - me
        try:
            q, k = R.reference_encode_row(nib, sc, me)
            got = True
        except R.InexactError:
            got = False
        assert got == want, (span, me, sc)
        if got:
            _o_check_row(nib, sc, q, k, me)
            passed += 1
        else:
            failed += 1
    assert passed > 250 and failed > 250, (passed, failed)


# --------------------------------------------------------------- real rows
def test_real_rows():
    root = os.environ.get("KIMI_MXFP4_DIR")
    if not root:
        pytest.skip("set KIMI_MXFP4_DIR to a converted MXFP4 checkpoint to check real rows")
    assert os.path.exists(os.path.join(root, "model.safetensors.index.json")), f"no checkpoint index in {root}"
    src = R.Source(root)
    pfx, fused = src.fused_layers()
    assert pfx == "language_model.model" and sorted(fused) == list(range(1, 61))
    assert set(fused.values()) == {384}
    lut = R.build_lut(R.DEFAULT_MIN_EXP)
    ks, lo_all, rows_done = [], [], 0
    for L in (1, 30, 60):
        base = f"{pfx}.layers.{L}.mlp.experts.routed_experts"
        for wn, starts in (("w13", (0, 2048, 4064)), ("w2", (0, 7136))):
            E, Rr, C = src.meta(f"{base}.{wn}_weight")[1]
            nb = src.meta(f"{base}.{wn}_weight_scale")[1][2]
            for e in (0, 191, 383):
                for r0 in starts:
                    wb = src.expert_rows(f"{base}.{wn}_weight", e, r0, r0 + 32)
                    sb = src.expert_rows(f"{base}.{wn}_weight_scale", e, r0, r0 + 32)
                    for i in range(32):
                        nib = R.unpack_row(wb[i * C : (i + 1) * C])
                        sc = list(sb[i * nb : (i + 1) * nb])
                        q, k = R.reference_encode_row(nib, sc, R.DEFAULT_MIN_EXP, lut)
                        lo, _ = _row_exact(nib, sc, q, k, R.DEFAULT_MIN_EXP)
                        ks.append(k)
                        if lo is not None:
                            lo_all.append(lo)
                        rows_done += 1
    print(f"{rows_done} real rows exact; k in [{min(ks)}, {max(ks)}], lowest emitted exponent {min(lo_all)}")
