# SPDX-License-Identifier: Apache-2.0
"""CPU tests for convert/: exactness of the MXFP4 to FP8 re-encoding and the compressed-tensors config.

The pure-Python checks run everywhere; the TorchCodec and BF16 checks skip without torch. An independent oracle
(e2m1 from its bit fields, e4m3fn through an fp32 bit pattern) checks every emitted byte, so a bug shared with
fp8_codec's lookup table cannot pass.
"""

from __future__ import annotations

import random
import re
import struct
from fractions import Fraction

import pytest

import fp8_codec as C
import repack as R

P = "language_model.model"
FUSED = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"]}


def oracle_e2m1(n: int) -> Fraction:
    s, e, m = n >> 3, (n >> 1) & 3, n & 1
    v = Fraction(m, 2) if e == 0 else (1 + Fraction(m, 2)) * Fraction(2) ** (e - 1)
    return -v if s else v


def oracle_e4m3fn(b: int) -> float | None:
    s, e, m = b >> 7, (b >> 3) & 0xF, b & 7
    if e == 0xF and m == 7:
        return None
    if e == 0:
        return (-1.0 if s else 1.0) * m * 2.0**-9
    return struct.unpack("<f", struct.pack("<I", (s << 31) | ((e - 7 + 127) << 23) | (m << 20)))[0]


def oracle_check_row(nib, scales, q, k, min_exp=C.DEFAULT_MIN_EXP):
    """Exact, finite on Gaudi 2, normal, floor kept, top element in the 2^7 binade."""
    exps = []
    for i, n in enumerate(nib):
        b = q[i]
        f = oracle_e4m3fn(b)
        assert f is not None and abs(f) <= 240 and b != 0x80, hex(b)
        e = (b >> 3) & 0xF
        assert e != 0 or b == 0, f"subnormal {b:#x}"
        assert e != 0xF, "exponent-15 code is not finite on Gaudi 2"
        assert Fraction(f) * Fraction(2) ** k == oracle_e2m1(n) * Fraction(2) ** (scales[i // 32] - 127)
        if b:
            exps.append(e - 7)
    if exps:
        assert max(exps) == 7 and min(exps) >= min_exp
    else:
        assert k == 0


def test_lut_agrees_with_oracle():
    lut_b, lut_ok = C.build_lut()
    for n in range(16):
        for d in range(C.D_MIN, C.D_MAX + 1):
            i = n * C.ND + (d - C.D_MIN)
            v = oracle_e2m1(n) * Fraction(2) ** d
            if lut_ok[i]:
                assert Fraction(oracle_e4m3fn(lut_b[i])) == v
                assert v == 0 or C.DEFAULT_MIN_EXP <= C.floor_log2(abs(v)) <= 7
            else:
                assert v != 0


@pytest.mark.parametrize("cols", [6144, 3072])
def test_reference_rows_exact(cols):
    rng, lut = random.Random(7), C.build_lut()
    rows = [C.synthetic_row(rng, cols, rng.randint(0, 9)) for _ in range(12)]
    rows += [C.adversarial_row(cols, 9), ([0] * cols, [127] * (cols // 32)), ([8] * cols, [3] * (cols // 32))]
    for nib, sc in rows:
        q, k = C.reference_encode_row(nib, sc, lut=lut)
        oracle_check_row(nib, sc, q, k)


def test_too_wide_or_nan_rows_refuse():
    for spread in (10, 11, 20):
        nib, sc = C.adversarial_row(3072, spread)
        with pytest.raises(C.InexactError):
            C.reference_encode_row(nib, sc)
    nib, sc = C.synthetic_row(random.Random(1), 3072, 2)
    sc[3] = C.E8M0_NAN
    with pytest.raises(C.InexactError):
        C.reference_encode_row(nib, sc)


def test_pack_roundtrip():
    nib = [random.Random(3).randrange(16) for _ in range(64)]
    assert C.unpack_row(C.pack_row(nib)) == nib


def test_shared_expert_values_are_exact_bf16():
    """Every e2m1 value at every finite E8M0 scale is an fp32 whose low 16 bits are zero."""
    for s in range(0, 255):
        for n in range(16):
            v = oracle_e2m1(n) * Fraction(2) ** (s - 127)
            if v == 0 or not (2.0**-126 <= abs(v) < 2.0**128):
                continue
            bits = struct.unpack("<I", struct.pack("<f", float(v)))[0]
            assert Fraction(float(v)) == v and bits & 0xFFFF == 0


# ------------------------------------------------------------------------------------------------ CT config
def _re(name, t):
    return re.match(t[3:], name) is not None if t.startswith("re:") else name == t


def _targeted(name, targets):
    if any(_re(name, t) for t in targets):
        return True
    proj = name.split(".")[-1]
    return proj in FUSED and all(any(_re(name.replace(proj, s), t) for t in targets) for s in FUSED[proj])


def _ignored(name, ignore):
    proj = name.split(".")[-1]
    if proj in FUSED:
        verdicts = {any(_re(name.replace(proj, s), t) for t in ignore) for s in FUSED[proj]}
        assert len(verdicts) == 1, f"fused {name}: shards disagree on ignore"
        return verdicts.pop()
    return any(_re(name, t) for t in ignore)


EXPERTS = [
    f"{P}.layers.{L}.block_sparse_moe.experts.{e}.{w}.weight"
    for L in (3, 31, 59)
    for e in (0, 9, 127)
    for w in ("w1", "w2", "w3")
]
REST = [
    "language_model.lm_head.weight",
    f"{P}.embed_tokens.weight",
    f"{P}.layers.0.mlp.gate_proj.weight",
    f"{P}.layers.2.mlp.down_proj.weight",
    f"{P}.layers.3.block_sparse_moe.gate.weight",
    f"{P}.layers.3.block_sparse_moe.shared_experts.gate_proj.weight",
    f"{P}.layers.59.block_sparse_moe.shared_experts.down_proj.weight",
    f"{P}.layers.3.self_attn.q_proj.weight",
    f"{P}.layers.3.self_attn.index_k_proj.weight",
    "multi_modal_projector.linear_1.weight",
    "vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
]
# Module names vLLM consults the config with (fused modules, the RoutedExperts probe, the VL mapper's prefixes).
VLLM_UNQUANTIZED = [
    f"{P}.layers.3.self_attn.qkv_proj",
    f"{P}.layers.3.self_attn.o_proj",
    f"{P}.layers.3.block_sparse_moe.shared_experts.gate_up_proj",
    f"{P}.layers.3.block_sparse_moe.gate",
    f"{P}.layers.0.mlp.gate_up_proj",
    "language_model.lm_head",
    "vision_tower.vision_model.encoder.layers.0.self_attn.qkv_proj",
    "vision_tower.vision_model.encoder.layers.0.fc1",
    "vision_tower.multi_modal_projector.linear_1",
    "vision_tower.patch_merge_mlp.linear_2",
]


def test_quant_config_targets_routed_experts_only():
    qc = R.build_quant_config()
    (group,) = qc["config_groups"].values()
    targets, ignore = group["targets"], qc["ignore"]
    assert group["weights"]["strategy"] == "channel" and group["input_activations"]["strategy"] == "token"
    assert qc["fp8_dtype_flavor"] == "float8_e4m3fnuz"
    n_exp, n_other = R.check_targets(EXPERTS + REST)
    assert (n_exp, n_other) == (len(EXPERTS), len(REST))
    for L in (3, 59):
        for p in ("gate_proj", "up_proj", "down_proj"):
            probe = f"{P}.layers.{L}.block_sparse_moe.experts.0.{p}"
            assert _targeted(probe, targets) and not _ignored(probe, ignore)
    for mod in VLLM_UNQUANTIZED:
        assert not _targeted(mod, targets) and _ignored(mod, ignore), mod


def test_check_targets_rejects_a_stray_match():
    # A tensor the converter does not treat as a routed expert but the FP8 target would quantize.
    with pytest.raises(SystemExit):
        R.check_targets([f"{P}.layers.3.block_sparse_moe.experts.0.gate_proj.weight"])


# ---------------------------------------------------------------------------------------------------- torch
def test_torch_codec_matches_reference():
    torch = pytest.importorskip("torch")
    rng, lut = random.Random(11), C.build_lut()
    codec = C.TorchCodec()
    for cols in (6144, 3072):
        rows = [C.synthetic_row(rng, cols, rng.randint(0, 9), base=rng.randint(110, 130)) for _ in range(8)]
        rows.append(C.adversarial_row(cols, 9))
        w = torch.tensor([list(C.pack_row(n)) for n, _ in rows], dtype=torch.uint8).unsqueeze(0)
        s = torch.tensor([sc for _, sc in rows], dtype=torch.uint8).unsqueeze(0)
        q, scale, _ = codec.encode(w, s)
        for r, (nib, sc) in enumerate(rows):
            ref, k = C.reference_encode_row(nib, sc, lut=lut)
            assert q[0, r].tolist() == ref and float(scale[0, r, 0]) == 2.0**k
        if hasattr(torch, "float8_e4m3fn"):
            dec = q.view(torch.float8_e4m3fn).double() * scale.double()
            mx = torch.tensor([[float(oracle_e2m1(n)) for n in nib] for nib, _ in rows], dtype=torch.float64)
            mx = mx * torch.tensor([[2.0 ** (sc[i // 32] - 127) for i in range(cols)] for _, sc in rows])
            assert torch.equal(dec[0], mx)


def test_bf16_codec_exact():
    torch = pytest.importorskip("torch")
    nib, sc = C.synthetic_row(random.Random(5), 256, 6, base=118)
    w = torch.tensor([list(C.pack_row(nib))], dtype=torch.uint8)
    s = torch.tensor([sc], dtype=torch.uint8)
    codec = R.Bf16Codec()
    out = torch.frombuffer(bytearray(codec.encode(w, s)), dtype=torch.int16).view(torch.bfloat16)
    assert torch.equal(out.double(), codec.decode64(w, s).flatten())
