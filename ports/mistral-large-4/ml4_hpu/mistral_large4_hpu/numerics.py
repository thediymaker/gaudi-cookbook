"""Numerics shared by every Mistral Large 4 HPU module.

Every helper reproduces the PR's rounding (vLLM PR #64534, the reference for the served model; HF where the PR has
no opinion) value for value, using fp32 arithmetic and int32 bit manipulation only:

  * no torch float8 casts on the device: Gaudi2's native FP8 is e4m3fnuz (max 240) and reads e4m3fn codes
    256..448 as inf/NaN; the indexer's FP8 q/k are FAKE-quantized here (e4m3fn grid, power-of-two scale, values
    kept in bf16, which holds every e4m3fn value exactly);
  * every bf16 rounding the reference makes is made here at the same place: a bf16 op on CUDA/CPU is "fp32
    arithmetic, one round to bf16". Where a gate needs that rounding bit for bit it is done on the bits
    (round_bf16): the HPU graph compiler removes a .to(bf16).float() pair between two fp32 ops, which keeps the
    intermediate in fp32 (measured: SSMax lambda differed in 27% of elements with plain casts, 0 with round_bf16);
  * nothing that depends on the device's transcendental functions: log(pos + 1) and the RoPE cos/sin come from
    CPU fp32 tables (rope.py), never from HPU log/sin/cos.

checks each helper against the PR's torch semantics on the CPU (FP8 via
torch's float8_e4m3fn cast) and on one HPU.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

# ML4_FAKE_QUANT=0 turns the indexer's FP8 fake-quant into the identity: the HF reference (and the oracle's
# --no-fp8-indexer control). TESTS ONLY.
FAKE_QUANT = os.environ.get("ML4_FAKE_QUANT", "1") != "0"

FP8_MAX = 448.0  # e4m3fn
FP8_EPS = 1e-10  # per_token_group_quant_fp8's amax floor
INDEX_GROUP = 128  # the indexer quantizes each 128-element head (q) / row (k) as one group
_INV_448 = 1.0 / 448.0
_EXP_MASK = 0x7F800000
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


# ----------------------------------------------------------------- bit helpers (from dsv41_hpu.numerics)
def _exp_floor(a: torch.Tensor) -> torch.Tensor:
    """floor(log2(a)) for non-negative fp32 `a` (normal values; 0 gives -127)."""
    return ((a.view(torch.int32) >> 23) & 0xFF) - 127


def _pow2(e: torch.Tensor) -> torch.Tensor:
    """2**e as fp32 for int32 e in [-126, 127]."""
    return ((e + 127) << 23).view(torch.float32)


# ML4_NUMERICS_RW=1: the same values with fewer device nodes. Integer / power-of-two forms of
# pow2_ceil, pow2_inv and round_e4m3fn, with their constants as persistent 0-d device tensors (no constant nodes in
# the graph). Bit-identical to the forms below on the HPU (exhaustive bf16 sweeps + 6 served shapes, eager and graph);
# -44 us per indexer layer at 1 row.
NUMERICS_RW = os.environ.get("ML4_NUMERICS_RW", "0") == "1"
_CONST: dict = {}


def _c(v, dtype, like: torch.Tensor) -> torch.Tensor:
    k = (v, dtype, like.device)
    t = _CONST.get(k)
    if t is None:
        t = _CONST[k] = torch.tensor(v, dtype=dtype, device=like.device)
    return t


_RW_CONSTS = (
    (0x7FFFFF, torch.int32),
    (0x7F800000, torch.int32),
    (0x7F000000, torch.int32),
    (0x7FFF, torch.int32),
    (16, torch.int32),
    (1, torch.int32),
    (-65536, torch.int32),
    (2.0**-6, torch.float32),
    (0.125, torch.float32),
)


def rw_init(device) -> None:
    """Build the constants once, outside any HPU graph capture (call before the decode / prompt graphs are captured:
    a constant first made inside a dry-run capture would be one graph's internal tensor)."""
    like = torch.empty(0, device=device)
    for v, dt in _RW_CONSTS:
        _c(v, dt, like)


def pow2_ceil(v: torch.Tensor) -> torch.Tensor:
    """2**ceil(log2(v)) for positive normal fp32 v: the reference's exp2(ceil(log2(scale_raw))) (ue8m0 scale).
    Equal to the log2 form for every amax * (1/448) of a bf16 amax: the only bf16 mantissa whose product lands
    within an fp32 ulp of a power of two is 224 (= 7 * 32), and that product rounds to the power exactly."""
    if NUMERICS_RW:  # round the bits up to the next exponent (carry when the mantissa is nonzero), keep it
        return ((v.view(torch.int32) + _c(0x7FFFFF, torch.int32, v)) & _c(0x7F800000, torch.int32, v)).view(
            torch.float32
        )
    bits = v.view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127
    return _pow2(e + ((bits & 0x7FFFFF) != 0).to(torch.int32))


def pow2_inv(s: torch.Tensor) -> torch.Tensor:
    """1 / s for a power of two s (exact; no device division)."""
    if NUMERICS_RW:  # exponent 254 - E of a power of two
        return (_c(0x7F000000, torch.int32, s) - s.view(torch.int32)).view(torch.float32)
    return _pow2(-_exp_floor(s))


def round_e4m3fn(v: torch.Tensor) -> torch.Tensor:
    """Nearest e4m3fn value (RNE), as fp32. Caller keeps |v| <= 448."""
    a = v.abs()
    if NUMERICS_RW:  # step = max(2**floor(log2 a), 2**-6) / 8, power-of-two float ops only
        p = (a.view(torch.int32) & _c(0x7F800000, torch.int32, a)).view(torch.float32)
        step = torch.maximum(p, _c(2.0**-6, torch.float32, a)) * _c(0.125, torch.float32, a)
        return torch.copysign(torch.round(a / step) * step, v)
    step = _pow2(_exp_floor(a).clamp(min=-6) - 3)  # subnormals share step 2**-9
    return torch.copysign(torch.round(a / step) * step, v)


def round_bf16(x: torch.Tensor) -> torch.Tensor:
    """fp32 x rounded to the nearest bf16 value (RNE), kept in fp32: integer ops on the bits (as the PR's Triton
    _round_bf16), so no graph compiler can drop the rounding. Finite inputs."""
    bits = x.float().contiguous().view(torch.int32)
    if NUMERICS_RW:  # same ops, constants as device tensors
        return (
            (bits + (_c(0x7FFF, torch.int32, bits) + ((bits >> _c(16, torch.int32, bits)) & _c(1, torch.int32, bits))))
            & _c(-65536, torch.int32, bits)
        ).view(torch.float32)
    bits = (bits + (0x7FFF + ((bits >> 16) & 1))) & -65536
    return bits.view(torch.float32)


# ------------------------------------------------------------- fake quantizers
def fake_fp8_ue8m0(x: torch.Tensor, group: int = INDEX_GROUP) -> torch.Tensor:
    """per_token_group_quant_fp8(x, group, use_ue8m0=True) dequantized back to x.dtype: per group of `group`
    consecutive elements, s = 2^ceil(log2(max(amax, 1e-10) / 448)), q = e4m3fn(clamp(x / s, +-448)), value q * s.
    The indexer's q (per token and head) and k (per token) format. Exact in bf16 (4 significant bits)."""
    if not FAKE_QUANT:
        return x
    n = x.shape[-1]
    assert n % group == 0, (n, group)
    xb = x.float().reshape(*x.shape[:-1], n // group, group)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(FP8_EPS)
    s = pow2_ceil(amax * _INV_448)
    y = round_e4m3fn((xb * pow2_inv(s)).clamp(-FP8_MAX, FP8_MAX)) * s
    return y.reshape(x.shape).to(x.dtype)


def fp8_ue8m0_reference(x: torch.Tensor, group: int = INDEX_GROUP) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU only: the PR's per_token_group_quant_fp8(..., use_ue8m0=True) in torch (Triton kernel
    _per_token_group_quant_fp8 written out): returns (e4m3fn codes as uint8, fp32 scales). The gate's reference."""
    n = x.shape[-1]
    xb = x.float().reshape(*x.shape[:-1], n // group, group)
    amax = torch.clamp_min(xb.abs().amax(-1, keepdim=True), FP8_EPS)
    s = torch.exp2(torch.ceil(torch.log2(amax * (1.0 / FP8_MAX))))
    q = torch.clamp(xb / s, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.view(torch.uint8).reshape(x.shape), s.squeeze(-1)


# ------------------------------------------------------------------- norms
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, r5: bool = False, dual=False) -> torch.Tensor:
    """vLLM RMSNorm.forward_native / HF MistralLarge4RMSNorm: fp32 statistics, cast to x.dtype, THEN the weight
    multiply (one more bf16 rounding)."""
    from . import r5k

    # ML4_R5_RMS (r5=True call sites only): the tail in one kernel, same bits. Not where the output feeds a .float()
    # consumer (the indexer's layer_norm, the logits): the graph compiler drops the native bf16 round trip there.
    if r5 and r5k.rms_on(x, x.shape[-1]) and x.is_contiguous():
        return r5k.rms(x, weight, eps)
    if dual:  # ML4_R6_DUAL (full-indexer input norm, dual = "rope" | "nope"): bf16 out + the fp32 product
        from . import r6k  # the indexer's layer_norm gets

        if r6k.dual_on(x, dual):
            return r6k.rms_dual(x, weight, eps)
    xf = x.float()
    y = round_bf16(xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps))
    return (y * weight.float()).to(x.dtype)


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """LayerNorm with bias (the indexer's q_input_norm and k_norm, eps 1e-6): fp32 statistics and affine, one
    rounding to x.dtype (vLLM LayerNorm with fp32 parameters; the bf16-parameter q_input_norm differs only in the
    affine's rounding inside F.layer_norm, which is fp32 on every backend)."""
    return F.layer_norm(x.float(), (x.shape[-1],), weight.float(), bias.float(), eps).to(x.dtype)


# ------------------------------------------------------------------- SSMax
def ssmax_lambda(log_pos: torch.Tensor, s: torch.Tensor, b: torch.Tensor, r5: bool = False) -> torch.Tensor:
    """lambda[t, h] = s_h * log(pos_t + 1) + b_h in bf16, each op rounded (SSMaxScaling.forward; the fused kernel's
    _ssmax_scale): log_pos [...] fp32 from the CPU table, s/b [H] -> bf16 [..., H]. A bf16 product of two bf16
    values is exact in fp32, so fp32-then-round is the bf16 op."""
    from . import r5k

    if r5 and r5k.ssl_on(log_pos):  # ML4_R5_SSL (decode call sites that pass r5): one kernel
        return r5k.ssl(log_pos, s, b)
    lp = round_bf16(log_pos).unsqueeze(-1)
    prod = round_bf16(lp * s.to(torch.bfloat16).float())
    return (prod + b.to(torch.bfloat16).float()).to(torch.bfloat16)


def ssmax_apply(q: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """q [..., H, D] bf16 times lambda [..., H] bf16, one rounding (the product is exact in fp32)."""
    return (q.float() * lam.float().unsqueeze(-1)).to(q.dtype)


# ----------------------------------------------------------------- logits
def softcap(z: torch.Tensor, cap: float = 30.0) -> torch.Tensor:
    """30 * tanh(z / 30) in fp32 (final_logit_softcapping)."""
    return torch.tanh(z.float() * (1.0 / cap)) * cap


# ------------------------------------------------------------ MXFP4 (CPU refs)
def mxfp4_dequant(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """CPU reference: w [..., N, K/2] uint8 (OCP E2M1, low nibble = even k), s [..., N, K/32] uint8 E8M0 (bias 127)
    -> fp32 [..., N, K] (exact)."""
    lut = torch.tensor(E2M1, dtype=torch.float32)
    v = torch.stack([lut[(w & 0x0F).long()], lut[(w >> 4).long()]], -1).flatten(-2)
    return v * torch.pow(2.0, s.float() - 127.0).repeat_interleave(32, -1)
