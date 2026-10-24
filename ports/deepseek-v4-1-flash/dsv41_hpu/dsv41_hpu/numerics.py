# SPDX-License-Identifier: Apache-2.0
"""Numerics shared by every DeepSeek-V4.1 module: fake quantization, RMSNorm, RoPE.

Fake quantization reproduces DeepSeek's reference kernels (inference/kernel.py in the checkpoint: act_quant_kernel,
fp4_quant_kernel) value for value, using fp32 arithmetic and int32 bit manipulation only:

  * no torch float8/float4 casts: Gaudi 2's native FP8 is e4m3fnuz (max 240), not the e4m3fn grid (max 448) the
    model was trained against, and there is no FP4;
  * no torch.frexp/ldexp, whose HPU lowering is unverified.

Rounding is round-half-to-even on the quotient by a power-of-two step, which is exactly RNE on the mantissa for
these grids (the quotient's LSB is the mantissa LSB). tests/test_numerics.py checks every helper against torch's
CPU float8_e4m3fn cast and an exhaustive E2M1 nearest-value search.

Switches (all off by default; each is bit-identical to the off path, and only reduces graph nodes in decode, which
is launch-bound on Gaudi):
  DSV41_FQ_FASTBITS=1  power-of-two helpers as one masked int op each instead of shift/mask/subtract chains.
  DSV41_LEAN_OPS=1     mean of squares as sum x (1/n) (vllm-gaudi lowers mean to 4 nodes), where() with scalar fills
                       instead of zeros_like/full_like tensors, single-tile outputs written directly.
  DSV41_ROPE_FUSED=1   rope_tail as x * C + swap_pairs(x) * S over the whole row instead of slicing the tail, splitting
                       even/odd lanes, stacking and concatenating. Inputs of at most ROPE_FUSED_MAX elements only
                       (decode and short prompts): on a full 4096-token chunk the full-width fp32 form moves more
                       data than the tail-only one.
"""

from __future__ import annotations

import math
import os

import torch

FASTBITS = os.environ.get("DSV41_FQ_FASTBITS", "0") == "1"
LEAN = os.environ.get("DSV41_LEAN_OPS", "0") == "1"
ROPE_FUSED = os.environ.get("DSV41_ROPE_FUSED", "0") == "1"
ROPE_FUSED_MAX = 1 << 24  # 1024 tokens x 32 heads x 512

FP8_MAX = 448.0  # e4m3fn
FP4_MAX = 6.0  # e2m1
_INV_448 = 1.0 / 448.0
_INV_6 = 1.0 / 6.0
_EXP_MASK = 0x7F800000


def mean_sq(xf: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    """mean(x^2) over the last dim (fp32 x)."""
    if LEAN:
        return xf.square().sum(-1, keepdim=keepdim) * (1.0 / xf.shape[-1])
    return xf.square().mean(-1, keepdim=keepdim)


# ----------------------------------------------------------------- bit helpers
def _exp_floor(a: torch.Tensor) -> torch.Tensor:
    """floor(log2(a)) for non-negative fp32 `a` (normal values; 0 gives -127)."""
    return ((a.view(torch.int32) >> 23) & 0xFF) - 127


def _pow2(e: torch.Tensor) -> torch.Tensor:
    """2**e as fp32 for int32 e in [-126, 127]."""
    return ((e + 127) << 23).view(torch.float32)


def _pow2_floor(a: torch.Tensor) -> torch.Tensor:
    """2**floor(log2(a)) for non-negative fp32 a; 0.0 for zero and subnormals (same as _pow2(_exp_floor(a)))."""
    return (a.view(torch.int32) & _EXP_MASK).view(torch.float32)


def _step(a: torch.Tensor, emin: int, shift: int) -> torch.Tensor:
    """2**(max(floor(log2 a), emin) - shift) for non-negative fp32 a: the grid's quantization step."""
    if FASTBITS:
        return _pow2_floor(a).clamp_min(2.0**emin) * (2.0**-shift)
    return _pow2(_exp_floor(a).clamp(min=emin) - shift)


def pow2_ceil(v: torch.Tensor) -> torch.Tensor:
    """2**ceil(log2(v)) for positive normal fp32 v: the reference fast_round_scale."""
    if FASTBITS:
        # a nonzero mantissa carries into the exponent
        return ((v.view(torch.int32) + 0x7FFFFF) & _EXP_MASK).view(torch.float32)
    bits = v.view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127
    return _pow2(e + ((bits & 0x7FFFFF) != 0).to(torch.int32))


def round_e4m3fn(v: torch.Tensor) -> torch.Tensor:
    """Nearest e4m3fn value (RNE), as fp32. Caller keeps |v| <= 448."""
    a = v.abs()
    step = _step(a, -6, 3)  # subnormals share step 2**-9
    return torch.copysign(torch.round(a / step) * step, v)


def round_e2m1(v: torch.Tensor) -> torch.Tensor:
    """Nearest e2m1 value (RNE), as fp32. Caller keeps |v| <= 6."""
    a = v.abs()
    step = _step(a, 0, 1)  # 0.5 below 2, then 1, then 2
    return torch.copysign(torch.round(a / step) * step, v)


def _blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    n = x.shape[-1]
    assert n % block == 0, (n, block)
    return x.float().reshape(*x.shape[:-1], n // block, block)


# ------------------------------------------------------------- fake quantizers
def fake_fp8_ue8m0(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """act_quant(x, block, "ue8m0", inplace=True): FP8 e4m3fn with power-of-two per-block scales, dequantized back
    to x.dtype. The window KV cache format."""
    xb = _blocks(x, block)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = pow2_ceil(amax * _INV_448)
    y = round_e4m3fn((xb / s).clamp(-FP8_MAX, FP8_MAX)) * s
    return y.reshape(x.shape).to(x.dtype)


def fake_fp4_ue8m0(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """fp4_act_quant(x, block, inplace=True): E2M1 with power-of-two scales. The indexer's q and k."""
    xb = _blocks(x, block)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(6.0 * 2.0**-126)
    s = pow2_ceil(amax * _INV_6)
    y = round_e2m1((xb / s).clamp(-FP4_MAX, FP4_MAX)) * s
    return y.reshape(x.shape).to(x.dtype)


# Exact rounding of a RATIO without trusting division: Gaudi's fp32 division is not correctly rounded
# (reciprocal-multiply), and with an arbitrary e4m3 divisor a 1-ulp error flips FP4/FP8 rounding at grid midpoints
# (3.6% of compressed-KV elements on the first HPU run). Midpoints of these grids have <= 5 significant bits and the
# divisors <= 4, so midpoint * divisor is EXACT in fp32 and every decision compares the dividend against it.
_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def round_e2m1_ratio(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """e2m1(clamp(x / s, -6, 6)) (RNE) as a level value, for s > 0, without dividing:
    level index = #(|x| > mid_k * s) + #(|x| == mid_k * s and k odd)  (ties to the even code)."""
    a = x.abs()
    idx = torch.zeros_like(a, dtype=torch.int32)
    for k, m in enumerate(_E2M1_MIDS):
        t = s * m
        idx = idx + (a > t).to(torch.int32)
        if k % 2 == 1:
            idx = idx + (a == t).to(torch.int32)
    levels = torch.tensor(_E2M1_LEVELS, dtype=torch.float32, device=x.device)
    return torch.copysign(levels[idx], x)


def _e4m3_ulp(c: torch.Tensor) -> torch.Tensor:
    """Spacing of the e4m3fn grid just above non-negative grid value c."""
    return _step(c, -6, 3)


def round_e4m3fn_ratio(a: torch.Tensor, d: float) -> torch.Tensor:
    """e4m3fn(a / d) (RNE) for a >= 0 and a small integer d, exactly: take the approximate quotient's grid value,
    then settle it against the exact midpoints (both neighbours) by comparing a with midpoint * d."""
    c = round_e4m3fn((a / d).clamp(max=FP8_MAX))
    for _ in range(2):
        up = _e4m3_ulp(c)
        nxt = (c + up).clamp(max=FP8_MAX)
        # the spacing just below c halves when c is a power of two above the subnormals
        p2 = _pow2_floor(c) if FASTBITS else _pow2(_exp_floor(c))
        down = torch.where((c == p2) & (c > 2.0**-6), up * 0.5, up)
        prv = (c - down).clamp(min=0)
        hi, lo = (c + nxt) * 0.5 * d, (prv + c) * 0.5 * d
        c_even = torch.remainder(c / up, 2) == 0
        go_up = (a > hi) | ((a == hi) & ~c_even & (nxt > c))
        go_dn = (a < lo) | ((a == lo) & ~c_even & (prv < c))
        c = torch.where(go_up, nxt, torch.where(go_dn, prv, c))
    return c


def fake_fp4_e4m3(x: torch.Tensor, block: int = 16) -> torch.Tensor:
    """fp4_act_quant(x, 16, inplace=True, scale_dtype=e4m3fn): E2M1 with e4m3fn scales, the compressed-KV format.
    s = e4m3fn(amax / 6); values e2m1(x / s) * s. Both roundings are decided exactly (round_*_ratio), not by
    device division."""
    xb = _blocks(x, block)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(6.0 * 2.0**-9)
    s = round_e4m3fn_ratio(amax, FP4_MAX)
    y = round_e2m1_ratio(xb, s) * s
    return y.reshape(x.shape).to(x.dtype)


# ------------------------------------------------------------------- RMSNorm
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The reference RMSNorm: fp32 statistics, weight applied in fp32, cast back. eps is 1e-20 for this model, which
    is only meaningful in fp32."""
    xf = x.float()
    xf = xf * torch.rsqrt(mean_sq(xf) + eps)
    return (weight.float() * xf).to(x.dtype)


# ---------------------------------------------------------------------- RoPE
def rope_inv_freq(
    dim: int, base: float, original_seq_len: int, factor: float, beta_fast: float, beta_slow: float
) -> torch.Tensor:
    """Per-pair inverse frequencies, fp32, with YaRN exactly as the reference's precompute_freqs_cis: applied
    whenever original_seq_len > 0, no mscale."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim))
    if original_seq_len > 0:

        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32, device="cpu") - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def rope_tables(max_pos: int, inv_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [max_pos, dim/2] fp32, built on the CPU: HPU sin/cos is wrong for |angle| > 196608, and angles here
    reach ~1e6."""
    ang = torch.outer(torch.arange(max_pos, dtype=torch.float32, device="cpu"), inv_freq.cpu().float())
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate the last dim of x as adjacent (even, odd) pairs, GPT-J style, like the reference's view_as_complex
    rotation. cos/sin broadcast against x[..., ::2]. fp32 inside, x.dtype out."""
    xf = x.float()
    x0, x1 = xf[..., 0::2], xf[..., 1::2]
    if inverse:
        sin = -sin
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    return torch.stack([y0, y1], dim=-1).flatten(-2).to(x.dtype)


def rope_tail(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rd: int, inverse: bool = False) -> torch.Tensor:
    """RoPE on the last `rd` dims of x only (V4.1: 448 NoPE + 64 RoPE of 512)."""
    if ROPE_FUSED and x.numel() <= ROPE_FUSED_MAX:
        return rope_tail_fused(x, cos, sin, rd, inverse)
    return torch.cat([x[..., :-rd], apply_rope(x[..., -rd:], cos, sin, inverse)], dim=-1)


def rope_tail_fused(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rd: int, inverse: bool = False
) -> torch.Tensor:
    """DSV41_ROPE_FUSED: rope_tail as x * C + swap(x) * S over the full width. Pair k of the tail:
    y_2k = x_2k * cos_k + x_2k+1 * (-sin_k), y_2k+1 = x_2k+1 * cos_k + x_2k * sin_k; NoPE lanes: x * 1 + x' * 0."""
    W = x.shape[-1]
    nope = W - rd
    s = -sin if inverse else sin
    c2 = torch.stack([cos, cos], dim=-1).flatten(-2)  # [..., rd]
    s2 = torch.stack([-s, s], dim=-1).flatten(-2)
    C = torch.nn.functional.pad(c2, (nope, 0), value=1.0)  # one pad node each, no fill + cat
    S = torch.nn.functional.pad(s2, (nope, 0), value=0.0)
    xf = x.float()
    sw = xf.unflatten(-1, (W // 2, 2)).flip(-1).flatten(-2)  # swap each (even, odd) pair
    return (xf * C + sw * S).to(x.dtype)
