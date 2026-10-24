# SPDX-License-Identifier: Apache-2.0
"""OCP MXFP4 encode and decode for the GLM-5.x checkpoints.

Layout (what vllm-gaudi's MXFP4 MoE op and the kernels read): E2M1 elements packed two per byte, element 0 in the
low nibble; one E8M0 scale byte (2^(e - 127)) per block of 32 elements along the input (last) dimension.

Two encoders:

``quantize_mxfp4`` (the Quark rule)
    The shared exponent is floor(log2(amax)) - 2, bumped by one when amax / 2^X >= 7: of the two candidate exponents
    it takes the one that represents amax more closely (clamping to 6 costs |r - 6|, bumping costs |r - 8|; they
    cross at r = 7). Elements round to the nearest E2M1 value, ties to the even code; a value that rounds to zero
    keeps no sign. This reproduces amd/GLM-5.2-MXFP4 bit for bit from the BF16 weights (checked on the routed experts
    of GLM-5.2 layers 3 and 29), and it is the encoder of the GLM-5.3 checkpoint. The plain floor rule of the OCP
    spec matches only about 80% of AMD's scale bytes and has a larger error.

``quantize_mxfp4_floor`` (the plain OCP floor rule)
    The shared exponent is floor(log2(amax)) - 2 without the bump, elements are clamped to +-6 and round half down,
    and negative values that round to zero keep the sign bit. amd/GLM-5.2-MXFP4 leaves the MTP layer (78) in BF16;
    the served GLM-5.2 checkpoint quantized its routed experts with this encoder, and the GLM-5.2 converter keeps it
    so that its output matches that checkpoint byte for byte.
"""

from __future__ import annotations

import torch

BLOCK = 32
# E2M1 code -> value (index = nibble, bit 3 = sign)
E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
)
E2M1_MAX = 6.0
# midpoints between neighbouring positive E2M1 values
_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
# for a value exactly on midpoint i: True when the upper neighbour has the even code (codes 0, 2, 4, 6 are even)
_TIE_UP = torch.tensor([False, True, False, True, False, True, False])
_EMAX_ELEM = 2  # 6.0 = 1.5 * 2^2


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., N/2] -> codes [..., N] (int64), element 0 in the low nibble."""
    lo = (packed & 0x0F).to(torch.long)
    hi = ((packed >> 4) & 0x0F).to(torch.long)
    return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def dequant_mxfp4(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """MXFP4 -> float32. weight uint8 [R, C/2], scale uint8 [R, C/32]."""
    vals = E2M1_VALUES.to(weight.device)[unpack_nibbles(weight)]
    s = torch.pow(2.0, scale.to(torch.float32) - 127.0).repeat_interleave(BLOCK, dim=-1)[..., : vals.shape[-1]]
    return vals * s


def _round_half_even(mag: torch.Tensor) -> torch.Tensor:
    """|x| on the scaled grid -> 3-bit magnitude code, ties to the even code."""
    mid = _MID.to(mag.device)
    code = torch.bucketize(mag, mid, right=True)
    # Every midpoint times 4 is an integer, so only values with integral mag * 4 can be ties. Ties are frequent (the
    # scale is a power of two and bf16 has few mantissa bits), but checking every element against all 7 midpoints
    # dominates the run time, so only the candidates are checked.
    maybe = torch.frac(mag * 4.0) == 0
    if bool(maybe.any()):
        idx = maybe.nonzero(as_tuple=True)
        sub = mag[idx]
        on_tie = (sub.unsqueeze(-1) == mid).any(dim=-1)
        tie_at = torch.bucketize(sub, mid, right=False).clamp(max=mid.numel() - 1)
        down = on_tie & ~_TIE_UP.to(mag.device)[tie_at]
        code[idx] = torch.where(down, code[idx] - 1, code[idx])
    return code.clamp(0, 7)


def _pack(nib: torch.Tensor) -> torch.Tensor:
    return (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8).contiguous()


def quantize_mxfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """float [R, C] (C % 32 == 0) -> (uint8 [R, C/2], E8M0 uint8 [R, C/32]) with the Quark rule."""
    rows, cols = w.shape
    assert cols % BLOCK == 0, f"last dim {cols} is not a multiple of {BLOCK}"
    g = w.to(torch.float32).reshape(rows, cols // BLOCK, BLOCK)
    amax = g.abs().amax(dim=-1)
    exp = torch.floor(torch.log2(amax.clamp(min=1e-38))) - _EMAX_ELEM
    exp = exp + (amax / torch.pow(2.0, exp) >= 7.0).to(exp.dtype)
    exp = torch.where(amax == 0, torch.zeros_like(exp), exp).clamp(-127.0, 127.0)
    scaled = g / torch.pow(2.0, exp).unsqueeze(-1)
    # floor(log2()) can land a hair low just under a power of two: saturate instead of wrapping
    code = _round_half_even(scaled.abs().clamp(max=E2M1_MAX))
    sign = (torch.signbit(scaled) & (code != 0)).to(torch.long) << 3
    nib = (sign | code).reshape(rows, cols).to(torch.uint8)
    return _pack(nib), (exp + 127.0).to(torch.uint8)


def quantize_mxfp4_floor(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """float [R, C] (C % 32 == 0) -> (uint8 [R, C/2], E8M0 uint8 [R, C/32]) with the plain OCP floor rule."""
    rows, cols = w.shape
    assert cols % BLOCK == 0, f"last dim {cols} is not a multiple of {BLOCK}"
    f = w.to(torch.float32).reshape(rows, cols // BLOCK, BLOCK)
    amax = f.abs().amax(dim=-1)
    e = torch.where(amax > 0, torch.floor(torch.log2(amax)) - _EMAX_ELEM, torch.zeros_like(amax)).clamp(-127.0, 127.0)
    q = (f / torch.pow(2.0, e).unsqueeze(-1)).clamp(-E2M1_MAX, E2M1_MAX)
    idx = torch.bucketize(q.abs().contiguous(), _MID.to(q.device))
    nib = torch.where(q < 0, idx + 8, idx).to(torch.uint8).reshape(rows, cols)
    return _pack(nib), (e + 127.0).to(torch.uint8)
