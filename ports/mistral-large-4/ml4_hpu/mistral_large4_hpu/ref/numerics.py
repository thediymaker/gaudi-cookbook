"""Numerics of the Mistral Large 4 reference stack (pure torch, no vllm import).

Two compute modes, chosen per call by `dt` (the activation dtype):
  * fp32: every op in fp32 (the HF `model.float()` comparison);
  * bf16: the engine's rounding points. Activations are stored in bf16 between ops,
    GEMMs take bf16 values and accumulate in fp32 with one bf16 rounding of the output,
    norms compute in fp32, and softmax runs in fp32.

Follows the PR (vLLM #64534, bd4fa2633) where PR and HF differ:
the indexer's FP8 e4m3 fake-quant with power-of-two (ue8m0) scales, and SSMax computed
with bf16 rounding of each product and sum.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

F32 = torch.float32
BF16 = torch.bfloat16
FP8_MAX = 448.0
MXFP4_BLOCK = 32
# E2M1 code -> value (index = nibble, MSB = sign). OCP order: element 0 in the LOW nibble.
E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=F32, device="cpu"
)
_LUT = torch.stack([E2M1_VALUES[torch.arange(256) & 15], E2M1_VALUES[torch.arange(256) >> 4]], -1)


def rnd(x: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """Store rounding: cast to the activation dtype (no-op in fp32 mode)."""
    return x if x.dtype == dt else x.to(dt)


def linear(x: torch.Tensor, w: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """y = x @ w.T with fp32 accumulation, rounded once to `dt`. `w` is [out, in]."""
    return rnd(x.to(F32) @ w.to(F32).t(), dt)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float, dt: torch.dtype) -> torch.Tensor:
    """HF/vLLM RMSNorm: fp32 normalize, cast to the input dtype, then multiply by the weight."""
    xf = x.to(F32)
    y = rnd(xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps), dt)
    return rnd(y * w.to(dt), dt)


def layernorm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float, dt: torch.dtype) -> torch.Tensor:
    """LayerNorm with bias, computed in fp32, stored in `dt` (indexer q_input_norm, k_norm)."""
    return rnd(F.layer_norm(x.to(F32), (x.shape[-1],), w.to(F32), b.to(F32), eps), dt)


def silu_mul(g: torch.Tensor, u: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    return rnd(F.silu(g.to(F32)) * u.to(F32), dt)


def log_positions(positions: torch.Tensor) -> torch.Tensor:
    """log(pos + 1) in fp32: SSMax's n is the query's own 1-based position."""
    return torch.log(positions.to(F32) + 1.0)


def ssmax_scale(log_pos: torch.Tensor, s: torch.Tensor, b: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """lambda[t, h] = s_h * log(pos_t + 1) + b_h, [T, H].

    bf16: each input, the product and the sum are rounded to bf16 (PR ops/ssmax.py:28-37,
    HF casts to the query dtype). fp32: plain fp32."""
    lp, sv, bv = rnd(log_pos, dt), rnd(s.to(F32), dt), rnd(b.to(F32), dt)
    return rnd(rnd(lp[:, None] * sv[None, :], dt) + bv[None, :], dt)


def apply_ssmax(q: torch.Tensor, lam: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """q [T, H, D] * lambda [T, H]."""
    return rnd(q * lam[:, :, None], dt)


def softcap(z: torch.Tensor, cap: float) -> torch.Tensor:
    """30 * tanh(z / 30) in fp32 (PR logits_processor.py:113-116, HF:1152-1155)."""
    z = z.to(F32)
    return torch.tanh(z / cap) * cap


# ------------------------------------------------------------------ FP8 e4m3, ue8m0 scales
def ue8m0_scale(amax: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(max(amax, 1e-10) / 448)): the PR's per-group power-of-two scale
    (sparse_attn_indexer.py:263-265, ops/indexer.py:116)."""
    return torch.exp2(torch.ceil(torch.log2(amax.to(F32).clamp_min(1e-10) * (1.0 / FP8_MAX))))


def fp8_fakequant_ue8m0(x: torch.Tensor, group: int) -> torch.Tensor:
    """Quantize the last dim to FP8 e4m3fn per `group` elements with a ue8m0 scale and
    dequantize back to fp32. Bit-exact to (x_fp8.float() * scale) of the PR's quantizers:
    division by a power of two is exact, the e4m3fn cast is round-to-nearest-even."""
    shp = x.shape
    g = x.to(F32).reshape(*shp[:-1], shp[-1] // group, group)
    sc = ue8m0_scale(g.abs().amax(-1, keepdim=True))
    q = (g / sc).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(F32)
    return (q * sc).reshape(shp)


def fp8_block_dequant(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 [R, C] weight with a per-block fp32 scale [ceil(R/b), ceil(C/b)] -> fp32.
    The block size is inferred from the shapes (128 in the real checkpoint, 32 in tiny)."""
    R, C = w.shape
    sr, sc = scale.shape
    br, bc = math.ceil(R / sr), math.ceil(C / sc)
    s = scale.to(F32).repeat_interleave(br, 0)[:R].repeat_interleave(bc, 1)[:, :C]
    return w.to(F32) * s


# ------------------------------------------------------------------------------- MXFP4
def mxfp4_dequant(u8: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    """OCP MXFP4 -> fp32. u8 [R, C/2] (element 2i in the LOW nibble), scale_u8 [R, C/32] E8M0.
    Every value is e2m1 * 2^k, exact in bf16 for normal scale ranges."""
    R = u8.shape[0]
    v = F.embedding(u8.long() if u8.dtype != torch.int32 else u8, _LUT).view(R, -1, MXFP4_BLOCK)
    return (v * torch.exp2(scale_u8.to(F32) - 127.0)[..., None]).view(R, -1)


def _round_e2m1(a: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 magnitude for 0 <= a <= 6, round-half-even on the mantissa code."""
    step = torch.where(a < 2.0, 0.5, torch.where(a < 4.0, 1.0, 2.0))
    return torch.round(a / step) * step


def act_mxfp4(x: torch.Tensor) -> torch.Tensor:
    """W4A4 emulation (`act_mxfp4` switch): fake-quantize activations to MXFP4.

    Per 32-element group: Quark `scale_calculation_mode: even` exponent (floor(log2 amax) - 2,
    +1 when amax / 2^e >= 7; same rule as muse_spark_hpu.numerics.act_mxfp4, derived from
    AMD's bytes), then E2M1 round-half-even. Returns x.dtype."""
    shp = x.shape
    g = x.to(F32).reshape(*shp[:-1], shp[-1] // MXFP4_BLOCK, MXFP4_BLOCK)
    amax = g.abs().amax(-1, keepdim=True)
    e = torch.floor(torch.log2(amax.clamp(min=1e-38))) - 2.0
    e = e + (amax / torch.exp2(e) >= 7.0).to(e.dtype)
    e = torch.where(amax == 0, torch.zeros_like(e), e).clamp(-127.0, 127.0)
    sc = torch.exp2(e)
    q = torch.copysign(_round_e2m1((g / sc).abs().clamp(max=6.0)), g) * sc
    return q.reshape(shp).to(x.dtype)


def mxfp4_quantize_rtn(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 [R, C] -> (u8 [R, C/2], e8m0 [R, C/32]) with the `even` scale rule and RNE.
    Used by make_syn.py to write syn8l's experts in AMD's byte layout."""
    R, C = w.shape
    g = w.to(F32).reshape(R, C // MXFP4_BLOCK, MXFP4_BLOCK)
    amax = g.abs().amax(-1, keepdim=True)
    e = torch.floor(torch.log2(amax.clamp(min=1e-38))) - 2.0
    e = e + (amax / torch.exp2(e) >= 7.0).to(e.dtype)
    e = torch.where(amax == 0, torch.zeros_like(e), e).clamp(-127.0, 127.0)
    a = _round_e2m1((g / torch.exp2(e)).abs().clamp(max=6.0))
    code = torch.where(a < 2.0, a * 2, torch.where(a < 4.0, a + 2, a / 2 + 4)).to(torch.uint8)
    code = code | ((g < 0) & (a > 0)).to(torch.uint8) << 3
    code = code.reshape(R, C // 2, 2)
    u8 = code[..., 0] | (code[..., 1] << 4)
    return u8.contiguous(), (e[..., 0] + 127).to(torch.uint8).contiguous()
