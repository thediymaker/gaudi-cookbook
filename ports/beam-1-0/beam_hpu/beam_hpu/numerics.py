# SPDX-License-Identifier: Apache-2.0
"""Beam's elementwise numerics: norms, residuals, the attention output gate, SSMax, SwiGLU, softcap.

Pure torch, no vLLM import: the same functions run on CPU (tests) and on HPU (the model).

Rounding points follow Hugging Face transformers' modeling_beam.py and the kernels of vLLM PR #62074,
which agree:
  * RMSNorm: fp32 x * rsqrt(mean(x^2) + eps) * w, ONE rounding to the activation dtype (input,
    post-attention, pre-FFN, post-FFN, final, and the per-head q/k norms);
  * SSMax (NoPE layers): q * (1 + (pos + 1) / 2048) ** 0.1 in fp32, one rounding;
  * output gate: g + b rounded, sigmoid rounded, o * sigmoid rounded (temperature 1 and floor 0 are
    exact no-ops);
  * depth-scaled residual: x * (1 / sqrt(n)) rounded, r * sqrt((n - 1) / n) rounded, sum rounded;
  * SwiGLU: silu(g) * u in fp32, one rounding;
  * final logits: lm_head in bf16, softcap cap * tanh(z / cap) in fp32.

On Gaudi 2 in lazy mode the graph compiler removes a cast pair fp32 -> bf16 -> fp32 inside one graph
(the bf16 rounding disappears) and fuses a following .float() into an FP8 GEMM. With
BEAM_EXACT_ROUNDING=1 every intermediate rounding goes through rnd() / store() and every upcast of a
nominally bf16 tensor through up(), which on HPU round to bf16 with integer ops (round to nearest
even on the fp32 bit pattern) that the compiler cannot remove. BEAM_EXACT_ROUNDING=0 (the recipe's
setting) uses plain casts: about one bf16 ulp of difference at the dropped points, smaller graphs and
a faster decode step; the correctness gates pass in both modes.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

F32 = torch.float32
BF16 = torch.bfloat16

EXACT_ROUNDING = os.environ.get("BEAM_EXACT_ROUNDING", "1") == "1"


def rne_bf16_bits(t: torch.Tensor) -> torch.Tensor:
    """fp32 t rounded to the nearest bf16 (ties to even), kept as fp32, with integer ops."""
    i = t.contiguous().view(torch.int32)
    i = (i + (0x7FFF + ((i >> 16) & 1))) & -65536
    return i.view(F32)


def _exact(t: torch.Tensor) -> bool:
    return EXACT_ROUNDING and t.device.type == "hpu"


def rnd(t: torch.Tensor, dtype: torch.dtype = BF16) -> torch.Tensor:
    """fp32 tensor -> fp32 holding the dtype-rounded value (an intermediate rounding point)."""
    t = t.to(F32)
    if dtype == BF16 and _exact(t):
        return rne_bf16_bits(t)
    return t.to(dtype).to(F32)


def store(t: torch.Tensor, dtype: torch.dtype | None = BF16) -> torch.Tensor:
    """A rounding store: the dtype tensor holds the correctly rounded value even when the compiler
    later drops this cast in front of an upcast (None keeps fp32)."""
    if dtype is None:
        return t
    if dtype == BF16 and _exact(t):
        return rne_bf16_bits(t.to(F32)).to(BF16)
    return t.to(dtype)


def up(x: torch.Tensor) -> torch.Tensor:
    """Upcast to fp32; a bf16 tensor is re-rounded on HPU so that a value the compiler kept in fp32
    still enters as its bf16 value."""
    if x.dtype == BF16 and _exact(x):
        return rne_bf16_bits(x.to(F32))
    return x.to(F32)


def rms_affine(
    x: torch.Tensor, w: torch.Tensor | None, eps: float, out_dtype: torch.dtype | None = BF16
) -> torch.Tensor:
    """x * rsqrt(mean(x^2) + eps) * w over the last dim, fp32 math, one rounding."""
    xf = up(x)
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if w is not None:
        y = y * w.to(F32)
    return store(y, out_dtype)


def depth_coeffs(layer_id: int, is_attention: bool) -> tuple[float, float]:
    """(x_scale, r_scale) of the residual add: n = 2L for the attention add and 2L + 1 for the MLP add,
    L = layer_id + 1."""
    n = 2 * (layer_id + 1) + (0 if is_attention else 1)
    return 1.0 / math.sqrt(n), math.sqrt((n - 1) / n)


def scaled_residual(
    x: torch.Tensor, r: torch.Tensor, coeffs: tuple[float, float] | None, dtype: torch.dtype = BF16
) -> torch.Tensor:
    """h = round(round(x * xs) + round(r * rs)) (plain round(x + r) without coeffs)."""
    if coeffs is None:
        return store(up(x) + up(r), dtype)
    xs, rs = coeffs
    return store(rnd(up(x) * xs, dtype) + rnd(up(r) * rs, dtype), dtype)


def boundary(
    x: torch.Tensor,
    residual: torch.Tensor,
    w_post: torch.Tensor | None,
    eps: float,
    coeffs: tuple[float, float] | None,
    dtype: torch.dtype = BF16,
) -> torch.Tensor:
    """Sandwich post-norm plus depth-scaled residual add: the new residual stream h (bf16)."""
    p = rms_affine(x, w_post, eps, dtype) if w_post is not None else store(up(x), dtype)
    return scaled_residual(p, residual, coeffs, dtype)


def ssmax_scale(pos: torch.Tensor, offset: int = 1) -> torch.Tensor:
    """SSMax root10 query scale [T] fp32: (1 + (pos + offset) / 2048) ** 0.1."""
    return torch.exp(0.1 * torch.log1p((pos.to(F32) + offset) / 2048.0))


def apply_ssmax(q: torch.Tensor, pos: torch.Tensor, offset: int = 1, out_dtype: torch.dtype = BF16) -> torch.Tensor:
    """q [T, H, D] bf16 -> round(q * ssmax_scale(pos))."""
    return store(up(q) * ssmax_scale(pos, offset)[:, None, None], out_dtype)


def output_gate(
    o: torch.Tensor,
    g: torch.Tensor,
    bias: torch.Tensor | None,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype = BF16,
) -> torch.Tensor:
    """o, g [T, H*D] -> o * sigmoid(g + bias_h) with the reference's roundings, [T, H*D]."""
    T = o.shape[0]
    gf = up(g.reshape(T, num_heads, head_dim))
    if bias is not None:
        gf = rnd(gf + up(bias)[None, :, None], dtype)
    s = rnd(torch.sigmoid(gf), dtype)
    return store(up(o.reshape(T, num_heads, head_dim)) * s, dtype).reshape(T, num_heads * head_dim)


def silu_mul(gate_up: torch.Tensor, dtype: torch.dtype = BF16) -> torch.Tensor:
    """[.., 2I] -> silu(g) * u in fp32, one rounding."""
    g, u = gate_up.chunk(2, dim=-1)
    return store(F.silu(up(g)) * up(u), dtype)


def softcap(z: torch.Tensor, cap: float | None) -> torch.Tensor:
    """cap * tanh(z / cap) in fp32 on the bf16 logits; fp32 z without a cap."""
    zf = up(z)
    if cap is None:
        return zf
    return torch.tanh(zf / cap) * cap
