"""YaRN + LeRoPE rotary tables in fp32 and the interleaved (GPT-J) rotation.

inv_freq_i = exp(alpha_i) * yarn_i. The PR multiplies the extrapolated and the interpolated
frequencies by exp(alpha) before the YaRN blend (PR common/rope.py:56-74); HF multiplies the
blended frequency (HF:117-121). Both are the same in exact arithmetic; this follows the PR.
No mscale: cos/sin are unscaled (apply_scale false). One table per model, shared by every MLA
layer and every indexer. NoPE layers never rotate.
"""

from __future__ import annotations

import math

import torch

F32 = torch.float32


def yarn_correction_range(beta_fast: float, beta_slow: float, dim: int, base: float, orig_max: int):
    def cdim(rot):
        return (dim * math.log(orig_max / (rot * 2 * math.pi))) / (2 * math.log(base))

    low, high = math.floor(cdim(beta_fast)), math.ceil(cdim(beta_slow))
    return max(low, 0), min(high, dim - 1)


def yarn_inv_freq(
    dim: int,
    base: float,
    factor: float,
    orig_max: int,
    beta_fast: float,
    beta_slow: float,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """[dim/2] fp32 inverse frequencies, YaRN blend, optional LeRoPE alpha (PR form)."""
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=F32) / dim)
    ext = 1.0 / pos_freqs
    inter = 1.0 / (factor * pos_freqs)
    if alpha is not None:
        e = torch.exp(alpha.to(F32))
        ext, inter = ext * e, inter * e
    low, high = yarn_correction_range(beta_fast, beta_slow, dim, base, orig_max)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=F32) - low) / (high - low), 0, 1)
    m = 1 - ramp  # extrapolation weight
    return inter * (1 - m) + ext * m


def inv_freq_from_cfg(cfg, alpha: torch.Tensor | None) -> torch.Tensor:
    return yarn_inv_freq(
        cfg.qk_rope_head_dim,
        cfg.rope_theta,
        cfg.rope_factor,
        cfg.rope_original_max,
        cfg.rope_beta_fast,
        cfg.rope_beta_slow,
        alpha,
    )


def cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor, dtype: torch.dtype = F32):
    """cos, sin [T, dim/2] for 0-based absolute positions. fp32 tables;
    pass dtype=bf16 to reproduce the PR's bf16 cos_sin_cache."""
    freqs = positions.to(F32)[:, None] * inv_freq[None, :].to(F32)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (x[2i], x[2i+1]) by angle i; output keeps the interleaved layout (PR).
    x [T, ..., dim]; cos/sin [T, dim/2]. Computed in fp32, returned in fp32."""
    shp = x.shape
    xf = x.to(F32).reshape(*shp[:-1], shp[-1] // 2, 2)
    c = cos.to(F32).reshape(cos.shape[0], *([1] * (x.dim() - 2)), -1)
    s = sin.to(F32).reshape(sin.shape[0], *([1] * (x.dim() - 2)), -1)
    x0, x1 = xf[..., 0], xf[..., 1]
    return torch.stack([x0 * c - x1 * s, x1 * c + x0 * s], -1).reshape(shp)


def rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Half-split (NeoX) rotation: the wrong layout, used only by the --rope-noninterleaved control."""
    shp = x.shape
    h = shp[-1] // 2
    xf = x.to(F32)
    c = cos.to(F32).reshape(cos.shape[0], *([1] * (x.dim() - 2)), -1)
    s = sin.to(F32).reshape(sin.shape[0], *([1] * (x.dim() - 2)), -1)
    x0, x1 = xf[..., :h], xf[..., h:]
    return torch.cat([x0 * c - x1 * s, x1 * c + x0 * s], -1)
