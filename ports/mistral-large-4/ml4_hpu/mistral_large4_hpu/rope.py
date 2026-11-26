"""RoPE for Mistral Large 4: interleaved (GPT-J) YaRN with LeRoPE, tables on the CPU in fp32.

Ground truth: the PR's MistralRotaryEmbedding (vllm/models/mistral/common/rope.py, on DeepseekScalingRotaryEmbedding):
  pf_i   = 10000^(2i/64), i = 0..31
  ext_i  = exp(alpha_i) / pf_i            int_i = exp(alpha_i) / (2048 pf_i)        (alpha: lerope.alpha [32], bf16)
  ramp   = yarn_linear_ramp_mask(low 3, high 16) on the UNSCALED frequencies (beta_fast 32, beta_slow 1, orig 512)
  inv_i  = int_i (1 - m_i) + ext_i m_i,   m = 1 - ramp
  cos/sin[p, i] = cos/sin(p * inv_i)      fp32, NO mscale (apply_scale false: attention_factor 1.0)
Every op is written in the PR's order, so the table is bit-equal to the PR's fp32 cache before its bf16 cast (and
to the oracle's, which builds it the same way). One global table: every MLA layer and every indexer uses it; NoPE
layers never rotate. The DSpark draft uses plain YaRN (alpha=None), its own table.

Rotation: pairs (x_2i, x_2i+1) by angle p * inv_i, in fp32 from the fp32 table, one rounding to bf16 (the fused
kernels' rounding). MLA rope dims are the LAST 64 of a q head ([nope | rope]) and k_pe; the indexer's are
the FIRST 64 ([rope | nope]).

The tables are built once per process on the CPU and moved to the device once (materialize()), never inside a
forward an HPU graph could record: HPU sin/cos is wrong for |angle| > ~1e5..2e5 (vllm-gaudi #1817; angles here reach
p * exp(alpha_0) ~ 1e6 at the 1M context), and log(p + 1) for SSMax comes from the same kind of CPU table.
"""

from __future__ import annotations

import math
import os

import torch

ROPE_DIM = 64
# Controls for the gates (tests only): each is a deliberately wrong variant that must FAIL its gate.
#   ML4_FAULT_ROPE=neox      rotate halves (i, i+32) instead of interleaved pairs
#   ML4_FAULT_ROPE=noalpha   LeRoPE off (plain YaRN tables for the target model)
#   ML4_FAULT_ROPE=hpu       cos/sin computed on the device (the vllm-gaudi #1817 trap)
FAULTS = {"rope": os.environ.get("ML4_FAULT_ROPE", "")}


def yarn_correction_range(beta_fast: float, beta_slow: float, dim: int, base: float, orig: int) -> tuple[int, int]:
    """yarn_find_correction_range(truncate=True): (3, 16) for Mistral Large 4."""

    def corr(rot):
        return (dim * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(base))

    low, high = math.floor(corr(beta_fast)), math.ceil(corr(beta_slow))
    return max(low, 0), min(high, dim - 1)


def inv_freq(
    dim: int = ROPE_DIM,
    base: float = 10000.0,
    orig: int = 512,
    factor: float = 2048.0,
    beta_fast: float = 32,
    beta_slow: float = 1,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """[dim/2] fp32 inverse frequencies, MistralRotaryEmbedding._compute_inv_freq op for op."""
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
    ext = 1.0 / pos_freqs
    inter = 1.0 / (factor * pos_freqs)
    if alpha is not None:
        s = torch.exp(alpha.detach().float().cpu().to(pos_freqs))
        ext = ext * s
        inter = inter * s
    low, high = yarn_correction_range(beta_fast, beta_slow, dim, base, orig)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float) - low) / (high - low), 0, 1)
    mask = 1 - ramp
    return inter * (1 - mask) + ext * mask


def tables(max_pos: int, inv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [max_pos, dim/2] fp32 on the CPU (DeepseekScalingRotaryEmbedding._compute_cos_sin_cache with
    mscale 1.0: t * inv as an einsum, cos/sin of the fp32 angle)."""
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv.cpu().float())
    return freqs.cos(), freqs.sin()


def log_pos_table(max_pos: int) -> torch.Tensor:
    """log(p + 1) [max_pos] fp32 on the CPU (the model's log_positions = log(positions.float() + 1))."""
    return torch.log(torch.arange(max_pos, dtype=torch.float32) + 1)


_TABLES: dict = {}


class RopeCache:
    """One table set (cos, sin, log(p+1)) per (max_pos, alpha bytes), shared by every layer that asks for it.
    Not an nn.Module: nothing here is a parameter or a buffer that a checkpoint or an HPU graph could touch."""

    def __init__(
        self,
        max_pos: int,
        alpha: torch.Tensor | None = None,
        base: float = 10000.0,
        orig: int = 512,
        factor: float = 2048.0,
        beta_fast: float = 32,
        beta_slow: float = 1,
    ):
        if FAULTS["rope"] == "noalpha":
            alpha = None
        a = None if alpha is None else alpha.detach().to(torch.bfloat16).cpu()
        akey = None if a is None else bytes(a.view(torch.int16).numpy().tobytes())
        self.key = (int(max_pos), akey, float(base), int(orig), float(factor), float(beta_fast), float(beta_slow))
        self.max_pos = int(max_pos)
        if (self.key, "cpu") not in _TABLES:
            inv = inv_freq(ROPE_DIM, base, orig, factor, beta_fast, beta_slow, a)
            cos, sin = tables(max_pos, inv)
            _TABLES[(self.key, "cpu")] = (cos, sin, log_pos_table(max_pos), inv)

    @property
    def inv(self) -> torch.Tensor:
        return _TABLES[(self.key, "cpu")][3]

    def materialize(self, device) -> None:
        dt = torch.device(device).type
        k = (self.key, dt)
        hpu_fault = FAULTS["rope"] == "hpu" and dt != "cpu"
        if hpu_fault:
            k = (self.key, dt, "fault")
        if k in _TABLES:
            self._dev_key = k
            return
        self._dev_key = k
        cos, sin, lp, inv = _TABLES[(self.key, "cpu")]
        if hpu_fault:
            # the trap this module exists to avoid: angles and cos/sin evaluated on the device
            t = torch.arange(self.max_pos, dtype=torch.float32, device=device)
            ang = t.unsqueeze(1) * inv.to(device).unsqueeze(0)
            _TABLES[k] = (torch.cos(ang), torch.sin(ang), lp.to(device), inv)
            return
        _TABLES[k] = (cos.to(device), sin.to(device), lp.to(device), inv)

    def at(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(cos [..., 32], sin [..., 32], log(p+1) [...]) at positions pos (padding -1 reads position 0)."""
        k = getattr(self, "_dev_key", None)
        if pos.device.type != "cpu":
            assert k is not None and k[1] == pos.device.type, "RopeCache.materialize(device) first"
            t = _TABLES[k]
        else:
            t = _TABLES[(self.key, "cpu")]
        p = pos.clamp(min=0)
        return t[0][p], t[1][p], t[2][p]


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the last dim (64) of x as interleaved pairs: y_2i = x_2i cos - x_2i+1 sin, y_2i+1 = x_2i+1 cos + x_2i sin;
    fp32 from the fp32 table, one rounding to x.dtype. cos/sin [..., 32] broadcast against x[..., ::2]."""
    xf = x.float()
    if FAULTS["rope"] == "neox":
        h = xf.shape[-1] // 2
        x0, x1 = xf[..., :h], xf[..., h:]
        return torch.cat([x0 * cos - x1 * sin, x1 * cos + x0 * sin], -1).to(x.dtype)
    x0, x1 = xf[..., 0::2], xf[..., 1::2]
    return torch.stack([x0 * cos - x1 * sin, x1 * cos + x0 * sin], -1).flatten(-2).to(x.dtype)


def rope_tail(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rd: int = ROPE_DIM, r5: bool = False
) -> torch.Tensor:
    """MLA layout [nope | rope]: rotate the last rd dims."""
    from . import r5k

    if r5 and r5k.rope_on(x, rd):  # ML4_R5_ROPE (decode call sites that pass r5): one kernel, same bits
        return r5k.rope(x, cos, sin, x.shape[-1] - rd)
    return torch.cat([x[..., :-rd], rotate(x[..., -rd:], cos, sin)], -1)


def rope_head(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rd: int = ROPE_DIM, r5: bool = False
) -> torch.Tensor:
    """Indexer layout [rope | nope]: rotate the first rd dims."""
    from . import r5k

    if r5 and r5k.rope_on(x, rd):  # ML4_R5_ROPE (decode call sites that pass r5): one kernel, same bits
        return r5k.rope(x, cos, sin, 0)
    return torch.cat([rotate(x[..., :rd], cos, sin), x[..., rd:]], -1)
