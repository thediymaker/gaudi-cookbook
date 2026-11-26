"""ML4_FP8_LINEARS=1 (off by default): the non-expert linears that come from the checkpoint's block FP8 (attention wq /
wo, the indexer's wq_b, the shared experts, the layer-0 dense FFN) are kept on the device as 8-bit codes with one
power-of-two scale per 128 x 128 block, and dequantized to BF16 right before each matmul.
Saves half their BF16 bytes (2.86 GiB per card at TP8) for the KV pool.

The weights are the same values. The loader writes BF16 = e4m3fn code x 2^k (exact). Here each block
gets the scale s = 2^ceil(log2(amax / 240)), and code = W / s is a power-of-two shift of the checkpoint's code, on
the e4m3 grid and within Gaudi2's +-240 range (where Gaudi2's e4m3 and e4m3fn are the same bytes). Only values that
the shift moves below e4m3's normal range (under ~1e-4 of the block's amax) lose their last bit; convert() counts
them. Dequantizing is exact (a <= 4-bit mantissa times a power of two is a BF16 value).

Measured: 332 linears, 5.73 -> 2.86 GiB per card, ~0.02% of values shifted; PPL KL 0.0221
vs the fp32 oracle with int8 KV (0.0191 without), decode -6 to -11%. It fails the PPL bar, so it stays off.

No FP8 matmul: Gaudi2's fp8_gemm_v2 is not supported on this stack, and an FP8 activation would change the values.
The dequantize costs one cast and one multiply per weight per call, which decode pays on every step.

Conversion runs on the device right after each layer's load (loader.load_stack), so the BF16 copies of only one
layer are alive at once. Not with ML4_DECODE_LEAN (its merged GEMV reads wq.weight).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

# "1": e4m3 codes (above); "i8": int8 codes, scale = block amax / 127 (BF16), dequantized the same way (not exact:
# the checkpoint's e4m3 values are not on an int8 grid); "fake": the "1" values stored back as BF16 at load (no memory
# saved, no per-call dequantize: separates the values from the runtime path in an A/B).
MODE = os.environ.get("ML4_FP8_LINEARS", "0")
ENABLED = MODE in ("1", "i8", "fake")
BLOCK = 128
E4M3_MAX = 240.0


class Fp8BlockLinear(nn.Module):
    """y = x W^T with W [out, in] held as 8-bit codes (uint8 e4m3 or int8) + BF16 block scales [out / 128, in / 128]."""

    def __init__(self, codes: torch.Tensor, scale: torch.Tensor, int8: bool = False):
        super().__init__()
        self.codes = nn.Parameter(codes, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)
        self.int8 = int8

    def dequant(self) -> torch.Tensor:
        o, i = self.codes.shape
        c = self.codes if self.int8 else self.codes.view(torch.float8_e4m3fn)
        w = c.to(torch.bfloat16).view(o // BLOCK, BLOCK, i // BLOCK, BLOCK)
        return (w * self.scale.view(o // BLOCK, 1, i // BLOCK, 1)).view(o, i)

    def forward(self, x):
        return F.linear(x, self.dequant())


class _Bf16Linear(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight)


def to_fp8(weight: torch.Tensor, mode: str = "1") -> tuple[nn.Module, int]:
    """(module, elements whose value changed) for a BF16 weight [out, in], both multiples of 128."""
    o, i = weight.shape
    assert o % BLOCK == 0 and i % BLOCK == 0, (o, i)
    w = weight.float().view(o // BLOCK, BLOCK, i // BLOCK, BLOCK)
    amax = w.abs().amax(dim=(1, 3), keepdim=True)
    if mode == "i8":
        s = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax)).to(torch.bfloat16).float()
        codes = torch.round(w / s).clamp(-127, 127).to(torch.int8).reshape(o, i)
        mod = Fp8BlockLinear(
            codes.contiguous(), s.reshape(o // BLOCK, i // BLOCK).to(torch.bfloat16).contiguous(), int8=True
        )
    else:
        s = torch.where(
            amax > 0, torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-38) / E4M3_MAX))), torch.ones_like(amax)
        )
        codes = (w / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8).reshape(o, i)
        mod = Fp8BlockLinear(codes.contiguous(), s.reshape(o // BLOCK, i // BLOCK).to(torch.bfloat16).contiguous())
    deq = mod.dequant()
    changed = int((deq != weight).sum())
    if mode == "fake":
        mod = _Bf16Linear(deq.contiguous())
    return mod, changed


def _targets(layer):
    """(parent, attribute) of every FP8-sourced linear of one stack.ML4Layer."""
    a = layer.attn
    yield a, "wq"
    yield a, "wo"
    if a.indexer is not None:
        yield a.indexer, "wq_b"
    f = layer.ffn
    if layer.role.is_moe:
        yield f.shared, "w13"
        yield f.shared, "w2"
    else:
        yield f, "w13"
        yield f, "w2"


def convert_layer(layer, st: dict) -> None:
    """Replace the FP8-sourced BF16 linears of one decoder layer in place; counts into st."""
    for parent, name in _targets(layer):
        lin = getattr(parent, name)
        w = lin.weight.data
        mod, changed = to_fp8(w, MODE)
        st["fp8lin_linears"] = st.get("fp8lin_linears", 0) + 1
        st["fp8lin_bf16_bytes"] = st.get("fp8lin_bf16_bytes", 0) + w.numel() * 2
        st["fp8lin_fp8_bytes"] = st.get("fp8lin_fp8_bytes", 0) + (
            mod.codes.numel() + mod.scale.numel() * 2 if hasattr(mod, "codes") else w.numel() * 2
        )
        st["fp8lin_changed"] = st.get("fp8lin_changed", 0) + changed
        st["fp8lin_elements"] = st.get("fp8lin_elements", 0) + w.numel()
        setattr(parent, name, mod)
        del lin, w


def summary(st: dict) -> str:
    return (
        "FP8 linears ("
        + MODE
        + "): %d converted, %.2f -> %.2f GiB, %d of %d values changed (subnormal shifts)"
        % (
            st.get("fp8lin_linears", 0),
            st.get("fp8lin_bf16_bytes", 0) / 2**30,
            st.get("fp8lin_fp8_bytes", 0) / 2**30,
            st.get("fp8lin_changed", 0),
            st.get("fp8lin_elements", 0),
        )
    )
