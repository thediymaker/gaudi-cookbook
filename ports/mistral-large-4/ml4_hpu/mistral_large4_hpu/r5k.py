"""Exact node cuts, set 1. Default off; each flag is
one TPC kernel in the decode-indexer library (libk1_tpc.so, k1ix) that replaces a chain of small dependent native
nodes with the same bits. Flags are read when a graph is traced (module attributes; tests flip them per arm):
  ML4_R5_RMS=1     numerics.rms_norm tail after the native rsqrt (round_bf16, weight multiply, bf16 cast): r5_rms
  ML4_R5_ROPE=1    rope.rope_head / rope_tail on 128-wide heads (the slices, the rotation, the cats): r5_rope
  ML4_R5_ROUTER=1  the decode MoE router after the native softmax (top-4, renormalise, virtual ids, bf16 weights): r5_rt
  ML4_R5_FF8=rope|x|bf16  the indexer's fake_fp8_ue8m0 of q and k: r5_ff8 (bf16 in) on RoPE layers; x: NoPE layers too,
                   from the fp32 values before their bf16 cast (r5_ff8f); bf16: every layer from the bf16 tensors
  ML4_R5_LAT=1     an owner layer's latent cache row (kv_a_norm tail, k_pe RoPE, the i8-t128-f32 pack): r5_lat
  ML4_R5_K2FOLD=1  the decode attention gather's per-key addressing folded into the gather kernel: k2_g3
  ML4_R5_SSL=bf16|f32  numerics.ssmax_lambda: r5_ssl (bf16 lambda) / r5_sslf (fp32 lambda before the cast)
: ROPE and LAT are NOT exact as built: the graph compiler rotates tensors of < 8192 pairs as
y0 = fma(-x1, s, fl(x0 c)), y1 = fma(x1, c, fl(x0 s)) (every decode q_tail up to B*S 28, indexer q up to B*S 7, every
k_pe), which r5_rope / r5_lat's separate roundings miss on ~1 element in 10^4 (shared layers failed at B 1 / 2 / 4 / 8).
Use them only with ML4_R6_ROPEFMA=1 (r6k.rope_form: the matching form per call, native in between).
Every kernel keeps the native rounding points (fp32 IEEE multiplies / adds, RNE on the bits, the native division as
x * reciprocal) and leaves anything with exp / rsqrt / mean on the native path. Needs the r5 build of the library."""

from __future__ import annotations

import os

import torch

RMS = os.environ.get("ML4_R5_RMS", "0") == "1"
ROPE = os.environ.get("ML4_R5_ROPE", "0") == "1"
ROUTER = os.environ.get("ML4_R5_ROUTER", "0") == "1"
ROUTER_MODE = int(
    os.environ.get("ML4_R5_ROUTER_MODE", "1")
)  # 1 = (w0+w1)+(w2+w3), reciprocal_f32: matches the served graph at T <= 128 ; 4 (cephes) is inexact in the server
FF8 = os.environ.get("ML4_R5_FF8", "0")
SSL = os.environ.get("ML4_R5_SSL", "0")
LAT = os.environ.get("ML4_R5_LAT", "0") == "1"
K2FOLD = os.environ.get("ML4_R5_K2FOLD", "0") == "1"
_Z: dict = {}
_ZB: dict = {}
_ZI: dict = {}


def _dec(x: torch.Tensor) -> bool:
    """decode / verify rows only (S <= ML4_DECODE_S): the shapes the r5 gates ran; prompt graphs stay native."""
    from .attention import DECODE_S

    return x.dim() >= 2 and x.shape[1] <= DECODE_S


def ready() -> bool:
    from . import k1ix

    return bool(k1ix._ops_loaded) and hasattr(torch.ops.custom_op, "r5_rms")


def init(device) -> None:
    """Persistent tensors, made before any graph capture (as numerics.rw_init)."""
    _z(device)
    _zb(device)
    _zi(device)


def _zb(device) -> torch.Tensor:
    k = str(device)
    if k not in _ZB:
        _ZB[k] = torch.zeros(1, dtype=torch.bfloat16, device=device)
    return _ZB[k]


def _zi(device) -> torch.Tensor:
    k = str(device)
    if k not in _ZI:
        _ZI[k] = torch.zeros(1, dtype=torch.int8, device=device)
    return _ZI[k]


def _z(device) -> torch.Tensor:
    k = str(device)
    if k not in _Z:
        _Z[k] = torch.zeros(1, dtype=torch.int32, device=device)
    return _Z[k]


def rms_on(x: torch.Tensor, D: int) -> bool:
    return RMS and _dec(x) and x.device.type == "hpu" and x.dtype == torch.bfloat16 and D % 128 == 0 and ready()


def rms(x: torch.Tensor, weight: torch.Tensor, eps: float, D: int | None = None) -> torch.Tensor:
    """== numerics.rms_norm(x[..., :D], weight, eps) for bf16 x; x [..., Dx] contiguous, Dx >= D."""
    D = D or x.shape[-1]
    xf = x[..., :D].float()
    r = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    y = torch.ops.custom_op.r5_rms(x.reshape(-1, x.shape[-1]), r.reshape(-1, 1), weight)
    return y.view(*x.shape[:-1], D)


def rope_on(x: torch.Tensor, rd: int) -> bool:
    if not (
        ROPE
        and _dec(x)
        and x.device.type == "hpu"
        and x.dtype == torch.bfloat16
        and x.shape[-1] == 128
        and rd == 64
        and ready()
    ):
        return False
    from . import r6k

    return r6k.rope_form(x.numel() // 4) is not None  # ML4_R6_ROPEFMA: only where the native form is known


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, off: int) -> torch.Tensor:
    """x [B, S, (H,) 128] bf16; cos / sin [B, S, (1,) 32] fp32 -> rope_head (off 0) / rope_tail (off 64) of x."""
    if x.dim() == cos.dim() and cos.shape[-2] == 1:
        N, H = x[..., 0, 0].numel(), x.shape[-2]
    else:
        N, H = x[..., 0].numel(), 1
    c = cos.reshape(N, 32)
    s = sin.reshape(N, 32)
    from . import r6k

    form = r6k.rope_form(N * H * 32) or 0
    return torch.ops.custom_op.r5_rope(x.reshape(N, H, 128), c, s, off | (form << 8)).view(x.shape)


def router_on(p: torch.Tensor) -> bool:
    return ROUTER and p.device.type == "hpu" and p.shape[-1] == 128 and ready()


def router(p: torch.Tensor):
    """p [T, 128] fp32 softmax -> (vid int32 [T, 8], vw bf16 [T, 8]) as moe._forward_decode (placement B, no hot
    replicas, no fitted order) and mxk.decode's bf16 cast of the weights."""
    vid, vwb = torch.ops.custom_op.r5_rt(_z(p.device), p, ROUTER_MODE)
    return vid, vwb.view(torch.bfloat16)


def ff8_on(x: torch.Tensor) -> bool:
    return FF8 in ("rope", "x", "bf16") and _dec(x) and x.device.type == "hpu" and x.shape[-1] % 128 == 0 and ready()


def ff8(x: torch.Tensor) -> torch.Tensor:
    """== numerics.fake_fp8_ue8m0(x) for bf16 x [..., 128 k] (groups of 128)."""
    return torch.ops.custom_op.r5_ff8(x.reshape(-1, 128)).view(x.shape)


def ff8_f32(xf: torch.Tensor) -> torch.Tensor:
    """fake_fp8_ue8m0 of xf.to(bf16) as the graph compiler evaluates it when the cast feeds the quantizer directly
    (the cast pair dropped): from the fp32 values, out bf16."""
    return torch.ops.custom_op.r5_ff8f(_zb(xf.device), xf.reshape(-1, 128)).view(xf.shape)


def ssl_on(lp: torch.Tensor) -> bool:
    return SSL in ("bf16", "f32") and _dec(lp) and lp.device.type == "hpu" and ready()


def ssl(lp: torch.Tensor, s: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """numerics.ssmax_lambda(lp [...], s [H], b [H]) -> [..., H]: bf16 (SSL=bf16) or the fp32 value before the cast."""
    lp2 = lp.reshape(-1, 1)
    sb, bb = s.to(torch.bfloat16), b.to(torch.bfloat16)
    if SSL == "f32":
        y = torch.ops.custom_op.r5_sslf(lp2, sb, bb)
    else:
        y = torch.ops.custom_op.r5_ssl(sb, bb, lp2)
    return y.view(*lp.shape, s.shape[0])


def lat_on(kv: torch.Tensor, cache) -> bool:
    if not (
        LAT and _dec(kv) and kv.device.type == "hpu" and kv.dtype == torch.bfloat16 and kv.is_contiguous() and ready()
    ):
        return False
    from . import paged as PG, r6k

    if r6k.rope_form(kv[..., 0].numel() * 32) is None:  # ML4_R6_ROPEFMA: k_pe rotation form unknown at this size
        return False
    return (
        isinstance(cache, PG.Fp8LatentCache)
        and not cache.fake
        and cache.dcp is None
        and PG.KV_FP8 == "1"
        and PG.KV_CODEC == "i8-t128-f32"
        and cache.data.shape[-1] == 328
    )


def latent(kv: torch.Tensor, w: torch.Tensor, eps: float, cos: torch.Tensor, sin: torch.Tensor, use_rope: bool):
    """== paged.fp8_latent_pack(attention.latent_rows(...)): kv [B, S, 576 | 512] (the wkv_a output) -> [B, S, 328]
    bf16-typed cache rows (656 bytes)."""
    B, S, W = kv.shape
    xf = kv[..., :512].float()
    r = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    from . import r6k

    form = (r6k.rope_form(B * S * 32) or 0) if use_rope else 0
    row = torch.ops.custom_op.r5_lat(
        _zi(kv.device),
        kv.reshape(B * S, W),
        r.reshape(-1, 1),
        w,
        cos.reshape(B * S, 32),
        sin.reshape(B * S, 32),
        (1 if use_rope else 0) | (form << 8),
    )
    return row.view(torch.bfloat16).view(B, S, 328)
