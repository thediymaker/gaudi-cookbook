"""Exact node cuts, set 2. Default off; each flag is one
TPC kernel in the decode-indexer library (libk1_tpc.so, k1ix; r6 build) that replaces a chain of small dependent native
nodes with the same bits. Flags are read when a graph is traced (module attributes; tests flip them per arm):
  ML4_R6_FFK=rope|x  the indexer k's fake_fp8_ue8m0 and the index-K U8 pack (paged.index_u8_pack) in one kernel
                     (r6_ffk), written to the cache as the encoded row. rope: RoPE layers (k from rope_head, bf16);
                     x: NoPE layers too, from the fp32 layer_norm output the compiler keeps ahead of fake_fp8 (r6_ffkf)
  ML4_R6_DUAL=1|nope the full-indexer layers' input rms_norm tail as one kernel with two outputs (r6_rmsd): the bf16
                     value for the linears and the fp32 product the graph compiler hands the indexer's layer_norm
                     (it drops the bf16 round trip there, which is why ML4_R5_RMS leaves this call site native).
                     1: every full-indexer layer; nope: NoPE full-indexer layers only (the harness measured a gain on
                     NoPE / KV-shared layers and a small loss on RoPE ones next to ML4_R6_FFK)
  ML4_R6_ROPEFMA=1   the r5 RoPE kernels (ML4_R5_ROPE r5_rope, ML4_R5_LAT's k_pe) in the native arithmetic: the graph
                     compiler rotates small tensors (<= ML4_R6_ROPEFMA_LO pairs) as y0 = fma(-x1, s, fl(x0 c)),
                     y1 = fma(x1, c, fl(x0 s)) and large ones (>= ML4_R6_ROPEFMA_HI pairs) with separate roundings
                     (t/ropemap.py, every site and shape mapped); in between the call stays native. Without it the
                     r5 kernels always use separate roundings, which is NOT exact (shared layers at B 1 / 2 / 4 / 8)
Decode / verify shapes only (S <= ML4_DECODE_S), HPU, the r6 library loaded. Every kernel keeps the native rounding
points; anything with rsqrt / mean / exp stays native."""

from __future__ import annotations

import os

import torch

FFK = os.environ.get("ML4_R6_FFK", "0")
FFK_MODE = int(os.environ.get("ML4_R6_FFK_MODE", "0"))  # r6_ffk division forms (see the kernel); 0 = proven
DUAL = os.environ.get("ML4_R6_DUAL", "0")  # "0" | "1" | "nope"
ROPEFMA = os.environ.get("ML4_R6_ROPEFMA", "0") == "1"
ROPEFMA_LO = int(os.environ.get("ML4_R6_ROPEFMA_LO", "7168"))
ROPEFMA_HI = int(os.environ.get("ML4_R6_ROPEFMA_HI", "8192"))
# ML4_R6_DUAL_ROPE_ROWS=n (default 0 = no cap): with DUAL=1, RoPE full-indexer layers take r6_rmsd only when the
# call has at most n rows (B x S, padding rows included); above it they stay native (the DUAL=nope form).: in the
# real slice (35..42) DUAL=1 on RoPE layers costs +2.4..+3.5 us/layer at 24 rows (8-stream verify, T 3) and saves at
# 1-21 rows; 21 keeps every other served shape on DUAL=1.
DUAL_ROPE_ROWS = os.environ.get("ML4_R6_DUAL_ROPE_ROWS", "0")
DEFAULTS = {"FFK": "0", "DUAL": "0", "ROPEFMA": False, "DUAL_ROPE_ROWS": "0"}
_ZI: dict = {}
_DUAL: dict = {}


def ready() -> bool:
    from . import k1ix

    return bool(k1ix._ops_loaded) and hasattr(torch.ops.custom_op, "r6_ffk")


def init(device) -> None:
    """Persistent tensors, made before any graph capture (as numerics.rw_init)."""
    _zi(device)


def _zi(device) -> torch.Tensor:
    k = str(device)
    if k not in _ZI:
        _ZI[k] = torch.zeros(1, dtype=torch.int8, device=device)
    return _ZI[k]


def _dec(x: torch.Tensor) -> bool:
    from .attention import DECODE_S

    return x.dim() >= 2 and x.shape[1] <= DECODE_S


# ------------------------------------------------------------------ ROPEFMA
def rope_form(pairs: int):
    """The rotation form for a call rotating `pairs` (x0, x1) pairs: 0 separate roundings, 1 fma, None unknown (the
    caller stays native). Without ML4_R6_ROPEFMA: 0 (the r5 kernels as built)."""
    if not ROPEFMA:
        return 0
    if pairs <= ROPEFMA_LO:
        return 1
    if pairs >= ROPEFMA_HI:
        return 0
    return None


# ------------------------------------------------------------------ FFK
def ffk_cache_ok(cache) -> bool:
    """The index-K cache takes r6_ffk's rows as they are: U8 codec (not fake), 66-wide rows, no DCP."""
    from . import paged as PG

    return (
        isinstance(cache, PG.IndexU8Cache)
        and not cache.fake
        and cache.dcp is None
        and PG.INDEX_U8_MODE == "1"
        and cache.data.shape[-1] == PG.INDEX_U8_W
    )


def ffk_on(k: torch.Tensor, use_rope: bool) -> bool:
    from . import numerics as NU

    if not (
        FFK in ("rope", "x")
        and (use_rope or FFK == "x")
        and NU.FAKE_QUANT
        and _dec(k)
        and k.device.type == "hpu"
        and k.shape[-1] == 128
        and ready()
    ):
        return False
    return k.dtype == (torch.bfloat16 if use_rope else torch.float32)


def ffk(k: torch.Tensor) -> torch.Tensor:
    """k [B, S, 128] bf16 (RoPE layers, after rope_head) or fp32 (NoPE: the layer_norm output before its bf16 cast)
    -> [B, S, 66] bf16-typed index-K rows == paged.index_u8_pack(numerics.fake_fp8_ue8m0(k as bf16))."""
    lead = k.shape[:-1]
    if k.dtype == torch.float32:
        row = torch.ops.custom_op.r6_ffkf(_zi(k.device), k.reshape(-1, 128), FFK_MODE)
    else:
        row = torch.ops.custom_op.r6_ffk(_zi(k.device), k.reshape(-1, 128), FFK_MODE)
    return row.view(torch.bfloat16).view(*lead, 66)


# ------------------------------------------------------------------ DUAL
def dual_on(x: torch.Tensor, kind: str = "rope") -> bool:
    """kind: the layer's "rope" / "nope" (stack.ML4Layer passes it for full-indexer layers)."""
    if DUAL not in ("1", True) and not (DUAL == "nope" and kind == "nope"):
        return False
    cap = int(DUAL_ROPE_ROWS or 0)
    if kind == "rope" and cap and x.dim() >= 2 and x.shape[0] * x.shape[1] > cap:
        return False
    return (
        _dec(x)
        and x.device.type == "hpu"
        and x.dtype == torch.bfloat16
        and x.shape[-1] % 128 == 0
        and x.is_contiguous()
        and ready()
    )


def rms_dual(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """== numerics.rms_norm(x, weight, eps) (bf16); the fp32 product before its cast is kept for take()."""
    D = x.shape[-1]
    xf = x.float()
    r = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    y2, p = torch.ops.custom_op.r6_rmsd(r.reshape(-1, 1), x.reshape(-1, D), weight)
    y = y2.view(torch.bfloat16).view(x.shape)
    _DUAL[id(y)] = (y, p.view(x.shape))
    return y


def take(a: torch.Tensor):
    """The fp32 product behind a (rms_dual's output), once; None for any other tensor."""
    e = _DUAL.pop(id(a), None)
    if e is None or e[0] is not a:
        return None
    return e[1]
