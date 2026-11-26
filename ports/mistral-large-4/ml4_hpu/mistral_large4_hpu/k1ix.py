"""ML4_INDEX_KERNEL=1: the fused decode-indexer TPC kernels for decode / verify rows. Default off. Same indices in the same order as ML4Indexer.select's decode path
(DECODE_FAST + select_topk_key) on every gated case; anything outside the gate takes the native path unchanged.

Pipeline per layer (one lazy graph region, 10 nodes incl. the cast, q tail and k write stay native):
  k1_kg    TPC  gather + e4m3 decode of index-K from the paged U8 cache, tag check -> K f32 [B, Lk, 128] (NaN = invalid)
  matmul   MME  d = q . K^T (the plugin's own fp32 dot: same operands and shapes -> same bits)
  k1_sc    TPC  relu(d) * w summed in the plugin's order, bf16, order key (+ unseen flag), radix pass 1
  k1_p2..4 TPC  radix passes 2-4 -> the k-th largest key per row
  k1_fin   TPC  per-chunk output offsets / tie quotas (ties to the lower position)
  k1_cm1   TPC  compaction phase 1: per-chunk take / see lane masks (vector)
  k1_cm2   TPC  compaction phase 2: ascending positions, -1 for unseen, dense_topk rows for pos <= k-1 (int32; cast
                natively)

Libraries (ML4_INDEX_KERNEL_LIB, default /opt/gaudi-cookbook-ml4/k1; built by the port's kernels/k1/build_k1.sh):
  libk1_tpc.so      the eight GUIDs + forwarding of every other GUID (K1_TPC_FWD_LIB, default the stock library)
  libk1_op_lazy.so  torch.ops.custom_op.k1_* (lazy-mode legacy custom ops)
setup_env() runs from mxk.setup_env() (before the device opens) and puts libk1_tpc.so in the kernel library chain:
  with the mxk MoE library:   GC_KERNEL_PATH=libmxk_tpc.so, MXK_TPC_FWD_LIB=libk1_tpc.so, K1_TPC_FWD_LIB=<stock>
  without (ML4_MXK=0):        GC_KERNEL_PATH=libk1_tpc.so, K1_TPC_FWD_LIB=<stock>
and PT_HPU_LAZY_ACC_PAR_MODE=0 (the lazy accumulation thread breaks custom ops fed by same-graph tensors).
Missing libraries, or the env not in place when the device opened: the flag turns itself off with one log line.
"""

from __future__ import annotations

import collections
import os

ENABLED = os.environ.get("ML4_INDEX_KERNEL", "0") == "1"
# ML4_ATTN_KERNEL=1 (default off): the decode sparse-attention gather kernel k2_g2 in the same library (k2attn.py)
ATTN = os.environ.get("ML4_ATTN_KERNEL", "0") == "1"
LIB_DIR = os.environ.get("ML4_INDEX_KERNEL_LIB", "/opt/gaudi-cookbook-ml4/k1")
NCH = int(os.environ.get("ML4_INDEX_KERNEL_NCH", "0"))  # chunks per row; 0: 24 // S (NCH * S ~ the 24 TPC cores)
MAX_ROWS = 8  # decode / verify rows (indexer.DECODE_ROWS)
STOCK = "/usr/lib/habanalabs/libtpc_kernels.so"
_ops_loaded = False
_warned: set = set()
# Lazy mode outside an HPU graph: a native-op output fed to a legacy custom op and freed by Python before the lazy
# graph runs can crash the bridge. Keep the small int inputs of recent calls alive (r2-k1 finding).
_KEEP: collections.deque = collections.deque(maxlen=512)


def tpc_lib() -> str:
    return os.path.join(LIB_DIR, "libk1_tpc.so")


def op_lib() -> str:
    return os.path.join(LIB_DIR, "libk1_op_lazy.so")


def lib_version(path: str | None = None) -> str | None:
    try:
        data = open(path or tpc_lib(), "rb").read()
    except OSError:
        return None
    i = data.find(b"k1_tpc ")
    if i < 0:
        return None
    return data[i : data.index(b"\0", i)].decode(errors="replace")


def _log(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        print(f"[ml4-k1] {msg}", flush=True)


def setup_env(mxk_tpc: str | None) -> str | None:
    """Before the first HPU op (called by mxk.setup_env). mxk_tpc: the mxk single-file library when it is in use,
    else None. Returns the k1 version tag for the recipe-cache dir name, None when off. Idempotent."""
    global ENABLED, ATTN
    if not (ENABLED or ATTN):
        return None
    ver = lib_version()
    if not (ver and os.path.isfile(op_lib())):
        ENABLED = ATTN = False
        _log(f"ML4_INDEX_KERNEL / ML4_ATTN_KERNEL on but no libk1_tpc.so + libk1_op_lazy.so in {LIB_DIR}: native paths")
        return None
    tpc = tpc_lib()
    if mxk_tpc:
        nxt = os.environ.get("MXK_TPC_FWD_LIB", STOCK)
        if nxt != tpc:
            os.environ["K1_TPC_FWD_LIB"] = nxt if nxt and nxt != mxk_tpc else STOCK
            os.environ["MXK_TPC_FWD_LIB"] = tpc
    else:
        cur = [p for p in os.environ.get("GC_KERNEL_PATH", STOCK).split(":") if p and p != tpc]
        os.environ["K1_TPC_FWD_LIB"] = cur[0] if len(cur) == 1 else STOCK
        if len(cur) > 1:
            _log(f"GC_KERNEL_PATH had {len(cur)} libraries; forwarding to {STOCK} only")
        os.environ["GC_KERNEL_PATH"] = tpc
    os.environ["PT_HPU_LAZY_ACC_PAR_MODE"] = "0"
    return "k1-" + ver.split()[-1]


def env_ok() -> bool:
    tpc = tpc_lib()
    chained = os.environ.get("GC_KERNEL_PATH") == tpc or os.environ.get("MXK_TPC_FWD_LIB") == tpc
    return chained and os.environ.get("PT_HPU_LAZY_ACC_PAR_MODE") == "0"


def load_ops() -> bool:
    """Load the op library (model build). False (and the flag off) if the env was not in place."""
    global _ops_loaded, ENABLED, ATTN
    if not (ENABLED or ATTN):
        return False
    if not _ops_loaded:
        if not env_ok():
            ENABLED = ATTN = False
            _log(
                "ML4_INDEX_KERNEL / ML4_ATTN_KERNEL on but the kernel library was not registered before the device "
                "opened (mxk.setup_env): native paths"
            )
            return False
        import torch

        torch.ops.load_library(op_lib())
        _ops_loaded = True
    return True


def eligible(S: int, Lk: int, k_out: int, cache) -> bool:
    """The rows the kernel is proven on (gate 1): decode / verify rows on the native DECODE_FAST + key-select path,
    the U8 index cache (not fake), no DCP, k_out < Lk, multiples of 64. Host-side checks only (graph-safe)."""
    if not (ENABLED and _ops_loaded) or S > MAX_ROWS or k_out >= Lk or Lk % 64 or k_out % 64:
        return False
    from . import indexer as IX
    from .paged import INDEX_U8_MODE, TAG_MODE

    if not IX.DECODE_FAST or S > IX.DECODE_ROWS or IX.TOPK_SELECT != "thresh" or IX.decode_select() != "key":
        return False
    if IX.dcp_rows(cache, S) or IX.prompt_mask(S, Lk):
        return False
    if INDEX_U8_MODE != "1" or getattr(cache, "fake", True) or TAG_MODE not in ("mask", "off"):
        return False
    return True


def _own(ctx):
    import torch

    B, NB = ctx.block_table.shape
    if ctx.block_epoch is not None:
        e = ctx.block_epoch
        return e if e.dtype == torch.int32 else ctx.shared(("k1_own32",), lambda: e.to(torch.int32))
    return ctx.shared(("k1_own32",), lambda: ctx.req_uid.to(torch.int32).view(B, 1).expand(B, NB).contiguous())


def select(q, w, ctx, cache, k_out: int, nch: int | None = None):
    """== ML4Indexer.select(q, w, ctx, cache, k_out) for eligible() rows: int64 [B, S, k_out]."""
    import torch
    from .paged import TAG_MODE

    ops = torch.ops.custom_op
    pos = ctx.positions
    nch = nch or (NCH if NCH > 0 else max(1, 24 // pos.shape[1]))
    p32 = ctx.shared(("k1_pos32",), lambda: pos.to(torch.int32))
    bt32 = ctx.shared(("k1_bt32",), lambda: ctx.block_table.to(torch.int32))  # once per forward (44 layers)
    own = _own(ctx)
    _KEEP.append((p32, bt32, own))
    Ki = ops.k1_kg(cache.tags.view(-1), cache.data, bt32, own, ctx.block_size, 0 if TAG_MODE == "off" else 1)
    K = Ki.view(torch.float32)  # int32-declared fp32 bits
    B = pos.shape[0]
    d = torch.matmul(q.float().reshape(B, -1, q.shape[-1]), K.transpose(1, 2))
    keys, c1 = ops.k1_sc(Ki, d, p32, w.float().contiguous(), nch)
    c2 = ops.k1_p2(keys, c1, nch, k_out)
    c3 = ops.k1_p3(keys, c1, c2, nch, k_out)
    c4 = ops.k1_p4(keys, c1, c2, c3, nch, k_out)
    nv = d.shape[-1] // 64
    mw = ((-(-nv // nch)) + 15) // 16 * 64 + 64  # per-chunk scratch words (mask words + sentinel)
    fin = ops.k1_fin(c1, c2, c3, c4, nch, k_out, 0)
    mk = ops.k1_cm1(keys, p32, fin, nch, k_out, mw)
    t32 = ops.k1_cm2(keys, p32, fin, mk, nch, k_out, mw)
    t64 = t32.long()
    if ctx.memo is not None:  # ML4_R5_K2FOLD: k2_g3 reads the int32 top-k (one forward's memo)
        ctx.memo[("k1_t32", id(t64))] = t32
    return t64
