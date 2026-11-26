"""Ranged per-rank loader for the AMD Quark MXFP4 Mistral Large 4 checkpoint (and syn8l, written the same way).

vLLM's default iterator would read the whole 578 GB snapshot on every rank. This reads the safetensors headers and
reads exactly the bytes one TP/EP rank keeps, straight into host staging tensors (os.preadv), then copies them to
the device: ~68 GB of routed experts + ~7 GB of the rest per rank at TP8/EP8.

Checkpoint names are Mistral-native. What each parameter of stack.ML4Stack receives (r = rank, tp = TP
size = EP size, Hl = 64 / tp heads, q = 8192 / tp):
  embed / lm_head              tok_embeddings / output rows [r V/tp, +V/tp) (vocab-parallel; lm_head kept fp32)
  norm, lerope alpha           whole
  attn_norm / ffn_norm         whole
  attn.wq                      FP8 wq rows [r q, +q) -> BF16 on the host (block scales, below)
  attn.wkv_a, kv_a_norm        whole (replicated; absent on the KV-shared layers)
  attn.wkv_b                   rows [r Hl (P + 128), +Hl (P + 128)) (by head; P = 64 RoPE / 128 NoPE)
  attn.wo                      FP8 wo columns [r q, +q) -> BF16
  attn.ssmax_s / _b            [r Hl, +Hl)
  attn.indexer.*               whole (replicated); wq_b FP8 -> BF16; k_norm -> fp32
  ffn (L0) w13 / w2            feed_forward.w1 / w3 rows [r I/tp, +I/tp) stacked [w1; w3]; w2 columns -> BF16
  ffn.gate_w                   gate.weight whole (replicated)
  ffn.shared.w13 / w2          shared_experts.w1 / w3 rows, w2 columns, as the dense FFN -> BF16
  ffn.w13 / w13_scale          routed experts, virtual re-slice (moe.py): local slot v holds (e, h):
                               rows [0, 3072) <- w1 rows [3072h, +3072), rows [3072, 6144) <- w3 the same rows
                               (packed U8 [6144, 3072] + E8M0 U8 [6144, 192]), each read straight into its slot
  ffn.w2 / w2_scale            w2 columns [3072h, +3072) = packed bytes [1536h, +1536) + scale bytes [96h, +96):
                               one preadv per tensor scatters each row's halves into the two slots
No byte is stored twice and no expert byte is transformed: AMD's bytes are the kernels' served layout (low nibble
first, E8M0 bias 127). Packed weights and scales only travel as uint8 (an int8 / float8_e8m0 typed copy on HPU
clamps bytes).

Block FP8 -> BF16: w = e4m3fn(code) * scale[row // 128, col // 128] in fp32, rounded to BF16 once. With
power-of-two scales (use_ue8m0_for_qscales) every product has <= 4 significant bits and is exact in BF16; the loader
counts non-power-of-two scales (gates them to 0) and never casts an FP8 code on the device (Gaudi2 reads e4m3fn
codes 256..448 as inf/NaN).

Skipped by construction (only named tensors are read): mtp_dspark.* (the DSpark drafter loads its own),
vision_encoder / patch_merger / pre_mm_projector_norm / vision_language_adapter (text-only).
ML4_LOAD_SKIP=experts (tests / fast restarts): routed experts left zero, nothing else changes.
ML4_LOAD_THREADS (16): reader threads for the expert reads.
ML4_LOAD_FAST=1 (default off; load_stack_fast): the same bytes into the same parameters, pipelined. Reader threads stage
whole layers ahead of the device copies (ML4_LOAD_PREFETCH layers, default 2; host memory per rank ~ (prefetch + 2) x
1.1 GB), so reads of layers L+1..L+p (other shard files) overlap the copies and flush of layer L; FP8 -> BF16 runs in
the reader threads with the scale broadcast (no repeat_interleave copies); each expert w2 is one contiguous read plus
a strided host copy of the kept half instead of 2 iovecs per row (786K Python buffers per layer). Bit-identical
by construction (same table lookup, same fp32 product, one rounding; same bytes).
No vLLM imports (the gates and the tiny engine use it directly); vllm_model.py calls load_stack().
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import logging
import os
import struct
import threading
import time

import torch

from . import moe as M

logger = logging.getLogger("vllm.mistral_large4_hpu")
LOAD_SKIP = {s for s in os.environ.get("ML4_LOAD_SKIP", "").split(",") if s}
THREADS = int(os.environ.get("ML4_LOAD_THREADS", "16"))
FAST = os.environ.get("ML4_LOAD_FAST", "0") == "1"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:  # a malformed value must not break the default load path at import
        return default


PREFETCH = max(1, _env_int("ML4_LOAD_PREFETCH", 2))
PIN = os.environ.get("ML4_LOAD_PIN", "0") == "1"  # pinned expert staging (unmeasured; falls back if refused)
LAYER_HOOK = None  # tests: LAYER_HOOK(layer, timings) after each layer's flush (fast path)
IOV_MAX = 1024
FP8_BLOCK = 128
_DT = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "U8": torch.uint8,
    "F8_E4M3": torch.uint8,
    "F16": torch.float16,
    "I32": torch.int32,
}
_ISZ = {"BF16": 2, "F32": 4, "U8": 1, "F8_E4M3": 1, "F16": 2, "I32": 4}
_E4M3 = None


def e4m3_values() -> torch.Tensor:
    """fp32 value of each e4m3fn byte code (host only)."""
    global _E4M3
    if _E4M3 is None:
        _E4M3 = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).float()
    return _E4M3


def _np(t: torch.Tensor):
    """Writable byte view of a contiguous CPU tensor for os.preadv."""
    assert t.is_contiguous() and t.device.type == "cpu"
    return t.view(torch.uint8).numpy() if t.dtype != torch.uint8 else t.numpy()


class Ckpt:
    """safetensors headers of every shard; positional reads of byte ranges (no mmap, no whole-tensor reads)."""

    def __init__(self, path: str):
        self.path = path
        files = None
        for idx in ("consolidated.safetensors.index.json", "model.safetensors.index.json"):
            p = os.path.join(path, idx)
            if os.path.exists(p):
                files = sorted(set(json.load(open(p))["weight_map"].values()))
                break
        if files is None:
            files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
        self.meta: dict[str, tuple] = {}  # name -> (file, abs offset, nbytes, dtype, shape)
        for fn in files:
            with open(os.path.join(path, fn), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.meta[k] = (fn, 8 + n + a, b - a, v["dtype"], tuple(v["shape"]))
        self._fds: dict[str, int] = {}
        self._lock = threading.Lock()
        self.bytes = 0

    def has(self, name: str) -> bool:
        return name in self.meta

    def shape(self, name: str) -> tuple:
        return self.meta[name][4]

    def _fd(self, fn: str) -> int:
        with self._lock:
            if fn not in self._fds:
                self._fds[fn] = os.open(os.path.join(self.path, fn), os.O_RDONLY)
            return self._fds[fn]

    def _preadv(self, fd: int, bufs: list, off: int) -> None:
        bufs = [memoryview(b).cast("B") for b in bufs]  # flat byte views: len() == bytes
        for i in range(0, len(bufs), IOV_MAX):
            part = bufs[i : i + IOV_MAX]
            want = sum(len(b) for b in part)
            got = 0
            while got < want:  # short reads: re-issue the rest
                n = os.preadv(fd, part, off + got) if got == 0 else self._rest(fd, part, off, got)
                if n <= 0:
                    raise IOError(f"short read at {off + got}")
                got += n
            off += want
        with self._lock:
            self.bytes += sum(len(b) for b in bufs)

    @staticmethod
    def _rest(fd, part, off, got):
        skip, rest = got, []
        for b in part:
            if skip >= len(b):
                skip -= len(b)
                continue
            rest.append(b[skip:] if skip else b)
            skip = 0
        return os.preadv(fd, rest, off + got)

    def read_rows(self, name: str, out: torch.Tensor, r0: int = 0, r1: int | None = None) -> torch.Tensor:
        """Rows [r0, r1) of tensor `name` into `out` (contiguous, same itemsize as the checkpoint dtype)."""
        fn, off, nb, dt, shape = self.meta[name]
        r1 = shape[0] if r1 is None else r1
        row = nb // shape[0]
        buf = _np(out)
        assert buf.nbytes == (r1 - r0) * row, (name, buf.nbytes, r0, r1, row)
        self._preadv(self._fd(fn), [buf], off + r0 * row)
        return out

    def read_col_parts(self, name: str, outs: list) -> None:
        """Every row split into len(outs) equal column parts, part j of each row into outs[j] [R, C / n]
        (None: the part is read into a scratch row and dropped). One positional read over the whole tensor."""
        fn, off, nb, dt, shape = self.meta[name]
        R = shape[0]
        row = nb // R
        k = len(outs)
        part = row // k
        assert part * k == row
        scratch = bytearray(part)
        views = [None if o is None else _np(o).reshape(R, part) for o in outs]
        bufs = []
        for r in range(R):
            for j in range(k):
                bufs.append(memoryview(scratch) if views[j] is None else views[j][r])
        self._preadv(self._fd(fn), bufs, off)

    def tensor(self, name: str, rows: tuple[int, int] | None = None) -> torch.Tensor:
        """A new CPU tensor of rows [r0, r1) (all rows by default) in the checkpoint dtype (FP8 as uint8)."""
        fn, off, nb, dt, shape = self.meta[name]
        r0, r1 = rows or (0, shape[0])
        t = torch.empty((r1 - r0, *shape[1:]), dtype=_DT[dt])
        return self.read_rows(name, t, r0, r1)

    def raw_sha(self, name: str) -> str:
        """sha256 of the tensor's bytes as stored (gates)."""
        fn, off, nb, dt, shape = self.meta[name]
        h, fd, done = hashlib.sha256(), self._fd(fn), 0
        while done < nb:  # one pread returns at most ~2 GiB
            b = os.pread(fd, min(nb - done, 1 << 28), off + done)
            if not b:
                raise IOError(f"short read in {name}")
            h.update(b)
            done += len(b)
        return h.hexdigest()


# ------------------------------------------------------------------ FP8 block -> BF16
def fp8_block_bf16(
    ck: Ckpt,
    name: str,
    rows: tuple[int, int] | None = None,
    cols: tuple[int, int] | None = None,
    stats: dict | None = None,
) -> torch.Tensor:
    """BF16 [rows, cols] of the block-FP8 linear `name` (+ `name`_scale F32 [R/128, C/128]), on the host:
    e4m3fn(code) * scale in fp32, one rounding. rows/cols are multiples of 128."""
    R, C = ck.shape(name + ".weight")
    r0, r1 = rows or (0, R)
    c0, c1 = cols or (0, C)
    assert r0 % FP8_BLOCK == 0 and c0 % FP8_BLOCK == 0 and (r1 - r0) % FP8_BLOCK == 0 and (c1 - c0) % FP8_BLOCK == 0
    w = ck.tensor(name + ".weight", (r0, r1))[:, c0:c1]
    s = ck.tensor(name + ".weight_scale", (r0 // FP8_BLOCK, r1 // FP8_BLOCK))[:, c0 // FP8_BLOCK : c1 // FP8_BLOCK]
    if stats is not None:
        bits = s.contiguous().view(torch.int32)
        stats["fp8_scales"] = stats.get("fp8_scales", 0) + s.numel()
        stats["fp8_scales_not_pow2"] = stats.get("fp8_scales_not_pow2", 0) + int(((bits & 0x7FFFFF) != 0).sum())
        stats["fp8_codes_nan"] = stats.get("fp8_codes_nan", 0) + int(((w & 0x7F) == 0x7F).sum())
    v = e4m3_values()[w.to(torch.int32)]
    v = v * s.repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)
    return v.to(torch.bfloat16)


# ------------------------------------------------------------------ routed experts
def expert_host(moe: "M.ML4MoE") -> dict:
    """Host staging tensors for one layer's local experts (reused across layers)."""
    return {
        n: torch.empty(tuple(getattr(moe, n).shape), dtype=torch.uint8) for n in ("w13", "w13_scale", "w2", "w2_scale")
    }


def read_experts(ck: Ckpt, layer: int, slots: list, host: dict, pool=None, swap_w13: bool | None = None) -> None:
    """The virtual re-slice of one layer into host staging tensors (shapes of ML4MoE's parameters).
    slots: [(real expert e, half h)] per local virtual slot (moe.local_experts). swap_w13 (control; default
    moe.FAULTS["swap_w13"]): w3 rows where w1 belongs and vice versa."""
    if swap_w13 is None:
        swap_w13 = M.FAULTS["swap_w13"]
    IV = host["w13"].shape[1] // 2
    SV = host["w13_scale"].shape[1] // 2
    assert IV == SV
    p = f"layers.{layer}.experts"
    first, second = ("w3", "w1") if swap_w13 else ("w1", "w3")
    jobs = []
    for v, (e, h) in enumerate(slots):
        for proj, dst in ((first, slice(0, IV)), (second, slice(IV, 2 * IV))):
            jobs.append((ck.read_rows, (f"{p}.{e}.{proj}.weight", host["w13"][v, dst], IV * h, IV * (h + 1))))
            jobs.append(
                (ck.read_rows, (f"{p}.{e}.{proj}.weight_scale", host["w13_scale"][v, dst], IV * h, IV * (h + 1)))
            )
    by_e: dict[int, dict] = {}
    for v, (e, h) in enumerate(slots):
        by_e.setdefault(e, {})[h] = v
    for e, halves in by_e.items():
        for t, suffix in (("w2", "weight"), ("w2_scale", "weight_scale")):
            outs = [host[t][halves[h]] if h in halves else None for h in (0, 1)]
            jobs.append((ck.read_col_parts, (f"{p}.{e}.w2.{suffix}", outs)))
    if pool is None:
        for fn, a in jobs:
            fn(*a)
    else:
        for f in [pool.submit(fn, *a) for fn, a in jobs]:
            f.result()


# ------------------------------------------------------------------ whole stack
def _put(param: torch.nn.Parameter, t: torch.Tensor) -> None:
    assert tuple(param.shape) == tuple(t.shape), (tuple(param.shape), tuple(t.shape))
    with torch.no_grad():
        param.data.copy_(t.to(param.dtype) if t.dtype != param.dtype else t)


def load_stack(model, path: str, stats: dict | None = None) -> dict:
    """Fill every parameter of a stack.ML4Stack (one rank) from the checkpoint at `path`. Returns stats."""
    if FAST:
        return load_stack_fast(model, path, stats)
    t0 = time.time()
    ck = Ckpt(path)
    st = stats if stats is not None else {}
    r, tp = model.tp_rank, model.tp_size
    Vp = model.vp
    _put(model.embed, ck.tensor("tok_embeddings.weight", (r * Vp, (r + 1) * Vp)))
    _put(model.lm_head, ck.tensor("output.weight", (r * Vp, (r + 1) * Vp)).float())
    _put(model.norm, ck.tensor("norm.weight"))
    _put(model.alpha, ck.tensor("lerope.alpha"))
    pool = cf.ThreadPoolExecutor(THREADS, thread_name_prefix="ml4-load")
    host: dict = {}
    # Lazy mode: without a mark_step every copy_ joins one pending graph for the whole load, and the bridge's
    # accumulation thread (AccThreadPool, one core) slows with its size (second start: 136 s per layer by layer 49,
    # read rate ~5 MB/s per rank, the worker's AccThreadPool at 100 %). One flush per layer keeps it flat.
    flush = None
    if model.norm.device.type == "hpu":
        import habana_frameworks.torch.core as htcore

        flush = htcore.mark_step
    from . import fp8lin

    for L, blk in enumerate(model.layers):
        load_layer(ck, blk, L, r, tp, st, pool, host)
        if flush is not None:
            flush()
        if fp8lin.ENABLED:  # ML4_FP8_LINEARS: FP8-sourced linears as 8-bit codes on the device (fp8lin.py)
            fp8lin.convert_layer(blk, st)
            if flush is not None:
                flush()
        logger.warning("ml4 load: rank %d layer %d done, %.1f GB read, %.0fs", r, L, ck.bytes / 1e9, time.time() - t0)
    pool.shutdown()
    st.update(bytes=ck.bytes, seconds=round(time.time() - t0, 1))
    if fp8lin.ENABLED:
        logger.warning("ml4 load: rank %d %s", r, fp8lin.summary(st))
    logger.warning(
        "ml4 load: rank %d done, %.1f GB read in %.0fs%s (fp8 scales not pow2: %d)",
        r,
        ck.bytes / 1e9,
        time.time() - t0,
        f" (TEST: {sorted(LOAD_SKIP)} left zero)" if LOAD_SKIP else "",
        st.get("fp8_scales_not_pow2", 0),
    )
    return st


def load_layer(ck: Ckpt, blk, L: int, r: int, tp: int, st: dict, pool=None, host: dict | None = None) -> None:
    """Every parameter of decoder layer L (stack.ML4Layer) of rank r of tp. host: reusable expert staging."""
    p, a, role = f"layers.{L}.", blk.attn, blk.role
    at = p + "attention."
    _put(blk.attn_norm, ck.tensor(p + "attention_norm.weight"))
    _put(blk.ffn_norm, ck.tensor(p + "ffn_norm.weight"))
    q = a.Hl * 128
    _put(a.wq.weight, fp8_block_bf16(ck, at + "wq", rows=(r * q, (r + 1) * q), stats=st))
    _put(a.wo.weight, fp8_block_bf16(ck, at + "wo", cols=(r * q, (r + 1) * q), stats=st))
    kvb = a.Hl * (a.P + 128)
    _put(a.wkv_b, ck.tensor(at + "wkv_b.weight", (r * kvb, (r + 1) * kvb)))
    _put(a.ssmax_s, ck.tensor(at + "ssmax_scaling.s")[r * a.Hl : (r + 1) * a.Hl])
    _put(a.ssmax_b, ck.tensor(at + "ssmax_scaling.b")[r * a.Hl : (r + 1) * a.Hl])
    if role.owns_kv:
        _put(a.wkv_a.weight, ck.tensor(at + "wkv_a_with_mqa.weight"))
        _put(a.kv_a_norm, ck.tensor(at + "kv_a_norm.weight"))
    else:
        assert not ck.has(at + "wkv_a_with_mqa.weight"), f"layer {L} is KV-shared but has wkv_a"
    if role.has_indexer:
        i, ix = a.indexer, at + "indexer."
        _put(i.q_norm_w, ck.tensor(ix + "q_input_norm.weight"))
        _put(i.q_norm_b, ck.tensor(ix + "q_input_norm.bias"))
        _put(i.wq_b.weight, fp8_block_bf16(ck, ix + "wq_b", stats=st))
        _put(i.wk.weight, ck.tensor(ix + "wk.weight"))
        _put(i.k_norm_w, ck.tensor(ix + "k_norm.weight").float())
        _put(i.k_norm_b, ck.tensor(ix + "k_norm.bias").float())
        _put(i.weights_proj.weight, ck.tensor(ix + "weights_proj.weight"))
        _put(i.ssmax_s, ck.tensor(ix + "ssmax_scaling.s"))
        _put(i.ssmax_b, ck.tensor(ix + "ssmax_scaling.b"))
    f = blk.ffn
    if role.is_moe:
        _put(f.gate_w, ck.tensor(p + "gate.weight"))
        _load_gated(ck, f.shared, p + "shared_experts", r, tp, st)
        if "experts" in LOAD_SKIP:
            for n in ("w13", "w13_scale", "w2", "w2_scale"):
                getattr(f, n).data.zero_()
        else:
            if host is None:
                host = {}
            if not host or tuple(host["w13"].shape) != tuple(f.w13.shape):
                host.update(expert_host(f))
            read_experts(ck, L, f.slots, host, pool)
            for n in ("w13", "w13_scale", "w2", "w2_scale"):
                _put(getattr(f, n), host[n])
    else:
        _load_gated(ck, f, p + "feed_forward", r, tp, st)


def _load_gated(ck: Ckpt, ffn, prefix: str, r: int, tp: int, st: dict) -> None:
    """[w1 local; w3 local] rows into ffn.w13, w2 columns into ffn.w2 (BF16 from block FP8)."""
    I = ck.shape(prefix + ".w1.weight")[0]
    n = I // tp
    w1 = fp8_block_bf16(ck, prefix + ".w1", rows=(r * n, (r + 1) * n), stats=st)
    w3 = fp8_block_bf16(ck, prefix + ".w3", rows=(r * n, (r + 1) * n), stats=st)
    _put(ffn.w13.weight, torch.cat([w1, w3]))
    _put(ffn.w2.weight, fp8_block_bf16(ck, prefix + ".w2", cols=(r * n, (r + 1) * n), stats=st))


# ------------------------------------------------------------------ ML4_LOAD_FAST=1: pipelined staging
def _fp8_into(ck: Ckpt, name: str, out: torch.Tensor, rows=None, cols=None, st=None, lock=None) -> torch.Tensor:
    """fp8_block_bf16(ck, name, rows, cols, st) written into `out` (BF16 [rows, cols], may be a row slice of a
    bigger tensor). Same e4m3 table lookup and the same fp32 product per element; the scale is broadcast over its
    128 x 128 block instead of materialized twice by repeat_interleave; one rounding to BF16 (copy_ = .to)."""
    R, C = ck.shape(name + ".weight")
    r0, r1 = rows or (0, R)
    c0, c1 = cols or (0, C)
    B = FP8_BLOCK
    assert r0 % B == 0 and c0 % B == 0 and (r1 - r0) % B == 0 and (c1 - c0) % B == 0
    w = ck.tensor(name + ".weight", (r0, r1))[:, c0:c1]
    s = ck.tensor(name + ".weight_scale", (r0 // B, r1 // B))[:, c0 // B : c1 // B]
    if st is not None:
        bits = s.contiguous().view(torch.int32)
        a, b, c = s.numel(), int(((bits & 0x7FFFFF) != 0).sum()), int(((w & 0x7F) == 0x7F).sum())
        with lock:
            st["fp8_scales"] = st.get("fp8_scales", 0) + a
            st["fp8_scales_not_pow2"] = st.get("fp8_scales_not_pow2", 0) + b
            st["fp8_codes_nan"] = st.get("fp8_codes_nan", 0) + c
    nr, nc = r1 - r0, c1 - c0
    v = e4m3_values()[w.to(torch.int32)]  # [nr, nc] fp32, contiguous
    v.view(nr // B, B, nc // B, B).mul_(s.reshape(nr // B, 1, nc // B, 1))
    assert tuple(out.shape) == (nr, nc), (name, tuple(out.shape), nr, nc)
    out.copy_(v)
    return out


def _read_halves(ck: Ckpt, name: str, outs: list) -> None:
    """read_col_parts(name, outs) for 2 parts: one contiguous read of the whole tensor, then a strided host copy of
    each wanted half (same bytes; no per-row iovecs)."""
    fn, off, nb, dt, shape = ck.meta[name]
    R = shape[0]
    k = len(outs)
    part = nb // R // k
    assert part * k * R == nb
    scratch = torch.empty((R, k, part), dtype=torch.uint8)
    ck._preadv(ck._fd(fn), [_np(scratch)], off)
    for j, o in enumerate(outs):
        if o is not None:
            o.view(torch.uint8).view(R, part).copy_(scratch[:, j])


class _HostRing:
    """Reusable expert staging dicts, keyed by shape; at most `n` per shape exist (bounds host memory)."""

    def __init__(self, n: int):
        self.n, self.cv, self.free, self.made = n, threading.Condition(), {}, {}

    def get(self, moe) -> dict:
        key = tuple(tuple(getattr(moe, x).shape) for x in ("w13", "w13_scale", "w2", "w2_scale"))
        with self.cv:
            while True:
                if self.free.get(key):
                    return self.free[key].pop()
                if self.made.get(key, 0) < self.n:
                    self.made[key] = self.made.get(key, 0) + 1
                    break
                self.cv.wait()
        h = {}
        for x, shp in zip(("w13", "w13_scale", "w2", "w2_scale"), key):
            t = None
            if PIN:
                try:
                    t = torch.empty(shp, dtype=torch.uint8, pin_memory=True)
                except Exception:  # no pinned allocator for this device: plain host memory
                    t = None
            h[x] = t if t is not None else torch.empty(shp, dtype=torch.uint8)
        h["_key"] = key
        return h

    def put(self, h: dict) -> None:
        with self.cv:
            self.free.setdefault(h["_key"], []).append(h)
            self.cv.notify_all()


def _stage_layer(ck: Ckpt, blk, L: int, r: int, tp: int, st: dict, lock, pool, ring: _HostRing) -> tuple:
    """Host tensors of every parameter of layer L, in load_layer's order: ([(param, host)], expert ring entry,
    timings). Runs on a coordinator thread; reads / conversions fan out to `pool` (never waits on itself)."""
    t0 = time.perf_counter()
    p, a, role = f"layers.{L}.", blk.attn, blk.role
    at = p + "attention."
    jobs = []  # (param, future -> host tensor)

    def sub(prm, fn, *args):
        jobs.append((prm, pool.submit(fn, *args)))

    def bf16(shape):
        return torch.empty(shape, dtype=torch.bfloat16)

    f = blk.ffn
    host = None
    if role.is_moe and "experts" not in LOAD_SKIP:
        # experts first: the biggest reads go to the pool's queue first
        host = ring.get(f)
        swap = M.FAULTS["swap_w13"]
        IV = host["w13"].shape[1] // 2
        assert IV == host["w13_scale"].shape[1] // 2
        first, second = ("w3", "w1") if swap else ("w1", "w3")
        ex, futs = f"{p}experts", []
        for v, (e, h) in enumerate(f.slots):
            for proj, dst in ((first, slice(0, IV)), (second, slice(IV, 2 * IV))):
                futs.append(
                    pool.submit(ck.read_rows, f"{ex}.{e}.{proj}.weight", host["w13"][v, dst], IV * h, IV * (h + 1))
                )
                futs.append(
                    pool.submit(
                        ck.read_rows, f"{ex}.{e}.{proj}.weight_scale", host["w13_scale"][v, dst], IV * h, IV * (h + 1)
                    )
                )
        by_e: dict[int, dict] = {}
        for v, (e, h) in enumerate(f.slots):
            by_e.setdefault(e, {})[h] = v
        for e, halves in by_e.items():
            for t, suffix in (("w2", "weight"), ("w2_scale", "weight_scale")):
                outs = [host[t][halves[h]] if h in halves else None for h in (0, 1)]
                futs.append(pool.submit(_read_halves, ck, f"{ex}.{e}.w2.{suffix}", outs))
        exp_futs = futs
    else:
        exp_futs = []
    sub(blk.attn_norm, ck.tensor, p + "attention_norm.weight")
    sub(blk.ffn_norm, ck.tensor, p + "ffn_norm.weight")
    q = a.Hl * 128
    sub(a.wq.weight, _fp8_into, ck, at + "wq", bf16(tuple(a.wq.weight.shape)), (r * q, (r + 1) * q), None, st, lock)
    sub(a.wo.weight, _fp8_into, ck, at + "wo", bf16(tuple(a.wo.weight.shape)), None, (r * q, (r + 1) * q), st, lock)
    kvb = a.Hl * (a.P + 128)
    sub(a.wkv_b, ck.tensor, at + "wkv_b.weight", (r * kvb, (r + 1) * kvb))
    sub(a.ssmax_s, lambda: ck.tensor(at + "ssmax_scaling.s")[r * a.Hl : (r + 1) * a.Hl])
    sub(a.ssmax_b, lambda: ck.tensor(at + "ssmax_scaling.b")[r * a.Hl : (r + 1) * a.Hl])
    if role.owns_kv:
        sub(a.wkv_a.weight, ck.tensor, at + "wkv_a_with_mqa.weight")
        sub(a.kv_a_norm, ck.tensor, at + "kv_a_norm.weight")
    else:
        assert not ck.has(at + "wkv_a_with_mqa.weight"), f"layer {L} is KV-shared but has wkv_a"
    if role.has_indexer:
        i, ix = a.indexer, at + "indexer."
        sub(i.q_norm_w, ck.tensor, ix + "q_input_norm.weight")
        sub(i.q_norm_b, ck.tensor, ix + "q_input_norm.bias")
        sub(i.wq_b.weight, _fp8_into, ck, ix + "wq_b", bf16(tuple(i.wq_b.weight.shape)), None, None, st, lock)
        sub(i.wk.weight, ck.tensor, ix + "wk.weight")
        sub(i.k_norm_w, lambda: ck.tensor(ix + "k_norm.weight").float())
        sub(i.k_norm_b, lambda: ck.tensor(ix + "k_norm.bias").float())
        sub(i.weights_proj.weight, ck.tensor, ix + "weights_proj.weight")
        sub(i.ssmax_s, ck.tensor, ix + "ssmax_scaling.s")
        sub(i.ssmax_b, ck.tensor, ix + "ssmax_scaling.b")

    def gated(ffn, prefix):
        I = ck.shape(prefix + ".w1.weight")[0]
        n = I // tp
        w13 = bf16(tuple(ffn.w13.weight.shape))
        assert w13.shape[0] == 2 * n
        f1 = pool.submit(_fp8_into, ck, prefix + ".w1", w13[:n], (r * n, (r + 1) * n), None, st, lock)
        f3 = pool.submit(_fp8_into, ck, prefix + ".w3", w13[n:], (r * n, (r + 1) * n), None, st, lock)
        jobs.append((ffn.w13.weight, _Both(f1, f3, w13)))
        sub(
            ffn.w2.weight,
            _fp8_into,
            ck,
            prefix + ".w2",
            bf16(tuple(ffn.w2.weight.shape)),
            None,
            (r * n, (r + 1) * n),
            st,
            lock,
        )

    if role.is_moe:
        sub(f.gate_w, ck.tensor, p + "gate.weight")
        gated(f.shared, p + "shared_experts")
    else:
        gated(f, p + "feed_forward")
    items = [(prm, fu.result()) for prm, fu in jobs]
    for fu in exp_futs:
        fu.result()
    if host is not None:
        items += [(getattr(f, x), host[x]) for x in ("w13", "w13_scale", "w2", "w2_scale")]
    return items, host, time.perf_counter() - t0


class _Both:
    """A future-like for the [w1; w3] tensor whose halves two pool jobs fill (no pool thread waits on another)."""

    def __init__(self, f1, f3, t):
        self.f1, self.f3, self.t = f1, f3, t

    def result(self):
        self.f1.result()
        self.f3.result()
        return self.t


def load_stack_fast(model, path: str, stats: dict | None = None) -> dict:
    """load_stack with ML4_LOAD_FAST=1: layer L+1..L+PREFETCH staged on the host while layer L is copied."""
    pools: list = []
    try:
        return _load_stack_fast(model, path, stats, pools)
    except BaseException:
        for p in pools:  # a failed layer must not leave reader / staging threads behind
            p.shutdown(wait=False, cancel_futures=True)
        raise


def _load_stack_fast(model, path: str, stats: dict | None, pools: list) -> dict:
    t0 = time.time()
    e4m3_values()  # build the table once, before the reader threads share it
    ck = Ckpt(path)
    st = stats if stats is not None else {}
    lock = threading.Lock()
    r, tp, Vp = model.tp_rank, model.tp_size, model.vp
    pool = cf.ThreadPoolExecutor(THREADS, thread_name_prefix="ml4-load")
    coord = cf.ThreadPoolExecutor(PREFETCH + 1, thread_name_prefix="ml4-stage")
    pools.extend((coord, pool))
    ring = _HostRing(PREFETCH + 2)
    flush = None
    if model.norm.device.type == "hpu":
        import habana_frameworks.torch.core as htcore

        flush = lambda: htcore.mark_step()  # noqa: E731  (looked up per call: tests wrap it)
    from . import fp8lin

    layers = list(model.layers)
    futs = {}

    def submit(L):
        if L < len(layers) and L not in futs:
            futs[L] = coord.submit(_stage_layer, ck, layers[L], L, r, tp, st, lock, pool, ring)

    for L in range(PREFETCH):
        submit(L)
    top = [
        pool.submit(ck.tensor, "tok_embeddings.weight", (r * Vp, (r + 1) * Vp)),
        pool.submit(lambda: ck.tensor("output.weight", (r * Vp, (r + 1) * Vp)).float()),
        pool.submit(ck.tensor, "norm.weight"),
        pool.submit(ck.tensor, "lerope.alpha"),
    ]
    for prm, fu in zip((model.embed, model.lm_head, model.norm, model.alpha), top):
        _put(prm, fu.result())
    del top
    prev = None
    T = {"wait": 0.0, "put": 0.0, "flush": 0.0, "stage": 0.0}
    for L, blk in enumerate(layers):
        submit(L + PREFETCH)
        a = time.perf_counter()
        items, host, t_stage = futs.pop(L).result()
        b = time.perf_counter()
        for prm, t in items:
            _put(prm, t)
        del items
        c = time.perf_counter()
        if flush is not None:
            flush()
        if fp8lin.ENABLED:
            fp8lin.convert_layer(blk, st)
            if flush is not None:
                flush()
        d = time.perf_counter()
        if prev is not None:  # layer L-1's staging: its copies were flushed one mark_step ago
            ring.put(prev)
        prev = host
        tim = {"wait": b - a, "put": c - b, "flush": d - c, "stage": t_stage}
        for k, v in tim.items():
            T[k] += v
        if LAYER_HOOK is not None:
            LAYER_HOOK(L, {k: round(v, 3) for k, v in tim.items()})
        logger.warning(
            "ml4 load: rank %d layer %d done, %.1f GB read, %.0fs (fast: wait %.1fs put %.1fs flush %.1fs)",
            r,
            L,
            ck.bytes / 1e9,
            time.time() - t0,
            b - a,
            c - b,
            d - c,
        )
    if "experts" in LOAD_SKIP:
        for blk in layers:
            if blk.role.is_moe:
                for n in ("w13", "w13_scale", "w2", "w2_scale"):
                    getattr(blk.ffn, n).data.zero_()
        if flush is not None:
            flush()
    coord.shutdown()
    pool.shutdown()
    st.update(bytes=ck.bytes, seconds=round(time.time() - t0, 1), fast={k: round(v, 1) for k, v in T.items()})
    if fp8lin.ENABLED:
        logger.warning("ml4 load: rank %d %s", r, fp8lin.summary(st))
    logger.warning(
        "ml4 load: rank %d done (fast, prefetch %d), %.1f GB read in %.0fs%s (fp8 scales not pow2: %d; "
        "main thread: wait %.0fs, put %.0fs, flush %.0fs)",
        r,
        PREFETCH,
        ck.bytes / 1e9,
        time.time() - t0,
        f" (TEST: {sorted(LOAD_SKIP)} left zero)" if LOAD_SKIP else "",
        st.get("fp8_scales_not_pow2", 0),
        T["wait"],
        T["put"],
        T["flush"],
    )
    return st
