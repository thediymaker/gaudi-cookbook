# SPDX-License-Identifier: Apache-2.0
"""Ranged per-rank safetensors loader for the Muse Spark text model (AMD Quark MXFP4 checkpoint).

vLLM's default iterator reads the whole 509 GB snapshot on every rank. This reads the shard headers, plans
exactly which tensors and byte ranges one TP / EP rank needs, and reads only those (70.8 GB per rank at TP 8
with EP2 x TP4, most of it the rank's slices of 128 experts per layer). The slices go straight into the model's
parameters. vLLM's weight iterator is never consumed.

Expert layers, decided per layer from dtypes (Quark's exclude list does not map onto these names):
  MXFP4  layers.L.mlp.experts.E.{gate,up,down}_proj.weight U8 + .weight_scale (E2M1 packed low nibble first,
         E8M0 scale bytes, group 32 along the input dim); every layer but the first and the last
  BF16   the same names without a scale (layers 0 and 61)
Packed weights and scales only ever travel as uint8: an int8 or float8_e8m0 copy on the HPU clamps bytes.

Per-rank parameters (prefix model.layers.L. for per-layer names). H hidden, Hm moe_hidden, I moe_intermediate,
E experts, tp / r = TP size / rank, s = MUSE_SPARK_MOE_TP_SPLIT, j = r % s, n = E * s / tp local experts,
e0 = (r // s) * n, q = heads / tp * head_dim and k = kv_heads / tp * head_dim rows per rank:

  model.embed_tokens.weight, lm_head.weight  [Vp, H]  BF16  vocab-parallel rows (padding 64), untied
  model.norm.weight                          [H]      F32   raw w (plain gain w + 0)
  input_layernorm / pre_feedforward_layernorm / post_feedforward_layernorm .weight  [H]  F32  raw w (gain w + 1)
  self_attn.qkv_proj.weight                  [q+2k, H] BF16 q_proj rows [q*r, +q), k_proj and v_proj rows [k*r, +k)
  self_attn.gate_proj.weight                 [q, H]   BF16  gate_proj rows [q*r, +q) (same heads as q)
  self_attn.o_proj.weight                    [H, q]   BF16  o_proj columns [q*r, +q)
  post_attention / post_feedforward _residual_gate.gate  [H]  F32  raw g (the model builds alpha, beta)
  mlp.gate.weight, mlp.gate.e_score_correction_bias      [E, H], [E]  F32  replicated
  mlp.pre_expert_proj.weight, mlp.post_expert_proj.weight [Hm, H], [H, Hm]  BF16  replicated
  mlp.pre_expert_norm.weight, mlp.post_expert_norm.weight [Hm]  F32  (the latter from mlp.experts.post_expert_norm)
  mlp.w13_weight [+ _scale]   MXFP4 [n, 2I/s, Hm/2] (+ [n, 2I/s, Hm/32])  U8   rows [0, I/s) gate rows
                              [I/s*j, +I/s), rows [I/s, 2I/s) up rows, same range: w13 = cat([gate; up])
  mlp.w2_weight [+ _scale]    MXFP4 [n, Hm, I/2/s] (+ [n, Hm, I/32/s])    U8   down byte columns [I/2/s*j, +I/2/s)
  mlp.w13_weight, w2_weight   BF16  [n, 2I/s, Hm], [n, Hm, I/s]               the same rule, unpacked

model.audio_tower.* and model.vision_* are skipped (the port serves text only).

No vLLM imports: the CPU tests and the headers-only dry run use this module without an engine:

  python3 -m muse_spark_hpu.loader CKPT_DIR --tp 8 --ep 2      # plan every rank, read no tensor data
"""

from __future__ import annotations

import json
import logging
import mmap
import os
import re
import struct
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("vllm.muse_spark_hpu")

TEXT = "model.language_model."
VOCAB_PAD = 64  # vLLM's DEFAULT_VOCAB_PADDING_SIZE
MXFP4_GROUP = 32
LOAD_WORKERS = 4
# Lazy mode queues every host-to-device copy until the next mark_step and holds a device staging copy of each
# source; over a whole rank that doubles HBM. Flush and wait every this many bytes.
SYNC_BYTES = 2e9
PROGRESS_BYTES = 10e9  # a progress line per 10 GB read

# safetensors dtype -> (torch dtype, itemsize)
_DT = {
    "BF16": (torch.bfloat16, 2),
    "F16": (torch.float16, 2),
    "F32": (torch.float32, 4),
    "U8": (torch.uint8, 1),
    "I8": (torch.int8, 1),
    "F8_E8M0": (torch.uint8, 1),
    "F8_E4M3": (torch.uint8, 1),
    "I32": (torch.int32, 4),
    "I64": (torch.int64, 8),
}
_TORCH_NAME = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.uint8: "U8"}
# conversions copy_into allows (source -> parameter); every one is exact
_LOSSLESS = {("BF16", "BF16"), ("F32", "F32"), ("U8", "U8"), ("BF16", "F32"), ("F16", "F32")}
SKIP = (
    ("audio", re.compile(r"^model\.audio_tower\.")),
    ("vision", re.compile(r"^model\.vision_(tower|adapter|projection)\.")),
)


# ------------------------------------------------------------------ dims / layout
@dataclass(frozen=True)
class TextDims:
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    layers: int
    vocab: int
    experts: int
    moe_hidden: int
    moe_inter: int

    @classmethod
    def from_config(cls, cfg: Any) -> TextDims:
        """cfg: config.json as a dict, or a transformers config object (top-level or text)."""

        def get(o, k):
            return o[k] if isinstance(o, Mapping) else getattr(o, k)

        def has(o, k):
            return (k in o) if isinstance(o, Mapping) else hasattr(o, k)

        t = get(cfg, "text_config") if has(cfg, "text_config") else cfg
        heads = int(get(t, "num_attention_heads"))
        hd = int(get(t, "head_dim")) if has(t, "head_dim") else int(get(t, "hidden_size")) // heads
        return cls(
            hidden=int(get(t, "hidden_size")),
            heads=heads,
            kv_heads=int(get(t, "num_key_value_heads")),
            head_dim=hd,
            layers=int(get(t, "num_hidden_layers")),
            vocab=int(get(t, "vocab_size")),
            experts=int(get(t, "num_local_experts")),
            moe_hidden=int(get(t, "moe_hidden_size")),
            moe_inter=int(get(t, "moe_intermediate_size")),
        )


@dataclass(frozen=True)
class Layout:
    """Rank placement: ep_size expert groups of s = tp_size // ep_size ranks; rank r is in group r // s and
    holds slice r % s of each local expert's intermediate dim. ep_size defaults to tp_size (whole experts)."""

    tp_size: int
    rank: int
    ep_size: int | None = None

    def __post_init__(self):
        ep = self.tp_size if self.ep_size is None else self.ep_size
        object.__setattr__(self, "ep_size", ep)
        if not (0 <= self.rank < self.tp_size) or ep < 1 or self.tp_size % ep:
            raise ValueError(f"bad layout tp={self.tp_size} ep={ep} rank={self.rank}")

    @property
    def moe_tp_split(self) -> int:
        return self.tp_size // self.ep_size

    @property
    def ep_rank(self) -> int:
        return self.rank // self.moe_tp_split

    @property
    def moe_tp_rank(self) -> int:
        return self.rank % self.moe_tp_split

    def local_experts(self, n_experts: int) -> tuple[int, int]:
        """(e0, n): this rank owns experts e0 .. e0 + n - 1."""
        if n_experts % self.ep_size:
            raise ValueError(f"{n_experts} experts do not divide over ep={self.ep_size}")
        n = n_experts // self.ep_size
        return self.ep_rank * n, n


def vocab_rows(vocab: int, tp_size: int, rank: int) -> tuple[int, int, int]:
    """(start, stop, rows_per_rank) as vLLM's VocabParallelEmbedding shards the vocab."""
    padded = -(-vocab // VOCAB_PAD) * VOCAB_PAD
    if padded % tp_size:
        raise ValueError(f"padded vocab {padded} does not divide over tp={tp_size}")
    per = padded // tp_size
    return min(per * rank, vocab), min(per * (rank + 1), vocab), per


# ------------------------------------------------------------------ checkpoint
@dataclass(frozen=True)
class TensorInfo:
    name: str
    file: str
    dtype: str
    shape: tuple[int, ...]
    start: int  # absolute byte offset of the data in the file
    nbytes: int


def _read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


class Checkpoint:
    """Safetensors snapshot (sharded with model.safetensors.index.json, or single-file). Parses every shard
    header and reads data with pread (contiguous ranges) or a read-only mmap (column slices)."""

    def __init__(self, path: str):
        self.path = path
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                self.weight_map: dict[str, str] | None = json.load(f)["weight_map"]
            files = sorted(set(self.weight_map.values()))
        else:
            self.weight_map = None
            files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
            if not files:
                raise FileNotFoundError(f"no safetensors in {path}")
        self.files = files
        self.tensors: dict[str, TensorInfo] = {}
        for fn in files:
            hdr, base = _read_header(os.path.join(path, fn))
            for name, v in hdr.items():
                if name == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                if name in self.tensors:
                    raise ValueError(f"{name} in both {self.tensors[name].file} and {fn}")
                self.tensors[name] = TensorInfo(name, fn, v["dtype"], tuple(v["shape"]), base + a, b - a)
        with open(os.path.join(path, "config.json")) as f:
            self.config = json.load(f)
        self._fds: dict[str, int] = {}
        self._mms: dict[str, mmap.mmap] = {}
        self._lock = threading.Lock()
        self.bytes_read = 0

    def header_problems(self) -> list[str]:
        """Index vs headers, and every header entry's byte size vs dtype x shape."""
        out = []
        if self.weight_map is not None and set(self.weight_map) != set(self.tensors):
            out.append("index and shard headers list different tensors")
        for t in self.tensors.values():
            if t.dtype not in _DT:
                out.append(f"{t.name}: unknown dtype {t.dtype}")
            elif int(np.prod(t.shape, dtype=np.int64)) * _DT[t.dtype][1] != t.nbytes:
                out.append(f"{t.name}: {t.nbytes} B for {t.dtype}{list(t.shape)}")
        return out

    def _fd(self, fn: str) -> int:
        with self._lock:
            if fn not in self._fds:
                self._fds[fn] = os.open(os.path.join(self.path, fn), os.O_RDONLY)
            return self._fds[fn]

    def _mm(self, fn: str) -> mmap.mmap:
        fd = self._fd(fn)
        with self._lock:
            if fn not in self._mms:
                self._mms[fn] = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            return self._mms[fn]

    def close(self) -> None:
        with self._lock:
            for m in self._mms.values():
                m.close()
            for fd in self._fds.values():
                os.close(fd)
            self._mms.clear()
            self._fds.clear()

    def read(self, name: str, sel: tuple[tuple[int, int], ...] | None = None, as_u8: bool = False) -> torch.Tensor:
        """The box `sel` (per-dim [start, stop)) of tensor `name`, as a new CPU tensor."""
        t = self.tensors[name]
        tdt, isz = _DT[t.dtype]
        if as_u8:
            if isz != 1:
                raise TypeError(f"{name}: {t.dtype} is not a byte tensor")
            tdt = torch.uint8
        shape = t.shape
        sel = tuple((0, d) for d in shape) if sel is None else tuple(sel)
        inside = all(0 <= a < b <= d for (a, b), d in zip(sel, shape, strict=True))
        assert len(sel) == len(shape) and inside, (name, sel, shape)
        out_shape = tuple(b - a for a, b in sel)
        nbytes = int(np.prod(out_shape, dtype=np.int64)) * isz
        strides = [isz] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        k = len(shape) - 1  # first dim after which everything is full
        while k > 0 and sel[k] == (0, shape[k]):
            k -= 1
        if all(b - a == 1 for a, b in sel[:k]):
            # one contiguous byte range
            off = t.start + sum(a * s for (a, _), s in zip(sel[: k + 1], strides[: k + 1], strict=False))
            buf = torch.empty(nbytes, dtype=torch.uint8)
            mv, fd, got = memoryview(buf.numpy()), self._fd(t.file), 0
            while got < nbytes:
                n = os.preadv(fd, [mv[got:]], off + got)
                if n <= 0:
                    raise OSError(f"short read on {t.file} at {off + got}")
                got += n
        else:
            # column slices: a read-only mmap touches only the pages that hold the box
            raw = np.frombuffer(self._mm(t.file), dtype=np.uint8, count=t.nbytes, offset=t.start)
            raw = raw.reshape(shape[:-1] + (shape[-1] * isz,))
            box = tuple(slice(a, b) for a, b in sel[:-1]) + (slice(sel[-1][0] * isz, sel[-1][1] * isz),)
            buf = torch.from_numpy(np.ascontiguousarray(raw[box])).reshape(-1)
            del raw
        with self._lock:
            self.bytes_read += nbytes
        return buf.view(tdt).reshape(out_shape)


# ------------------------------------------------------------------ plan
@dataclass(frozen=True)
class ParamSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    rows_valid: int | None = None  # vocab-parallel: rows past this are zero padding


@dataclass(frozen=True)
class Read:
    src: str  # checkpoint tensor
    sel: tuple[tuple[int, int], ...]  # box in the source, per dim [start, stop)
    dst: str  # parameter name
    at: tuple[tuple[int, int], ...]  # box in the parameter (same volume as sel)
    as_u8: bool
    nbytes: int


@dataclass
class RankPlan:
    layout: Layout
    dims: TextDims
    expert_kind: list[str]  # per layer: "mxfp4" | "bf16"
    params: dict[str, ParamSpec]
    reads: list[Read]
    skipped: dict[str, list[str]]  # rule -> checkpoint names never loaded
    not_mine: list[str]  # tensors owned by other ranks
    unknown: list[str]  # checkpoint tensors no rule claims (must be empty)
    missing: list[str]  # tensors the model needs that the checkpoint lacks

    @property
    def nbytes(self) -> int:
        return sum(r.nbytes for r in self.reads)


def _full(shape) -> tuple[tuple[int, int], ...]:
    return tuple((0, d) for d in shape)


def expert_kinds(ck: Checkpoint, dims: TextDims) -> list[str]:
    """Per layer "mxfp4" (U8 + .weight_scale) or "bf16" (no scale); every expert of a layer must agree."""
    T, kinds = ck.tensors, []
    for layer in range(dims.layers):
        p = f"{TEXT}layers.{layer}.mlp.experts."
        g0 = T.get(p + "0.gate_proj.weight")
        if g0 is None:
            raise ValueError(f"layer {layer}: no per-expert tensors under {p}")
        kind = "mxfp4" if g0.dtype == "U8" and p + "0.gate_proj.weight_scale" in T else "bf16"
        for e in range(dims.experts):
            for proj in ("gate", "up", "down"):
                w, s = T.get(f"{p}{e}.{proj}_proj.weight"), T.get(f"{p}{e}.{proj}_proj.weight_scale")
                if kind == "mxfp4":
                    ok = w is not None and w.dtype == "U8" and s is not None and s.dtype in ("U8", "F8_E8M0")
                else:
                    ok = w is not None and w.dtype == "BF16" and s is None
                if not ok:
                    raise ValueError(f"layer {layer} expert {e} {proj}_proj does not match the layer's {kind} layout")
        kinds.append(kind)
    return kinds


def param_specs(dims: TextDims, layout: Layout, kinds: list[str]) -> dict[str, ParamSpec]:
    """Every parameter the loader fills on this rank, with its per-rank shape and dtype."""
    D, tp, r = dims, layout.tp_size, layout.rank
    if D.heads % tp or D.kv_heads % tp:
        raise ValueError(f"heads {D.heads}/{D.kv_heads} do not divide over tp={tp}")
    q, k = D.heads // tp * D.head_dim, D.kv_heads // tp * D.head_dim
    H, Hm, inter, E = D.hidden, D.moe_hidden, D.moe_inter, D.experts
    s = layout.moe_tp_split
    _, n = layout.local_experts(E)
    if inter % s:
        raise ValueError(f"moe_intermediate {inter} does not split {s} ways")
    Is = inter // s
    a, b, per = vocab_rows(D.vocab, tp, r)
    out = {}

    def add(name, shape, dtype, rows_valid=None):
        out[name] = ParamSpec(name, tuple(shape), dtype, rows_valid)

    add("model.embed_tokens.weight", (per, H), "BF16", b - a)
    add("lm_head.weight", (per, H), "BF16", b - a)
    add("model.norm.weight", (H,), "F32")
    for layer in range(D.layers):
        p = f"model.layers.{layer}."
        add(p + "input_layernorm.weight", (H,), "F32")
        add(p + "self_attn.qkv_proj.weight", (q + 2 * k, H), "BF16")
        add(p + "self_attn.gate_proj.weight", (q, H), "BF16")
        add(p + "self_attn.o_proj.weight", (H, q), "BF16")
        add(p + "post_attention_residual_gate.gate", (H,), "F32")
        add(p + "pre_feedforward_layernorm.weight", (H,), "F32")
        add(p + "post_feedforward_layernorm.weight", (H,), "F32")
        add(p + "post_feedforward_residual_gate.gate", (H,), "F32")
        add(p + "mlp.gate.weight", (E, H), "F32")
        add(p + "mlp.gate.e_score_correction_bias", (E,), "F32")
        add(p + "mlp.pre_expert_proj.weight", (Hm, H), "BF16")
        add(p + "mlp.pre_expert_norm.weight", (Hm,), "F32")
        add(p + "mlp.post_expert_norm.weight", (Hm,), "F32")
        add(p + "mlp.post_expert_proj.weight", (H, Hm), "BF16")
        if kinds[layer] == "mxfp4":
            if Hm % MXFP4_GROUP or Is % MXFP4_GROUP:
                raise ValueError(f"layer {layer}: MXFP4 needs Hm and I/s multiples of {MXFP4_GROUP}")
            add(p + "mlp.w13_weight", (n, 2 * Is, Hm // 2), "U8")
            add(p + "mlp.w13_weight_scale", (n, 2 * Is, Hm // MXFP4_GROUP), "U8")
            add(p + "mlp.w2_weight", (n, Hm, Is // 2), "U8")
            add(p + "mlp.w2_weight_scale", (n, Hm, Is // MXFP4_GROUP), "U8")
        else:
            add(p + "mlp.w13_weight", (n, 2 * Is, Hm), "BF16")
            add(p + "mlp.w2_weight", (n, Hm, Is), "BF16")
    return out


def plan_rank(
    ck: Checkpoint, layout: Layout, *, dims: TextDims | None = None, kinds: list[str] | None = None
) -> RankPlan:
    """Everything rank `layout.rank` reads, from the headers alone (no tensor data)."""
    D = dims or TextDims.from_config(ck.config)
    kinds = kinds or expert_kinds(ck, D)
    specs = param_specs(D, layout, kinds)
    T, tp, r = ck.tensors, layout.tp_size, layout.rank
    q, k = D.heads // tp * D.head_dim, D.kv_heads // tp * D.head_dim
    inter, Hm, H = D.moe_inter, D.moe_hidden, D.hidden
    s, j = layout.moe_tp_split, layout.moe_tp_rank
    Is = inter // s
    e0, n = layout.local_experts(D.experts)
    reads: list[Read] = []
    expected: set[str] = set()  # every checkpoint name the model consumes (on any rank)
    missing: list[str] = []
    shape_errors: list[str] = []

    def need(name):
        expected.add(name)
        return name

    def rd(src, sel, dst, at=None, as_u8=False, want=None):
        """Plan one read; `want` is the full source shape the model expects."""
        if src not in T:
            missing.append(src)
            return
        t = T[src]
        if want is not None and tuple(want) != t.shape:
            shape_errors.append(f"{src}: checkpoint {list(t.shape)}, model expects {list(want)}")
            return
        sel = _full(t.shape) if sel is None else tuple(sel)
        at = _full(specs[dst].shape) if at is None else tuple(at)
        vol = int(np.prod([b - a for a, b in sel], dtype=np.int64))
        assert vol == int(np.prod([b - a for a, b in at], dtype=np.int64)), (src, sel, dst, at)
        reads.append(Read(src, sel, dst, at, as_u8, vol * _DT[t.dtype][1]))

    a, b, _ = vocab_rows(D.vocab, tp, r)
    rd(
        need(TEXT + "embed_tokens.weight"),
        ((a, b), (0, H)),
        "model.embed_tokens.weight",
        ((0, b - a), (0, H)),
        want=(D.vocab, H),
    )
    rd(need("lm_head.weight"), ((a, b), (0, H)), "lm_head.weight", ((0, b - a), (0, H)), want=(D.vocab, H))
    rd(need(TEXT + "norm.weight"), None, "model.norm.weight", want=(H,))

    for layer in range(D.layers):
        src, dst = f"{TEXT}layers.{layer}.", f"model.layers.{layer}."
        nq, nk = D.heads * D.head_dim, D.kv_heads * D.head_dim
        qkv = dst + "self_attn.qkv_proj.weight"
        rd(need(src + "self_attn.q_proj.weight"), ((q * r, q * r + q), (0, H)), qkv, ((0, q), (0, H)), want=(nq, H))
        rd(need(src + "self_attn.k_proj.weight"), ((k * r, k * r + k), (0, H)), qkv, ((q, q + k), (0, H)), want=(nk, H))
        rd(
            need(src + "self_attn.v_proj.weight"),
            ((k * r, k * r + k), (0, H)),
            qkv,
            ((q + k, q + 2 * k), (0, H)),
            want=(nk, H),
        )
        rd(
            need(src + "self_attn.gate_proj.weight"),
            ((q * r, q * r + q), (0, H)),
            dst + "self_attn.gate_proj.weight",
            want=(nq, H),
        )
        rd(
            need(src + "self_attn.o_proj.weight"),
            ((0, H), (q * r, q * r + q)),
            dst + "self_attn.o_proj.weight",
            want=(H, nq),
        )
        for nm in (
            "input_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "post_attention_residual_gate.gate",
            "post_feedforward_residual_gate.gate",
        ):
            rd(need(src + nm), None, dst + nm, want=(H,))
        rd(need(src + "mlp.gate.weight"), None, dst + "mlp.gate.weight", want=(D.experts, H))
        rd(
            need(src + "mlp.gate.e_score_correction_bias"),
            None,
            dst + "mlp.gate.e_score_correction_bias",
            want=(D.experts,),
        )
        rd(need(src + "mlp.pre_expert_proj.weight"), None, dst + "mlp.pre_expert_proj.weight", want=(Hm, H))
        rd(need(src + "mlp.post_expert_proj.weight"), None, dst + "mlp.post_expert_proj.weight", want=(H, Hm))
        rd(need(src + "mlp.pre_expert_norm.weight"), None, dst + "mlp.pre_expert_norm.weight", want=(Hm,))
        rd(need(src + "mlp.experts.post_expert_norm.weight"), None, dst + "mlp.post_expert_norm.weight", want=(Hm,))

        ex, w13, w2 = src + "mlp.experts.", dst + "mlp.w13_weight", dst + "mlp.w2_weight"
        mx = kinds[layer] == "mxfp4"
        cols = (Hm // 2, Hm // MXFP4_GROUP) if mx else (Hm,)  # w13 source columns: weight[, scale]
        dcols = (inter // 2, inter // MXFP4_GROUP) if mx else (inter,)  # w2 source columns
        suffixes = ("weight", "weight_scale") if mx else ("weight",)
        for e in range(D.experts):
            for sfx, c, dc in zip(suffixes, cols, dcols, strict=False):
                names = [need(f"{ex}{e}.{pj}_proj.{sfx}") for pj in ("gate", "up", "down")]
                if not e0 <= e < e0 + n:
                    continue
                el = e - e0
                tail = "_scale" if sfx == "weight_scale" else ""
                for half, nm in enumerate(names[:2]):  # w13 = cat([gate; up], 0)
                    rd(
                        nm,
                        ((Is * j, Is * j + Is), (0, c)),
                        w13 + tail,
                        ((el, el + 1), (half * Is, half * Is + Is), (0, c)),
                        mx,
                        want=(inter, c),
                    )
                dcs = dc // s
                rd(
                    names[2],
                    ((0, Hm), (dcs * j, dcs * j + dcs)),
                    w2 + tail,
                    ((el, el + 1), (0, Hm), (0, dcs)),
                    mx,
                    want=(Hm, dc),
                )

    if shape_errors:
        raise ValueError("checkpoint shapes do not match the config:\n  " + "\n  ".join(shape_errors[:20]))
    read_srcs = {x.src for x in reads}
    skipped: dict[str, list[str]] = defaultdict(list)
    not_mine, unknown = [], []
    for name in sorted(T):
        if name in read_srcs:
            continue
        if name in expected:
            not_mine.append(name)
            continue
        rule = next((label for label, rx in SKIP if rx.match(name)), None)
        if rule is None:
            unknown.append(name)
        else:
            skipped[rule].append(name)
    return RankPlan(layout, D, kinds, specs, reads, dict(skipped), not_mine, unknown, missing)


# ------------------------------------------------------------------ coverage checks
def _vol(box) -> int:
    return int(np.prod([b - a for a, b in box], dtype=np.int64))


def _overlap(x, y) -> bool:
    return all(a1 < b2 and a2 < b1 for (a1, b1), (a2, b2) in zip(x, y, strict=False))


def _tiles(boxes, full) -> bool:
    """True iff `boxes` are pairwise disjoint, inside `full`, and cover it exactly."""
    if any(not all(0 <= a < b <= d for (a, b), (_, d) in zip(bx, full, strict=False)) for bx in boxes):
        return False
    if sum(_vol(bx) for bx in boxes) != _vol(full):
        return False
    srt = sorted(boxes)
    for i, bx in enumerate(srt):  # sorted by dim-0 start: only boxes starting before bx ends can overlap
        k = i + 1
        while k < len(srt) and srt[k][0][0] < bx[0][1]:
            if _overlap(bx, srt[k]):
                return False
            k += 1
    return True


def param_problems(plan: RankPlan) -> list[str]:
    """Each parameter written exactly once everywhere (vocab padding rows excepted)."""
    by_dst: dict[str, list[Read]] = defaultdict(list)
    for x in plan.reads:
        by_dst[x.dst].append(x)
    out = []
    for name, spec in plan.params.items():
        rs = by_dst.pop(name, [])
        full = _full(spec.shape)
        if spec.rows_valid is not None:
            full = ((0, spec.rows_valid),) + full[1:]
        if not _tiles([x.at for x in rs], full):
            out.append(f"{name}: {len(rs)} reads do not tile {list(spec.shape)} exactly")
    out += [f"{d}: read into a parameter that has no spec" for d in by_dst]
    return out


def coverage(ck: Checkpoint, plans: list[RankPlan]) -> dict:
    """Across all ranks: every needed tensor is read whole by every rank (replicated) or partitioned so that
    each element is read by exactly one rank."""
    boxes: dict[str, list[tuple[int, tuple]]] = defaultdict(list)
    for p in plans:
        for x in p.reads:
            boxes[x.src].append((p.layout.rank, x.sel))
    R = len({p.layout.rank for p in plans})
    kinds: Counter = Counter()
    bad = []
    for name, lst in boxes.items():
        full = _full(ck.tensors[name].shape)
        per_rank: dict[int, list] = defaultdict(list)
        for rk, bx in lst:
            per_rank[rk].append(bx)
        if len(per_rank) == R and all(_tiles(v, full) for v in per_rank.values()):
            kinds["replicated" if R > 1 else "whole"] += 1
        elif _tiles([bx for _, bx in lst], full):
            kinds["partitioned"] += 1
        else:
            bad.append(name)
    skipped_names = {n for v in plans[0].skipped.values() for n in v}
    return {
        "tensors_in_checkpoint": len(ck.tensors),
        "tensors_consumed": len(boxes),
        "consumed_by_kind": dict(kinds),
        "bad_coverage": bad,
        "skipped": {k: len(v) for k, v in plans[0].skipped.items()},
        "unconsumed": sorted(set(ck.tensors) - set(boxes) - skipped_names),
        "unknown": sorted({u for p in plans for u in p.unknown}),
        "missing": sorted({m for p in plans for m in p.missing}),
    }


def dry_run(path: str, tp_size: int, ep_size: int | None = None) -> dict:
    """Headers only: plan every rank; check the index, per-rank parameter tiling, global coverage and bytes."""
    t0 = time.time()
    ck = Checkpoint(path)
    D = TextDims.from_config(ck.config)
    kinds = expert_kinds(ck, D)
    plans = [plan_rank(ck, Layout(tp_size, r, ep_size), dims=D, kinds=kinds) for r in range(tp_size)]
    rep = {
        "path": path,
        "tp": tp_size,
        "ep": plans[0].layout.ep_size,
        "header_problems": ck.header_problems(),
        "expert_kind": dict(Counter(kinds)),
        "param_problems": {p.layout.rank: q for p in plans if (q := param_problems(p))},
        "rank_bytes": [p.nbytes for p in plans],
    }
    rep.update(coverage(ck, plans))
    rep["data_bytes_read"] = ck.bytes_read
    rep["seconds"] = round(time.time() - t0, 2)
    rep["ok"] = not (
        rep["header_problems"]
        or rep["param_problems"]
        or rep["bad_coverage"]
        or rep["unconsumed"]
        or rep["unknown"]
        or rep["missing"]
        or ck.bytes_read
    )
    ck.close()
    return rep


# ------------------------------------------------------------------ load
def iter_reads(
    ck: Checkpoint, reads: Iterable[Read], workers: int = LOAD_WORKERS, window: int = 8
) -> Iterator[tuple[Read, torch.Tensor]]:
    """Read in (file, offset) order with a few reads in flight (pread releases the GIL)."""
    order = sorted(reads, key=lambda x: (ck.tensors[x.src].file, ck.tensors[x.src].start, x.sel))
    if workers <= 1:
        for x in order:
            yield x, ck.read(x.src, x.sel, x.as_u8)
        return
    with ThreadPoolExecutor(workers, thread_name_prefix="muse-load") as pool:
        it = iter(order)
        pending = [(x, pool.submit(ck.read, x.src, x.sel, x.as_u8)) for _, x in zip(range(window), it, strict=False)]
        while pending:
            x, fut = pending.pop(0)
            nxt = next(it, None)
            if nxt is not None:
                pending.append((nxt, pool.submit(ck.read, nxt.src, nxt.sel, nxt.as_u8)))
            yield x, fut.result()


def copy_into(param: torch.Tensor, x: Read, t: torch.Tensor) -> None:
    """Copy one planned read into its box of `param`; only exact dtype changes, made on the CPU."""
    src, dst = _TORCH_NAME.get(t.dtype), _TORCH_NAME.get(param.dtype)
    if (src, dst) not in _LOSSLESS:
        raise TypeError(f"{x.src} -> {x.dst}: {t.dtype} into {param.dtype} is not an exact copy")
    if t.dtype != param.dtype:
        t = t.to(param.dtype)
    box = param.data[tuple(slice(a, b) for a, b in x.at)]
    box.copy_(t.reshape(box.shape))


def load_into(params: Mapping[str, torch.Tensor], path: str, layout: Layout, *, workers: int = LOAD_WORKERS) -> dict:
    """Fill `params` (name -> parameter) with this rank's slices; vocab padding rows are zeroed."""
    t0 = time.time()
    ck = Checkpoint(path)
    plan = plan_rank(ck, layout)
    problems = param_problems(plan)
    if plan.unknown or plan.missing or problems:
        raise ValueError(
            f"muse_spark_hpu loader: unknown {plan.unknown[:5]} missing {plan.missing[:5]} tiling {problems[:5]}"
        )
    bad = [
        f"{n}: need {list(s.shape)}, model has {list(params[n].shape) if n in params else 'nothing'}"
        for n, s in plan.params.items()
        if n not in params or tuple(params[n].shape) != s.shape
    ]
    if bad:
        raise ValueError("muse_spark_hpu loader: parameter mismatch:\n  " + "\n  ".join(bad[:20]))
    for n, s in plan.params.items():
        if s.rows_valid is not None and s.rows_valid < s.shape[0]:
            params[n].data[s.rows_valid :].zero_()
    pending, flush, report_at = 0, None, PROGRESS_BYTES
    for x, t in iter_reads(ck, plan.reads, workers=workers):
        p = params[x.dst]
        copy_into(p, x, t)
        if p.device.type == "hpu":
            if flush is None:
                import habana_frameworks.torch as htorch

                def flush():
                    htorch.core.mark_step()
                    htorch.hpu.synchronize()

            pending += t.numel() * t.element_size()
            if pending >= SYNC_BYTES:
                flush()
                pending = 0
        del t
        if ck.bytes_read >= report_at:
            logger.warning(
                "muse_spark load: rank %d %.0f / %.0f GB, %.0f s",
                layout.rank,
                ck.bytes_read / 1e9,
                plan.nbytes / 1e9,
                time.time() - t0,
            )
            report_at += PROGRESS_BYTES
    if flush is not None:
        flush()
    dt = time.time() - t0
    assert ck.bytes_read == plan.nbytes, (ck.bytes_read, plan.nbytes)
    logger.warning(
        "muse_spark load: rank %d (tp %d ep %d) %.2f GB in %.0fs; expert layers %s",
        layout.rank,
        layout.tp_size,
        layout.ep_size,
        plan.nbytes / 1e9,
        dt,
        dict(Counter(plan.expert_kind)),
    )
    ck.close()
    return {"rank": layout.rank, "bytes": plan.nbytes, "seconds": dt, "loaded": set(plan.params)}


def checkpoint_dir(model: str) -> str:
    """vLLM's model argument as a local snapshot directory (a path, or a repo id in the offline HF cache)."""
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model, local_files_only=True)


def specs_for(path: str, layout: Layout) -> tuple[dict[str, ParamSpec], list[str]]:
    """(parameter specs, per-layer expert kind) from the headers, for the model's constructor."""
    ck = Checkpoint(path)
    try:
        D = TextDims.from_config(ck.config)
        kinds = expert_kinds(ck, D)
        return param_specs(D, layout, kinds), kinds
    finally:
        ck.close()


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Headers-only plan of every rank (no tensor data is read)")
    ap.add_argument("path")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--ep", type=int, default=None, help="expert groups (tp / MUSE_SPARK_MOE_TP_SPLIT)")
    a = ap.parse_args()
    rep = dry_run(a.path, a.tp, a.ep)
    print(json.dumps(rep, indent=1, default=list))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
