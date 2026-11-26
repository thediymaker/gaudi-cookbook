"""Checkpoint access for the reference stack: Mistral-native tensor names, no mmap.

TensorSource reads safetensors shards (a single `*.safetensors`, or the shards listed by
`consolidated.safetensors.index.json` / `model.safetensors.index.json`) with pread()s into RAM
and drops the page cache behind each read (POSIX_FADV_DONTNEED), so streaming a 578 GB
checkpoint does not fill the container's memory cgroup with page cache. `preload()` reads many
tensors in parallel and keeps them in RAM until `evict()`.

Dequant rules:
  * FP8 e4m3 linears + fp32 block scale (`weight_scale` AMD, `weight_scale_inv` tiny,
    `qscale_weight` FP8 checkpoint) -> fp32, block size from the shapes;
  * routed experts: U8 MXFP4 + U8 E8M0 `weight_scale` (AMD, syn8l) -> fp32, or FP8 with the
    tiny model's stacked scales (`experts.gate_up_proj_scale_inv` [E, 2I/b, d/b], w1 rows first,
    `experts.down_proj_scale_inv` [E, d/b, I/b]);
  * BF16/F32 tensors -> fp32 (values unchanged).
"""

from __future__ import annotations

import concurrent.futures as cf
import glob
import json
import os
import struct
import threading

import torch

from . import numerics as N

F32 = torch.float32
_DT = {
    "U8": torch.uint8,
    "I8": torch.int8,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I32": torch.int32,
    "I64": torch.int64,
    "F8_E4M3": torch.float8_e4m3fn,
}
SCALE_SUFFIXES = ("weight_scale_inv", "weight_scale", "qscale_weight")


class TensorSource:
    def __init__(self, path: str, threads: int = 16, drop_cache: bool = True):
        self.path = path
        self.drop_cache = drop_cache
        files = None
        for idx in ("consolidated.safetensors.index.json", "model.safetensors.index.json"):
            p = os.path.join(path, idx)
            if os.path.exists(p):
                files = sorted(set(json.load(open(p))["weight_map"].values()))
                break
        if files is None:
            files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(path, "*.safetensors")))
        self.meta = {}  # name -> (file, start, end, dtype, shape)
        for fn in files:
            with open(os.path.join(path, fn), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            base = 8 + n
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.meta[k] = (fn, base + a, base + b, v["dtype"], v["shape"])
        self.mem: dict[str, torch.Tensor] = {}
        self.fds: dict[str, int] = {}
        self._lock = threading.Lock()
        self.pool = cf.ThreadPoolExecutor(threads, thread_name_prefix="pread")
        self.bytes_read = 0

    # ------------------------------------------------------------------ raw I/O
    def has(self, name: str) -> bool:
        return name in self.meta

    def names(self, prefix: str = "") -> list[str]:
        return [k for k in self.meta if k.startswith(prefix)]

    def _fd(self, fn):
        with self._lock:
            if fn not in self.fds:
                self.fds[fn] = os.open(os.path.join(self.path, fn), os.O_RDONLY)
            return self.fds[fn]

    def _pread(self, fd, buf, off):
        mv, done = memoryview(buf.numpy()), 0
        while done < len(mv):
            n = os.preadv(fd, [mv[done:]], off + done)
            if n <= 0:
                raise IOError(f"short read at {off + done}")
            done += n
        if self.drop_cache:
            try:
                os.posix_fadvise(fd, off, len(mv), os.POSIX_FADV_DONTNEED)
            except (OSError, AttributeError):
                pass

    def _read(self, name: str, piece: int = 64 << 20, inline: bool = False) -> torch.Tensor:
        fn, s, e, dt, shape = self.meta[name]
        fd = self._fd(fn)
        buf = torch.empty(e - s, dtype=torch.uint8)
        if inline:  # already on a pool thread
            if e > s:
                self._pread(fd, buf, s)
        else:
            futs = [
                self.pool.submit(self._pread, fd, buf[a : min(e - s, a + piece)], s + a) for a in range(0, e - s, piece)
            ]
            for f in futs:
                f.result()
        self.bytes_read += e - s
        return buf.view(_DT[dt]).reshape(shape) if e > s else torch.empty(shape, dtype=_DT[dt])

    def get(self, name: str) -> torch.Tensor:
        t = self.mem.get(name)
        return t if t is not None else self._read(name)

    def preload(self, names) -> int:
        names = [n for n in names if n not in self.mem]
        futs = {n: self.pool.submit(self._read, n, 0, True) for n in names}
        tot = 0
        for n, f in futs.items():
            self.mem[n] = f.result()
            tot += self.mem[n].numel() * self.mem[n].element_size()
        return tot

    def evict(self, prefix: str = "") -> None:
        for k in [k for k in self.mem if k.startswith(prefix)]:
            del self.mem[k]

    # -------------------------------------------------------------- dequantized
    def scale_name(self, base: str) -> str | None:
        for suf in SCALE_SUFFIXES:
            if f"{base}.{suf}" in self.meta:
                return f"{base}.{suf}"
        return None

    def f32(self, name: str) -> torch.Tensor:
        return self.get(name).to(F32)

    def linear(self, base: str) -> torch.Tensor:
        """`base` without `.weight` (e.g. 'layers.1.attention.wq') -> fp32 [out, in]."""
        w = self.get(base + ".weight")
        if w.dtype == torch.float8_e4m3fn:
            return N.fp8_block_dequant(w, self.get(self.scale_name(base)))
        assert w.dtype in (torch.bfloat16, F32, torch.float16), (base, w.dtype)
        return w.to(F32)


class ExpertSource:
    """Callable e -> (w1, w3, w2) fp32 for one MoE layer. Formats: 'mxfp4' (U8 + E8M0),
    'fp8_stacked' (tiny), 'fp8' (per-expert block scale), 'bf16'."""

    def __init__(self, src: TensorSource, layer: int, cache: bool = False, cache_dtype=torch.bfloat16):
        """cache=True keeps each dequantized expert (in `cache_dtype`; MXFP4/FP8 values are exact in
        bf16) so several runs over one layer dequantize it once."""
        self.src, self.p = src, f"layers.{layer}.experts."
        w = src.meta[f"{self.p}0.w1.weight"]
        if w[3] == "U8":
            self.fmt = "mxfp4"
        elif w[3] == "F8_E4M3":
            self.fmt = "fp8_stacked" if src.has(f"{self.p}gate_up_proj_scale_inv") else "fp8"
        else:
            self.fmt = "bf16"
        self.cache = {} if cache else None
        self.cache_dtype = cache_dtype

    def proj(self, e: int, name: str) -> torch.Tensor:
        n = f"{self.p}{e}.{name}.weight"
        w = self.src.get(n)
        if self.fmt == "mxfp4":
            return N.mxfp4_dequant(w, self.src.get(f"{self.p}{e}.{name}.weight_scale"))
        if self.fmt == "fp8_stacked":
            if name == "w2":
                sc = self.src.get(f"{self.p}down_proj_scale_inv")[e]
            else:
                gu = self.src.get(f"{self.p}gate_up_proj_scale_inv")[e]
                h = gu.shape[0] // 2
                sc = gu[:h] if name == "w1" else gu[h:]
            return N.fp8_block_dequant(w, sc)
        if self.fmt == "fp8":
            return N.fp8_block_dequant(w, self.src.get(self.src.scale_name(f"{self.p}{e}.{name}")))
        return w.to(F32)

    def __call__(self, e: int):
        if self.cache is not None and e in self.cache:
            return tuple(t.to(F32) for t in self.cache[e])
        out = tuple(self.proj(e, n) for n in ("w1", "w3", "w2"))
        if self.cache is not None:
            self.cache[e] = tuple(t.to(self.cache_dtype) for t in out)
        return out

    def names(self) -> list[str]:
        return self.src.names(self.p)
