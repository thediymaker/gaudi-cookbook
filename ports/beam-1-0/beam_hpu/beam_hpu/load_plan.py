# SPDX-License-Identifier: Apache-2.0
"""Read planning, ranged parallel reads and batched expert staging for load format `beam` (loader.py).

Pure torch and stdlib, no vLLM import, so the CPU tests exercise the same code the loader runs.

Per rank:
  * routed-expert gate_proj / up_proj weights (2/3 of the checkpoint): only the rows the rank keeps
    are read, [r * n / tp, (r + 1) * n / tp) of the expert's MoE TP group (the narrow FusedMoE's
    _load_w13 takes); down_proj weights are read whole (the rank keeps a column strip, which spans
    every page); under expert parallelism, experts of other ranks are not read at all;
  * the Gaudi 2 FP8 weight rescale (`(w.float() * 0.5).to(fp8)`, vllm-gaudi's gaudi_weight_wrapper)
    is applied to the kept rows or strip only, in the reader threads;
  * each MoE layer's weight and scale stacks are assembled on the host in the parameters' exact layout
    and copied to the device once per parameter (4 synced copies per layer) instead of once per
    expert tensor.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass

import torch

DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BOOL": torch.bool,
}
EXPERT = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale)$"
)


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def read_header(path: str) -> tuple[int, dict]:
    """(byte offset of the data section, {name: header entry}) of a safetensors file."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return 8 + n, h


def gaudi2_rescale(w: torch.Tensor) -> torch.Tensor:
    """vllm-gaudi's gaudi_weight_wrapper on an FP8 weight (Gaudi 2 has the smaller e4m3 range)."""
    return (w.float() * 0.5).to(torch.float8_e4m3fn)


@dataclass
class Job:
    name: str
    path: str
    dtype: torch.dtype
    shape: tuple
    off: int  # absolute file offset of the tensor's first byte
    nbytes: int  # the whole tensor
    rows: tuple | None  # (a, b): read only rows [a, b) of dim 0; None = read everything
    rescale: tuple | None  # ("rows", a, b) | ("cols", a, b) | ("all",): region to pre-apply the Gaudi 2
    # rescale to; None = the stock wrapper handles the tensor
    exp: tuple | None = None  # routed expert: (layer, local expert id, proj, "weight" | "weight_scale")

    @property
    def read_bytes(self) -> int:
        if self.rows is None:
            return self.nbytes
        a, b = self.rows
        return (b - a) * (self.nbytes // self.shape[0])


def plan_jobs(files: list[str], moe: dict, rescale: bool, file_group: int = 1) -> tuple[list[Job], int]:
    """Read jobs and the number of expert tensors skipped (EP).

    `moe` maps a MoE layer id to (MoE TP size, MoE TP rank, global -> local expert id function, -1 for
    an expert of another rank). Order, the same on every rank: files in natural order, taken
    `file_group` at a time and interleaved tensor by tensor within a group, so the reads in flight span
    several files (a network file system often serves one cold file much slower than several). Bytes
    every rank reads (down_proj under TP, dense tensors) are read from storage once per node: the ranks
    read them at about the same time and share them through the page cache."""
    per_file, skipped = [], 0
    for path in sorted(files, key=natural_key):
        jobs = []
        per_file.append(jobs)
        base, h = read_header(path)
        for name, m in sorted(h.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            a, b = m["data_offsets"]
            dt = DTYPES[m["dtype"]]
            shape = tuple(m["shape"])
            rows = rs = None
            e = EXPERT.match(name)
            if e is not None:
                layer, eid, proj, kind = int(e.group(1)), int(e.group(2)), e.group(3), e.group(4)
                if layer not in moe:
                    raise RuntimeError(f"beam_hpu loader: expert tensor {name} for a non-MoE layer")
                tp, mrank, to_local = moe[layer]
                local = to_local(eid)
                if local == -1:  # owned by another EP rank: never read
                    skipped += 1
                    continue
                if kind == "weight" and dt == torch.float8_e4m3fn:
                    if proj in ("gate_proj", "up_proj"):  # FusedMoE _load_w13: narrow(0, rank * n / tp, n / tp)
                        n = shape[0]
                        per = n // tp
                        r0, r1 = per * mrank, min(per * mrank + per, n)
                        if r1 <= r0:
                            raise RuntimeError(f"beam_hpu loader: empty TP slice for {name}")
                        rows = None if (r0, r1) == (0, n) else (r0, r1)
                        rs = ("rows", r0, r1) if rows is not None else ("all",)
                    else:  # _load_w2: narrow(1, rank * n / tp, n / tp)
                        n = shape[1]
                        per = n // tp
                        c0, c1 = per * mrank, min(per * mrank + per, n)
                        if c1 <= c0:
                            raise RuntimeError(f"beam_hpu loader: empty TP slice for {name}")
                        rs = ("all",) if (c0, c1) == (0, n) else ("cols", c0, c1)
                    if not rescale:
                        rs = None
            exp = (layer, local, proj, kind) if e is not None else None
            jobs.append(Job(name, path, dt, shape, base + a, b - a, rows, rs, exp))
    out = []
    g = max(1, file_group)
    for i in range(0, len(per_file), g):
        group = [list(reversed(js)) for js in per_file[i : i + g]]
        while any(group):
            for js in group:
                if js:
                    out.append(js.pop())
    return out, skipped


class ExpertStage:
    """Batched expert placement for one MoE layer.

    The stock path calls the FusedMoE weight loader once per expert tensor (6 per expert, 1536 per
    layer, about 75,000 per rank for Beam at TP 8); vLLM's HPU platform follows every weight loader call
    with a device synchronize, and each call is a separate small lazy-mode copy from the worker's single
    host thread. Here the reader threads write every expert slice into host staging tensors shaped like
    the layer's parameters (w13_weight [E, 2I, H] fp8, w2_weight [E, H, I] fp8, w13_weight_scale
    [E, 2I, 1] fp32, w2_weight_scale [E, H, 1] fp32), with FusedMoE's placement (gate rows -> w13[e, :I],
    up rows -> w13[e, I:], down column strip -> w2[e]) and the vllm-gaudi wrapper's Gaudi 2 transform
    (FP8 x 0.5, done by the reader job; bf16 scales x 2.0, then cast to fp32 as the stock copy into the
    fp32 parameter does). A parameter that has all its tensors is copied to the device once. A layer
    whose parameters do not have exactly this layout is left to the stock path (check())."""

    PARAMS = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")

    def __init__(self, routed, scale_x2: bool):
        self.re = routed
        self.scale_x2 = scale_x2
        self.lock = threading.Lock()
        self.cpu: dict[str, torch.Tensor] = {}
        self.need: dict[str, int] = {}
        self.got: dict[str, int] = {}
        e = routed.local_num_experts
        for n in self.PARAMS:
            self.need[n] = e * (2 if n.startswith("w13") else 1)
            self.got[n] = 0

    @staticmethod
    def check(routed) -> str | None:
        """None if the layer's parameters have the layout this class writes, else why not."""
        e = getattr(routed, "local_num_experts", None)
        i_tp = getattr(routed, "intermediate_size_per_partition", None)
        h = getattr(routed, "hidden_size", None)
        if e is None or i_tp is None or h is None:
            return "missing local_num_experts/intermediate_size_per_partition/hidden_size"
        fp8, f32 = torch.float8_e4m3fn, torch.float32
        want = {
            "w13_weight": ((e, 2 * i_tp, h), fp8),
            "w2_weight": ((e, h, i_tp), fp8),
            "w13_weight_scale": ((e, 2 * i_tp, 1), f32),
            "w2_weight_scale": ((e, h, 1), f32),
        }
        for n, (shape, dt) in want.items():
            p = getattr(routed, n, None)
            if p is None or tuple(p.shape) != shape or p.dtype != dt:
                return f"{n}: {None if p is None else (tuple(p.shape), p.dtype)} != {(shape, dt)}"
            if getattr(p, "is_transposed", False):
                return f"{n} is_transposed"
        if not routed.moe_config.is_act_and_mul:
            return "not gated"
        return None

    def _buf(self, n: str) -> torch.Tensor:
        with self.lock:
            t = self.cpu.get(n)
            if t is None:
                p = getattr(self.re, n)
                t = self.cpu[n] = torch.empty(tuple(p.shape), dtype=p.dtype)
            return t

    def place(self, j: Job, t: torch.Tensor) -> None:
        """Reader thread: write tensor t (full checkpoint shape, FP8 region already rescaled) into staging."""
        _, e, proj, kind = j.exp
        if kind == "weight":
            if proj == "down_proj":
                c0, c1 = j.rescale[1:] if j.rescale is not None and j.rescale[0] == "cols" else (0, t.shape[1])
                self._buf("w2_weight")[e].copy_(t[:, c0:c1])
            else:
                buf = self._buf("w13_weight")
                half = buf.shape[1] // 2
                r0, r1 = j.rows if j.rows is not None else (0, t.shape[0])
                dst = buf[e, :half] if proj == "gate_proj" else buf[e, half:]
                dst[: r1 - r0].copy_(t[r0:r1])
        else:
            s = t * 2.0 if self.scale_x2 else t  # a bf16 op, as the wrapper's loaded_weight.data * 2.0
            if proj == "down_proj":
                self._buf("w2_weight_scale")[e].copy_(s)
            else:
                buf = self._buf("w13_weight_scale")
                half = buf.shape[1] // 2
                tp = self.re.moe_config.moe_parallel_config.tp_size
                r0 = (s.shape[0] // tp) * self.re.moe_config.tp_rank
                dst = buf[e, :half] if proj == "gate_proj" else buf[e, half:]
                dst.copy_(s[r0 : r0 + half])

    def done(self, j: Job) -> int:
        """Consumer thread: count j; copy every completed parameter to the device. Returns bytes copied."""
        _, _, proj, kind = j.exp
        n = ("w2" if proj == "down_proj" else "w13") + ("_weight" if kind == "weight" else "_weight_scale")
        self.got[n] += 1
        if self.got[n] < self.need[n]:
            return 0
        with self.lock:
            t = self.cpu.pop(n)
        p = getattr(self.re, n)
        p.data.copy_(t)
        if p.device.type == "hpu":
            torch.hpu.synchronize()
        return t.numel() * t.element_size()

    def complete(self) -> bool:
        return all(self.got[n] == self.need[n] for n in self.PARAMS)


class Reader:
    """Thread pool reading jobs into host tensors (positional reads straight into the tensor memory),
    at most `window` bytes in flight, handed back in submission order. Staged expert tensors are placed
    by the reader thread and handed back as None."""

    def __init__(self, jobs: list[Job], threads: int, window: int, stages: dict | None = None):
        self.jobs = jobs
        self.stages = stages or {}
        self.threads = threads
        self.window = window
        self.fds: dict[str, int] = {}
        self.lock = threading.Lock()
        self.t_read = 0.0
        self.t_rescale = 0.0
        self.bytes = 0

    def _fd(self, path: str) -> int:
        with self.lock:
            fd = self.fds.get(path)
            if fd is None:
                fd = self.fds[path] = os.open(path, os.O_RDONLY)
            return fd

    @staticmethod
    def _pread_into(fd: int, mv: memoryview, off: int) -> None:
        done, n = 0, len(mv)
        while done < n:
            k = os.preadv(fd, [mv[done:]], off + done)
            if k <= 0:
                raise OSError(f"short read at {off + done}")
            done += k

    def _run(self, j: Job) -> torch.Tensor | None:
        t0 = time.perf_counter()
        t = torch.empty(j.shape, dtype=j.dtype)
        if j.nbytes:
            mv = memoryview(t.view(torch.uint8).reshape(-1).numpy())
            fd = self._fd(j.path)
            if j.rows is None:
                self._pread_into(fd, mv, j.off)
            else:
                rb = j.nbytes // j.shape[0]
                a, b = j.rows
                self._pread_into(fd, mv[a * rb : b * rb], j.off + a * rb)
        t1 = time.perf_counter()
        if j.rescale is not None:
            kind = j.rescale[0]
            if kind == "all":
                t = gaudi2_rescale(t)
            elif kind == "rows":
                _, a, b = j.rescale
                t[a:b] = gaudi2_rescale(t[a:b])
            else:
                _, a, b = j.rescale
                t[:, a:b] = gaudi2_rescale(t[:, a:b])
        st = self.stages.get(j.exp[0]) if j.exp is not None else None
        if st is not None:
            st.place(j, t)
            t = None
        t2 = time.perf_counter()
        with self.lock:
            self.t_read += t1 - t0
            self.t_rescale += t2 - t1
            self.bytes += j.read_bytes
        return t

    def __iter__(self):
        q: deque = deque()
        inflight = 0
        i = 0
        with cf.ThreadPoolExecutor(self.threads, thread_name_prefix="beam-load") as ex:
            try:
                while i < len(self.jobs) or q:
                    while i < len(self.jobs) and (not q or inflight + self.jobs[i].read_bytes <= self.window):
                        j = self.jobs[i]
                        q.append((j, ex.submit(self._run, j)))
                        inflight += j.read_bytes
                        i += 1
                    j, fu = q.popleft()
                    t = fu.result()
                    inflight -= j.read_bytes
                    yield j, t
            finally:
                for _, fu in q:
                    fu.cancel()
                for fd in self.fds.values():
                    os.close(fd)
                self.fds.clear()
