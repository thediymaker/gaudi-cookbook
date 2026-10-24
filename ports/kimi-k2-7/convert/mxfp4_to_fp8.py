#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Re-encode Kimi-K2.7 MXFP4 routed experts as per-channel FP8 for the Gaudi FP8 MoE op, exactly.

Lossless: a re-encoding, not a quantization. The input is the output of repack_quark_mxfp4.py (vLLM's
gpt_oss_mxfp4 layout). Routed experts are fused per layer under <P>.layers.L.mlp.experts.routed_experts:
w13_weight = concat(gate, up) U8 [384, 4096, 3584] + w13_weight_scale E8M0 [384, 4096, 224],
w2_weight U8 [384, 7168, 1024] + w2_weight_scale E8M0 [384, 7168, 64], E2M1 packed low nibble first.
Every other tensor is already bf16 or f32.

The output is a compressed-tensors checkpoint (format float-quantized; weights FP8 e4m3 per output channel;
activations FP8 dynamic per token) that stock vllm-gaudi loads through HPUCompressedTensorsW8A8Fp8MoEMethod
(VllmMixtureOfExpertsOpFP8PerChannel), with no loader patch. Why per-channel and not block FP8: see
docs/ports/kimi-k2-7.md.

  routed experts   per expert, un-fused, under the names DeepSeek-V3's expert_params_mapping expects
                   (gate -> w1, up -> w3, down -> w2):
                     <P>.layers.L.mlp.experts.E.{gate,up}_proj.weight  F8_E4M3 [2048, 7168]
                     <P>.layers.L.mlp.experts.E.down_proj.weight       F8_E4M3 [7168, 2048]
                     <P>.layers.L.mlp.experts.E.*_proj.weight_scale    F32 [rows, 1], a power of two
                   Per-expert names also let `--enable-ep-weight-filter` skip non-local experts before reading them.
  everything else  raw byte passthrough (dtype, shape and bytes unchanged)

Why it is exact. An MXFP4 element is e2m1 (one mantissa bit, exponent -1..2) times 2^(s-127). Within one output
channel (a row) the E8M0 spread is at most 9 on every channel of the released checkpoint, so a row's nonzero elements
span at most 12 binades. With the row scale 2^k, k = (exponent of the row's largest element) - 7, every element lands
on an e4m3 normal with exponent in [-5, 7]: representable exactly, |q| <= 192. The window is one binade narrower
than e4m3's normals on purpose: on Gaudi 2 the loader halves FP8 weights when scale adjustment is on (vllm-gaudi
gaudi_weight_wrapper), and a -5 floor keeps the halved copy a normal too (at -6 it would land on e4m3 subnormals:
exact on the CPU, since e2m1 only yields mantissas 0 and 4, but Gaudi 2's FP8 subnormal handling is unverified).
Gaudi 2 reads e4m3fn bits with a maximum of 240 (exponent field 15, i.e. 256..448, is not finite there); those codes
are never emitted, nor 0x80 (-0 becomes +0).

The encode is a byte LUT (nibble, block shift) -> e4m3fn byte, built in pure Python and tested exhaustively without
torch (tests/test_fp8_exact.py). Every expert chunk is then decoded again in float64 (the MXFP4 side through the e2m1
table, the FP8 side through a 256-entry decode table) and must compare equal, or the worker exits 1.
validate_fp8.py re-checks the written checkpoint with its own decoders.

Tensors are read as raw bytes from the safetensors files and written by a small writer here: nothing goes through
a torch float8 dtype, so the installed safetensors and torch versions do not matter.

Modes:
  --layers 1-6   convert one range of expert layers; output shards are namespaced by the range
  --rest         copy the non-expert tensors
  --finalize     after every worker has finished: completeness gates, index, config, aux files
  --self-test    torch LUT path against the pure-Python reference, synthetic and real rows; writes nothing

To parallelise, run one process per layer range (ranges must cover every expert layer, 1-60 for Kimi-K2.7) and one
--rest process, then --finalize once all of them have finished; any job runner works. Each worker writes a manifest
last, so finalize refuses to run on a missing or unfinished worker. Then check the result with validate_fp8.py.

Usage:
  mxfp4_to_fp8.py --src SRC --dst DST --layers 1-6 [--threads 16]
  mxfp4_to_fp8.py --src SRC --dst DST --rest
  mxfp4_to_fp8.py --src SRC --dst DST --finalize
  mxfp4_to_fp8.py --src SRC --self-test [--layers 30-30]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import mmap
import os
import random
import re
import shutil
import struct
import time
from fractions import Fraction

# ----------------------------------------------------------------- numerics
# Pure Python on purpose: the tests import this module without torch or numpy. torch is imported only inside
# TorchCodec and the IO helpers that need it.

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
# floor(log2|v|) per nibble; None for +-0.
E2M1_EXP = (None, -1, 0, 0, 1, 1, 2, 2, None, -1, 0, 0, 1, 1, 2, 2)
BLOCK = 32  # MXFP4 elements per E8M0 scale
E8M0_BIAS = 127
E8M0_NAN = 255
FP8_EXP_MAX = 7  # Gaudi 2 e4m3: the top finite binade is 2^7 (max 240)
FP8_EXP_MIN_NORMAL = -6  # e4m3 smallest normal exponent
DEFAULT_MIN_EXP = -5  # one binade of headroom so the Gaudi 2 x0.5 load stays normal
D_MIN, D_MAX = -40, 40  # LUT range of the per-block shift d = s - 127 - k
ND = D_MAX - D_MIN + 1
K_LO, K_HI = -126, 127  # the row scale 2^k must be an fp32 normal
POW2_LO, POW2_HI = -200, 200
NO_EXP = -10_000  # "no nonzero element" sentinel in the vectorised path

EXPERT_TARGET = r"re:.*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"
# Redundant on purpose: the target above already matches routed experts only. vLLM applies `ignore` to its own
# module names (regex entries are not rewritten by KimiK25's hf_to_vllm_mapper), and checks fused modules
# (fused_qkv_a_proj, gate_up_proj) shard by shard through packed_modules_mapping.
IGNORE = [
    r"re:.*lm_head$",
    r"re:.*\.self_attn\..*",
    r"re:.*\.shared_experts\..*",
    r"re:.*\.mlp\.(gate_proj|up_proj|gate_up_proj|down_proj)$",
    r"re:.*\.mlp\.gate$",
    r"re:^vision_tower\..*",
    r"re:^mm_projector\..*",
]

_FUSED_RE = re.compile(
    r"^(?P<pfx>.+)\.layers\.(?P<L>\d+)\.mlp\.experts\.routed_experts\."
    r"(?P<w>w13_weight|w13_weight_scale|w2_weight|w2_weight_scale)$"
)
_PROJ = ("gate_proj", "up_proj", "down_proj")


class InexactError(RuntimeError):
    """A value would not survive the re-encoding bit-exactly."""


def floor_log2(a: Fraction) -> int:
    """floor(log2(a)) for a positive Fraction, exactly."""
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    return e


def e4m3fn_decode(b: int) -> Fraction | None:
    """e4m3fn byte -> exact value (bias 7, subnormals at exponent field 0, no inf, 0x7F/0xFF NaN -> None).

    Independent of the encoder below."""
    sign = -1 if b & 0x80 else 1
    e, m = (b >> 3) & 0xF, b & 0x7
    if e == 0xF and m == 0x7:
        return None
    if e == 0:
        return sign * Fraction(m, 8) * Fraction(1, 64)
    return sign * (1 + Fraction(m, 8)) * Fraction(2) ** (e - 7)


def e4m3fn_encode_normal(v: Fraction, min_exp: int = FP8_EXP_MIN_NORMAL) -> int | None:
    """Exact e4m3fn byte for v as a normal with exponent in [min_exp, FP8_EXP_MAX], or None if v needs rounding,
    a subnormal, or the top binade Gaudi 2 cannot read. 0 -> 0x00."""
    if v == 0:
        return 0x00
    sign = 0x80 if v < 0 else 0x00
    a = abs(v)
    e = floor_log2(a)
    if not (max(min_exp, FP8_EXP_MIN_NORMAL) <= e <= FP8_EXP_MAX):
        return None
    m8 = (a / Fraction(2) ** e - 1) * 8
    if m8.denominator != 1:
        return None
    return sign | ((e + 7) << 3) | int(m8)


def build_lut(min_exp: int = DEFAULT_MIN_EXP) -> tuple[bytes, bytes]:
    """(byte, ok) tables indexed nib * ND + (d - D_MIN): the e4m3fn encoding of E2M1[nib] * 2^d.

    Zero nibbles (+0 and -0) encode to 0x00 at any shift."""
    out, ok = bytearray(16 * ND), bytearray(16 * ND)
    for n in range(16):
        for d in range(D_MIN, D_MAX + 1):
            i = n * ND + (d - D_MIN)
            v = Fraction(E2M1[n]) * Fraction(2) ** d
            b = e4m3fn_encode_normal(v, min_exp)
            if b is not None:
                out[i], ok[i] = b, 1
    return bytes(out), bytes(ok)


FP8_DECODE = tuple(float("nan") if (x := e4m3fn_decode(b)) is None else float(x) for b in range(256))


def reference_encode_row(nibbles, scales, min_exp=DEFAULT_MIN_EXP, lut=None):
    """Pure-Python reference for one output channel: (e4m3fn bytes, k).

    k = (exponent of the largest element) - 7, so the row's top element lands in the top binade; all-zero rows get
    k = 0. Raises InexactError if any element would not be exact. TorchCodec.encode is the vectorised twin and must
    agree byte for byte.
    """
    if len(nibbles) != len(scales) * BLOCK:
        raise ValueError("row length != blocks * 32")
    lut_b, lut_ok = lut if lut is not None else build_lut(min_exp)
    emax = None
    for b, s in enumerate(scales):
        if s == E8M0_NAN:
            raise InexactError("E8M0 NaN (0xFF) scale")
        es = [E2M1_EXP[n] for n in nibbles[b * BLOCK : (b + 1) * BLOCK] if E2M1_EXP[n] is not None]
        if es:
            top = max(es) + s - E8M0_BIAS
            emax = top if emax is None else max(emax, top)
    k = 0 if emax is None else emax - FP8_EXP_MAX
    if not (K_LO <= k <= K_HI):
        raise InexactError(f"row scale 2^{k} outside fp32 normals")
    out = []
    for b, s in enumerate(scales):
        di = min(max(s - E8M0_BIAS - k - D_MIN, 0), ND - 1)
        for j, n in enumerate(nibbles[b * BLOCK : (b + 1) * BLOCK]):
            i = n * ND + di
            if not lut_ok[i]:
                raise InexactError(
                    f"block {b} elem {j}: {E2M1[n]} * 2^{s - E8M0_BIAS} "
                    f"is not an e4m3 normal in [2^{min_exp}, 2^{FP8_EXP_MAX}] "
                    f"after the row scale 2^{k}"
                )
            out.append(lut_b[i])
    return out, k


def unpack_row(packed: bytes) -> list[int]:
    """U8 row -> nibbles, element 0 in the low nibble (the layout repack_quark_mxfp4.py writes)."""
    out = []
    for byte in packed:
        out.append(byte & 0x0F)
        out.append(byte >> 4)
    return out


def pack_row(nibbles: list[int]) -> bytes:
    return bytes(nibbles[i] | (nibbles[i + 1] << 4) for i in range(0, len(nibbles), 2))


def synthetic_row(rng: random.Random, cols: int, spread: int, base: int | None = None):
    """Random row with an exact E8M0 spread; some all-zero and -0-heavy blocks."""
    nb = cols // BLOCK
    base = rng.randint(100, 135) if base is None else base
    scales = [base + rng.randint(0, spread) for _ in range(nb)]
    scales[rng.randrange(nb)] = base + spread
    scales[rng.randrange(nb)] = base
    nib = [rng.randrange(16) for _ in range(cols)]
    for _ in range(max(1, nb // 16)):
        b = rng.randrange(nb)
        fill = rng.choice((0, 8))
        nib[b * BLOCK : (b + 1) * BLOCK] = [fill] * BLOCK
    return nib, scales


def adversarial_row(cols: int, spread: int, base: int = 120):
    """6.0 in a block at scale base+spread and 0.5 in a block at base: the widest row a spread allows
    (spread + 3 binades between the largest and the smallest element)."""
    nb = cols // BLOCK
    scales = [base] * nb
    nib = [0] * cols
    scales[0] = base + spread
    nib[0:BLOCK] = [7] * BLOCK  # 6.0
    nib[BLOCK : 2 * BLOCK] = [9] * BLOCK  # -0.5
    return nib, scales


def parse_layers(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def build_quant_config() -> dict:
    """compressed-tensors config: FP8 per-channel weights, FP8 dynamic per-token activations, routed experts only.

    `fp8_dtype_flavor` is the vllm-gaudi switch (HPUCompressedTensorsConfig._update_scale_adjustment_if_needed)
    that turns the Gaudi 2 x0.5 load off in the process that parses the config. Set VLLM_SCALE_ADJUSTMENT=false in
    the serving environment too, since Ray workers get a pickled config and never run that hook. The checkpoint is
    exact either way."""
    return {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "quantization_status": "compressed",
        "fp8_dtype_flavor": "float8_e4m3fnuz",
        "config_groups": {
            "routed_experts": {
                "targets": [EXPERT_TARGET],
                "weights": {"num_bits": 8, "type": "float", "symmetric": True, "strategy": "channel", "dynamic": False},
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "token",
                    "dynamic": True,
                },
                "format": "float-quantized",
            },
        },
        "ignore": list(IGNORE),
    }


def ct_match(name: str, target: str) -> bool:
    """vLLM compressed_tensors.utils._is_equal_or_regex_match (no substring mode)."""
    if target.startswith("re:"):
        return re.match(target[3:], name) is not None
    return name == target


def check_targets(names) -> tuple[int, int]:
    """Every routed-expert module must hit the target and miss `ignore`; no other Linear-shaped weight may hit the
    target. Returns the (expert, other) weights counted."""
    n_exp = n_other = 0
    for n in names:
        if not n.endswith(".weight"):
            continue
        mod = n[: -len(".weight")]
        hit = ct_match(mod, EXPERT_TARGET)
        if re.search(r"\.mlp\.experts\.\d+\.", n):
            if not hit or any(ct_match(mod, t) for t in IGNORE):
                raise SystemExit(f"FATAL: expert module {mod} not targeted (or ignored)")
            n_exp += 1
        else:
            if hit:
                raise SystemExit(f"FATAL: non-expert module {mod} matches the FP8 target")
            n_other += 1
    return n_exp, n_other


# ----------------------------------------------------------------------- IO
class Shard:
    """Read-only raw view of one safetensors file."""

    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "rb")  # noqa: SIM115 (kept open for the mmap's lifetime)
        n = struct.unpack("<Q", self._f.read(8))[0]
        self.header = json.loads(self._f.read(n))
        self.header.pop("__metadata__", None)
        self._base = 8 + n
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

    def meta(self, name: str) -> tuple[str, list[int]]:
        m = self.header[name]
        return m["dtype"], m["shape"]

    def raw(self, name: str, start: int = 0, length: int | None = None) -> bytes:
        a, b = self.header[name]["data_offsets"]
        if length is None:
            length = b - a - start
        if start < 0 or start + length > b - a:
            raise ValueError(f"{name}: byte range out of bounds")
        return self._mm[self._base + a + start : self._base + a + start + length]


class Source:
    def __init__(self, root: str):
        self.root = root
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self._shards: dict[str, Shard] = {}

    def shard(self, name: str) -> Shard:
        fn = self.weight_map[name]
        if fn not in self._shards:
            self._shards[fn] = Shard(os.path.join(self.root, fn))
        return self._shards[fn]

    def meta(self, name: str) -> tuple[str, list[int]]:
        return self.shard(name).meta(name)

    def raw(self, name: str) -> bytes:
        return self.shard(name).raw(name)

    def expert_rows(self, name: str, e: int, r0: int, r1: int) -> bytes:
        """Rows [r0, r1) of expert e of a fused U8 [E, R, C] tensor, as bytes."""
        dt, (E, R, C) = self.meta(name)
        assert dt == "U8", (name, dt)
        return self.shard(name).raw(name, (e * R + r0) * C, (r1 - r0) * C)

    def experts_u8(self, name: str, e0: int, e1: int):
        """Experts [e0, e1) of a fused U8 [E, R, C] tensor as a uint8 torch tensor (copy)."""
        import torch

        dt, (E, R, C) = self.meta(name)
        assert dt == "U8", (name, dt)
        buf = bytearray(self.shard(name).raw(name, e0 * R * C, (e1 - e0) * R * C))
        return torch.frombuffer(buf, dtype=torch.uint8).reshape(e1 - e0, R, C)

    def fused_layers(self) -> tuple[str, dict[int, int]]:
        """(text-model prefix, {layer: n_experts}) from the fused routed-expert names."""
        pfx: set[str] = set()
        layers: dict[int, int] = {}
        for n in self.weight_map:
            m = _FUSED_RE.match(n)
            if m and m.group("w") == "w13_weight":
                pfx.add(m.group("pfx"))
                layers[int(m.group("L"))] = self.meta(n)[1][0]
        if len(pfx) != 1:
            raise SystemExit(f"FATAL: expected one expert prefix, got {sorted(pfx)}")
        return pfx.pop(), layers


_ITEMSIZE = {"F8_E4M3": 1, "U8": 1, "BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4}


class ShardWriter:
    """Buffers tensors as bytes, writes model-<tag>-NNNNN.safetensors (tmp + fsync + rename) once the buffer
    passes the limit, and keeps the name -> file manifest."""

    def __init__(self, dst: str, tag: str, limit_gb: float):
        self.dst, self.tag = dst, tag
        self.limit = int(limit_gb * (1024**3))
        self.buf: list[tuple[str, str, list[int], bytes]] = []
        self.bytes = 0
        self.idx = 0
        self.written: dict[str, str] = {}

    def clean_stale(self) -> None:
        """A restarted worker rewrites its own files; drop a previous attempt's."""
        for p in glob.glob(os.path.join(self.dst, f"model-{self.tag}-*.safetensors*")) + glob.glob(
            os.path.join(self.dst, f"_*-{self.tag}.json")
        ):
            os.remove(p)

    def emit(self, name: str, dtype: str, shape: list[int], data) -> None:
        if not isinstance(data, bytes | bytearray | memoryview):
            data = data.contiguous().numpy().tobytes()
        if len(data) != math.prod(shape) * _ITEMSIZE[dtype]:
            raise ValueError(f"{name}: {len(data)} bytes for {dtype}{shape}")
        self.buf.append((name, dtype, list(shape), bytes(data)))
        self.bytes += len(data)
        if self.bytes >= self.limit:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        fn = f"model-{self.tag}-{self.idx:05d}.safetensors"
        header, off = {}, 0
        for name, dtype, shape, data in self.buf:
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
            off += len(data)
        header["__metadata__"] = {"format": "pt"}
        hb = json.dumps(header, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)
        path = os.path.join(self.dst, fn)
        with open(path + ".tmp", "wb") as f:
            f.write(struct.pack("<Q", len(hb)))
            f.write(hb)
            for _, _, _, data in self.buf:
                f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(path + ".tmp", path)
        for name, *_ in self.buf:
            self.written[name] = fn
        print(f"[fp8] wrote {fn} ({self.bytes / 1e9:.1f} GB, {len(self.buf)} tensors)", flush=True)
        self.buf, self.bytes = [], 0
        self.idx += 1

    def save_manifest(self, stats: dict) -> None:
        # The manifest is the worker's completion marker, so it is written last.
        with open(os.path.join(self.dst, f"_stats-{self.tag}.json"), "w") as f:
            json.dump(stats, f, indent=1)
        with open(os.path.join(self.dst, f"_manifest-{self.tag}.json"), "w") as f:
            json.dump(self.written, f)


# -------------------------------------------------------------------- torch
class TorchCodec:
    """Vectorised twin of reference_encode_row (torch, CPU)."""

    def __init__(self, min_exp: int = DEFAULT_MIN_EXP):
        import torch

        self.torch = torch
        self.min_exp = min_exp
        lut_b, lut_ok = build_lut(min_exp)
        self.lut_bytes = torch.tensor(list(lut_b), dtype=torch.uint8)
        self.lut_ok = torch.tensor(list(lut_ok), dtype=torch.bool)
        self.e2m1_exp = torch.tensor([NO_EXP if x is None else x for x in E2M1_EXP], dtype=torch.int64)
        self.e2m1 = torch.tensor(E2M1, dtype=torch.float64)
        self.fp8 = torch.tensor(FP8_DECODE, dtype=torch.float64)
        self.pow2 = torch.tensor([math.ldexp(1.0, e) for e in range(POW2_LO, POW2_HI + 1)], dtype=torch.float64)

    def encode(self, w, s):
        """w uint8 [n, rows, cols/2], s uint8 [n, rows, cols/32] ->
        (q uint8 [n, rows, cols] e4m3fn bytes, scale f32 [n, rows, 1], stats)."""
        torch = self.torch
        n, rows, half = w.shape
        nb = s.shape[-1]
        if tuple(s.shape[:2]) != (n, rows) or nb * BLOCK != 2 * half:
            raise ValueError(f"weight {tuple(w.shape)} vs scale {tuple(s.shape)}")
        nib = torch.stack((w & 0x0F, w >> 4), dim=-1).reshape(n, rows, nb, BLOCK).long()
        sl = s.long()
        if bool((sl == E8M0_NAN).any()):
            raise InexactError("E8M0 NaN (0xFF) scale")
        top = self.e2m1_exp[nib].amax(-1) + sl - E8M0_BIAS  # [n, rows, nb]
        emax = top.amax(-1)  # [n, rows]
        zero = emax < NO_EXP // 2
        k = torch.where(zero, torch.zeros_like(emax), emax - FP8_EXP_MAX)
        if bool(((k < K_LO) | (k > K_HI)).any()):
            raise InexactError(f"row scale exponent outside [{K_LO}, {K_HI}]: {int(k.min())}..{int(k.max())}")
        di = (sl - E8M0_BIAS - k.unsqueeze(-1) - D_MIN).clamp(0, ND - 1)  # [n, rows, nb]
        idx = nib * ND + di.unsqueeze(-1)  # [n, rows, nb, 32]
        ok = self.lut_ok[idx]
        if not bool(ok.all()):
            e, r, b, j = (~ok).nonzero()[0].tolist()
            raise InexactError(
                f"{int((~ok).sum())} elements not exact; first: expert+{e} "
                f"row {r} block {b} elem {j} nibble {int(nib[e, r, b, j])} "
                f"scale {int(sl[e, r, b])} row-k {int(k[e, r])}"
            )
        q = self.lut_bytes[idx]
        # Independent re-decode in float64; must be equal (-0 == +0).
        mx = self.e2m1[nib] * self.pow2[sl - E8M0_BIAS - POW2_LO].unsqueeze(-1)
        f8 = self.fp8[q.long()] * self.pow2[k - POW2_LO][:, :, None, None]
        if not torch.equal(mx, f8):
            bad = (mx != f8).nonzero()[0].tolist()
            raise InexactError(
                f"float64 re-decode mismatch at {bad}: {float(mx[tuple(bad)])} vs {float(f8[tuple(bad)])}"
            )
        del mx, f8, idx, ok
        q = q.reshape(n, rows, 2 * half)
        scale = self.pow2[k - POW2_LO].to(torch.float32).unsqueeze(-1)
        qa = q & 0x7F
        nz = qa != 0
        qe = (qa >> 3).long()
        stats = {
            "k_min": int(k.min()),
            "k_max": int(k.max()),
            "q_exp_min": int(torch.where(nz, qe, 99).min()) - 7 if bool(nz.any()) else None,
            "q_exp_max": int(torch.where(nz, qe, -99).max()) - 7 if bool(nz.any()) else None,
            "zero_rows": int(zero.sum()),
            "neg_zero": int((nib == 8).sum()),
        }
        if stats["q_exp_min"] is not None and stats["q_exp_min"] < self.min_exp:
            raise InexactError(f"emitted exponent {stats['q_exp_min']} below {self.min_exp}")
        if bool(((q == 0x80) | ((qa >> 3) == 0xF)).any()):
            raise InexactError("emitted 0x80 or an exponent-15 code")
        return q, scale, stats


def _merge(acc: dict, st: dict) -> None:
    for key, fn in (("k_min", min), ("k_max", max), ("q_exp_min", min), ("q_exp_max", max)):
        if st[key] is not None:
            acc[key] = st[key] if acc.get(key) is None else fn(acc[key], st[key])
    for key in ("zero_rows", "neg_zero"):
        acc[key] = acc.get(key, 0) + st[key]


# ------------------------------------------------------------------- passes
def convert_layers(src: Source, w: ShardWriter, layers: list[int], codec: TorchCodec, chunk: int) -> dict:
    pfx, fused = src.fused_layers()
    todo = [L for L in layers if L in fused]
    missing = [L for L in layers if L not in fused]
    if missing:
        raise SystemExit(f"FATAL: no fused experts for layers {missing}")
    print(f"[fp8] layers {todo} (prefix '{pfx}')", flush=True)
    stats: dict = {}
    for L in todo:
        t0 = time.time()
        base = f"{pfx}.layers.{L}.mlp.experts.routed_experts"
        w13, w13s, w2, w2s = (f"{base}.{x}" for x in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"))
        E, R13, C13 = src.meta(w13)[1]
        _, R2, C2 = src.meta(w2)[1]
        inter, hidden = R13 // 2, 2 * C13
        if (
            hidden != R2
            or inter != 2 * C2
            or src.meta(w13s)[1] != [E, R13, hidden // BLOCK]
            or src.meta(w2s)[1] != [E, R2, inter // BLOCK]
        ):
            raise SystemExit(f"FATAL: layer {L}: unexpected fused shapes")
        st13: dict = {}
        st2: dict = {}
        for e0 in range(0, E, chunk):
            e1 = min(E, e0 + chunk)
            q13, s13, a = codec.encode(src.experts_u8(w13, e0, e1), src.experts_u8(w13s, e0, e1))
            q2, s2, b = codec.encode(src.experts_u8(w2, e0, e1), src.experts_u8(w2s, e0, e1))
            _merge(st13, a)
            _merge(st2, b)
            for j, e in enumerate(range(e0, e1)):
                p = f"{pfx}.layers.{L}.mlp.experts.{e}"
                # w13 rows [0, I) are gate (w1) and [I, 2I) are up (w3): the concatenated layout of the MXFP4 repack.
                w.emit(f"{p}.gate_proj.weight", "F8_E4M3", [inter, hidden], q13[j, :inter])
                w.emit(f"{p}.gate_proj.weight_scale", "F32", [inter, 1], s13[j, :inter])
                w.emit(f"{p}.up_proj.weight", "F8_E4M3", [inter, hidden], q13[j, inter:])
                w.emit(f"{p}.up_proj.weight_scale", "F32", [inter, 1], s13[j, inter:])
                w.emit(f"{p}.down_proj.weight", "F8_E4M3", [hidden, inter], q2[j])
                w.emit(f"{p}.down_proj.weight_scale", "F32", [hidden, 1], s2[j])
        stats[str(L)] = {"w13": st13, "w2": st2, "experts": E, "secs": round(time.time() - t0, 1)}
        print(f"[fp8] layer {L}: {E} experts exact; w13 {st13}; w2 {st2}; {time.time() - t0:.0f}s", flush=True)
    return stats


def convert_rest(src: Source, w: ShardWriter) -> dict:
    n = 0
    for name in sorted(src.weight_map):
        if _FUSED_RE.match(name):
            continue
        dtype, shape = src.meta(name)
        if dtype not in ("BF16", "F32"):
            # The FP8 config leaves every non-expert module unquantized: a quantized tensor here would load into a
            # bf16 Linear as garbage.
            raise SystemExit(f"FATAL: non-expert tensor {name} is {dtype}, expected BF16/F32")
        w.emit(name, dtype, shape, src.raw(name))
        n += 1
    print(f"[fp8] rest: {n} tensors copied byte for byte", flush=True)
    return {"tensors": n}


def expected_names(src: Source) -> tuple[set[str], set[str], dict[int, int], str]:
    pfx, fused = src.fused_layers()
    experts = {
        f"{pfx}.layers.{L}.mlp.experts.{e}.{p}.{s}"
        for L, E in fused.items()
        for e in range(E)
        for p in _PROJ
        for s in ("weight", "weight_scale")
    }
    rest = {n for n in src.weight_map if not _FUSED_RE.match(n)}
    return experts, rest, fused, pfx


def finalize(src_dir: str, dst: str) -> None:
    src = Source(src_dir)
    written: dict[str, str] = {}
    manifests = sorted(glob.glob(os.path.join(dst, "_manifest-*.json")))
    for mf in manifests:
        with open(mf) as f:
            part = json.load(f)
        dup = written.keys() & part.keys()
        if dup:
            raise SystemExit(f"FATAL: {len(dup)} tensors written twice (e.g. {sorted(dup)[0]})")
        written.update(part)
    if not os.path.exists(os.path.join(dst, "_manifest-rest.json")):
        raise SystemExit("FATAL: _manifest-rest.json missing: the --rest worker did not finish")

    # Completeness: exactly the expected names, no more, no fewer. A missing worker would give a checkpoint that
    # loads with experts silently absent.
    experts, rest, fused, pfx = expected_names(src)
    want = experts | rest
    got = set(written)
    if got != want:
        miss, extra = sorted(want - got), sorted(got - want)
        raise SystemExit(f"FATAL: {len(miss)} missing (e.g. {miss[:3]}), {len(extra)} unexpected (e.g. {extra[:3]})")

    files = set(written.values())
    on_disk = {os.path.basename(p) for p in glob.glob(os.path.join(dst, "*.safetensors"))}
    if files - on_disk:
        raise SystemExit(f"FATAL: manifests reference missing files {sorted(files - on_disk)[:3]}")
    if on_disk - files:
        raise SystemExit(
            f"FATAL: stale shards not in any manifest: {sorted(on_disk - files)[:5]} "
            "(delete them, then re-run finalize)"
        )

    # Header gate: dtype and shape of every expert tensor; the rest identical to the source.
    E0 = next(iter(fused.values()))
    _, (_, R13, C13) = src.meta(f"{pfx}.layers.{min(fused)}.mlp.experts.routed_experts.w13_weight")
    inter, hidden = R13 // 2, 2 * C13
    shapes = {"gate_proj": [inter, hidden], "up_proj": [inter, hidden], "down_proj": [hidden, inter]}
    for fn in sorted(files):
        sh = Shard(os.path.join(dst, fn))
        for name in sh.header:
            dt, shape = sh.meta(name)
            if name in experts:
                proj, kind = name.rsplit(".", 2)[-2:]
                want_dt, want_shape = ("F8_E4M3", shapes[proj]) if kind == "weight" else ("F32", [shapes[proj][0], 1])
                if (dt, shape) != (want_dt, want_shape):
                    raise SystemExit(f"FATAL: {name} is {dt}{shape}, want {want_dt}{want_shape}")
            elif (dt, shape) != src.meta(name):
                raise SystemExit(f"FATAL: {name} is {dt}{shape}, source {src.meta(name)}")
    n_exp, n_other = check_targets(written)

    total = sum(os.path.getsize(os.path.join(dst, f)) for f in files)
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(written.items()))}, f, indent=1)

    with open(os.path.join(src_dir, "config.json")) as f:
        cfg = json.load(f)
    cfg["quantization_config"] = build_quant_config()
    tc = cfg.get("text_config")
    if isinstance(tc, dict) and tc.get("quantization_config"):
        raise SystemExit("FATAL: source text_config carries a quantization_config")
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Tokenizer (tiktoken.model, tokenization_kimi.py), chat template, processor and remote-code files: everything
    # that is not a weight shard or regenerated here.
    skip = {"config.json", "model.safetensors.index.json"}
    copied = []
    for fn in sorted(os.listdir(src_dir)):
        sp = os.path.join(src_dir, fn)
        if fn in skip or fn.endswith(".safetensors") or fn.startswith("_") or not os.path.isfile(sp):
            continue
        shutil.copy2(sp, os.path.join(dst, fn))
        copied.append(fn)

    stats = {}
    for p in sorted(glob.glob(os.path.join(dst, "_stats-*.json"))):
        with open(p) as f:
            stats[os.path.basename(p)[len("_stats-") : -len(".json")]] = json.load(f)
    with open(os.path.join(dst, "_fp8_repack_stats.json"), "w") as f:
        json.dump(
            {"source": os.path.abspath(src_dir), "experts_per_layer": E0, "layers": sorted(fused), "workers": stats},
            f,
            indent=1,
        )

    print(f"[fp8] aux copied: {copied}")
    print(
        f"[fp8] FINALIZED -> {dst}: {total / 2**30:.1f} GiB, {len(written)} tensors "
        f"({n_exp} expert weights targeted, {n_other} other weights unquantized), "
        f"{len(fused)} expert layers, {len(files)} shards",
        flush=True,
    )


# ---------------------------------------------------------------- self-test
def self_test(src_dir: str | None, layers: list[int], min_exp: int) -> None:
    import torch

    codec = TorchCodec(min_exp)
    lut = build_lut(min_exp)
    rng = random.Random(12345)

    def check(rows, label):
        cols = len(rows[0][0])
        w = torch.tensor([list(pack_row(n)) for n, _ in rows], dtype=torch.uint8).unsqueeze(0)
        s = torch.tensor([sc for _, sc in rows], dtype=torch.uint8).unsqueeze(0)
        q, scale, st = codec.encode(w, s)
        for r, (n, sc) in enumerate(rows):
            ref, k = reference_encode_row(n, sc, min_exp, lut)
            if q[0, r].tolist() != ref or float(scale[0, r, 0]) != math.ldexp(1.0, k):
                raise SystemExit(f"FATAL self-test {label}: torch != reference on row {r}")
        print(f"[self-test] {label}: {len(rows)} rows x {cols} torch == reference; {st}", flush=True)

    for cols in (7168, 2048):
        check(
            [synthetic_row(rng, cols, rng.randint(0, 9)) for _ in range(48)]
            + [adversarial_row(cols, 9), ([0] * cols, [127] * (cols // BLOCK)), ([8] * cols, [3] * (cols // BLOCK))],
            f"synthetic cols={cols}",
        )
        for spread in (10, 11):
            try:
                n, sc = adversarial_row(cols, spread)
                codec.encode(
                    torch.tensor([[list(pack_row(n))]], dtype=torch.uint8), torch.tensor([[sc]], dtype=torch.uint8)
                )
            except InexactError:
                continue
            if spread > (FP8_EXP_MAX - min_exp) - 3:
                raise SystemExit(f"FATAL self-test: spread {spread} row passed at min_exp {min_exp}")
    if src_dir:
        src = Source(src_dir)
        pfx, fused = src.fused_layers()
        for L in layers[:2] or [min(fused)]:
            base = f"{pfx}.layers.{L}.mlp.experts.routed_experts"
            for wn in ("w13", "w2"):
                E, R, C = src.meta(f"{base}.{wn}_weight")[1]
                nb = src.meta(f"{base}.{wn}_weight_scale")[1][2]
                for e in (0, E - 1):
                    rows = []
                    for r0 in (0, R - 32):
                        wb = src.expert_rows(f"{base}.{wn}_weight", e, r0, r0 + 32)
                        sb = src.expert_rows(f"{base}.{wn}_weight_scale", e, r0, r0 + 32)
                        rows += [
                            (unpack_row(wb[i * C : (i + 1) * C]), list(sb[i * nb : (i + 1) * nb])) for i in range(32)
                        ]
                    check(rows, f"real layer {L} {wn} expert {e}")
    print("SELF_TEST_OK", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="MXFP4 checkpoint written by repack_quark_mxfp4.py")
    ap.add_argument("--dst", help="output dir for the FP8 checkpoint")
    ap.add_argument("--layers", help="expert layers for this worker, e.g. 1-6")
    ap.add_argument("--rest", action="store_true", help="copy the non-expert tensors")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument(
        "--min-exp",
        type=int,
        default=DEFAULT_MIN_EXP,
        help="lowest e4m3 exponent allowed (-5: the Gaudi 2 x0.5 load stays in normals; "
        "-6 needs VLLM_SCALE_ADJUSTMENT=false)",
    )
    ap.add_argument("--expert-chunk", type=int, default=4, help="experts per torch batch")
    ap.add_argument("--shard-size-gb", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if not (FP8_EXP_MIN_NORMAL <= args.min_exp <= 0):
        ap.error("--min-exp must be in [-6, 0]")

    if args.self_test:
        if args.threads:
            import torch

            torch.set_num_threads(args.threads)
        self_test(args.src, parse_layers(args.layers) if args.layers else [], args.min_exp)
        return
    if not args.dst:
        ap.error("--dst is required")
    os.makedirs(args.dst, exist_ok=True)
    if args.finalize:
        finalize(args.src, args.dst)
        return

    src = Source(args.src)
    if args.rest:
        w = ShardWriter(args.dst, "rest", args.shard_size_gb)
        w.clean_stale()
        stats = convert_rest(src, w)
    else:
        if not args.layers:
            ap.error("pass --layers, --rest, --finalize or --self-test")
        import torch

        if args.threads:
            torch.set_num_threads(args.threads)
        layers = parse_layers(args.layers)
        w = ShardWriter(args.dst, f"L{layers[0]:02d}-{layers[-1]:02d}", args.shard_size_gb)
        w.clean_stale()
        stats = convert_layers(src, w, layers, TorchCodec(args.min_exp), args.expert_chunk)
    w.flush()
    w.save_manifest(stats)
    print(f"[fp8] worker {w.tag} done: {len(w.written)} tensors", flush=True)


if __name__ == "__main__":
    main()
