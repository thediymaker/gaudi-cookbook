# SPDX-License-Identifier: Apache-2.0
"""MXFP4 to per-channel FP8 re-encoding, exact, plus a raw safetensors reader and writer.

An MXFP4 element is e2m1 (values 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives) times 2^(s-127), where s is
the E8M0 byte shared by a block of 32 elements. Per output channel (a weight row) we pick one power-of-two
scale 2^k with k = (exponent of the row's largest element) - 7, so the row's top element lands in e4m3's top
binade. Every element of the row is then an e4m3 normal with exponent in [-5, 7] as long as the row's nonzero
elements span at most 12 binades, and the value e4m3(q) * 2^k equals the MXFP4 value exactly. Rows outside
that window raise InexactError; nothing is ever rounded.

Gaudi 2 specifics the encoder respects:
- Gaudi 2 reads e4m3fn bits with a maximum of 240: exponent field 15 is not finite there. The top binade we
  emit is 2^7 (values up to 192 for e2m1 inputs), so exponent 15 never appears.
- The loader can halve FP8 weights (scale adjustment). The lowest exponent is -5, not e4m3's -6, so a halved
  copy is still a normal. The recipe also turns scale adjustment off.
- -0 is written as +0 (0x00), never 0x80.

The numerics are pure Python so the tests run without torch. TorchCodec is the vectorised twin used for the
real conversion; it re-decodes every chunk in float64 and raises if anything differs.
"""

from __future__ import annotations

import glob
import json
import math
import mmap
import os
import random
import struct
from fractions import Fraction

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
E2M1_EXP = (None, -1, 0, 0, 1, 1, 2, 2, None, -1, 0, 0, 1, 1, 2, 2)  # floor(log2|v|), None for zero
BLOCK = 32  # MXFP4 elements per E8M0 scale
E8M0_BIAS = 127
E8M0_NAN = 255
FP8_EXP_MAX = 7  # top binade Gaudi 2 can represent (max 240)
FP8_EXP_MIN_NORMAL = -6
DEFAULT_MIN_EXP = -5  # one binade of headroom for the x0.5 load
D_MIN, D_MAX = -40, 40  # LUT range of the per-block shift d = s - 127 - k
ND = D_MAX - D_MIN + 1
K_LO, K_HI = -126, 127  # the row scale 2^k must be an fp32 normal
POW2_LO, POW2_HI = -200, 200
NO_EXP = -10_000


class InexactError(RuntimeError):
    """A value would not survive the re-encoding bit-exactly."""


def floor_log2(a: Fraction) -> int:
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    return e


def e4m3fn_decode(b: int) -> Fraction | None:
    """e4m3fn byte to its exact value (bias 7, subnormals at exponent 0, 0x7F/0xFF NaN as None)."""
    sign = -1 if b & 0x80 else 1
    e, m = (b >> 3) & 0xF, b & 0x7
    if e == 0xF and m == 0x7:
        return None
    if e == 0:
        return sign * Fraction(m, 8) * Fraction(1, 64)
    return sign * (1 + Fraction(m, 8)) * Fraction(2) ** (e - 7)


def e4m3fn_encode_normal(v: Fraction, min_exp: int = FP8_EXP_MIN_NORMAL) -> int | None:
    """Exact e4m3fn byte for v as a normal with exponent in [min_exp, 7], else None. 0 encodes to 0x00."""
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
    """(byte, ok) tables indexed nibble * ND + (d - D_MIN): the e4m3fn encoding of E2M1[nibble] * 2^d."""
    out, ok = bytearray(16 * ND), bytearray(16 * ND)
    for n in range(16):
        for d in range(D_MIN, D_MAX + 1):
            b = e4m3fn_encode_normal(Fraction(E2M1[n]) * Fraction(2) ** d, min_exp)
            if b is not None:
                i = n * ND + (d - D_MIN)
                out[i], ok[i] = b, 1
    return bytes(out), bytes(ok)


FP8_DECODE = tuple(float("nan") if (x := e4m3fn_decode(b)) is None else float(x) for b in range(256))


def reference_encode_row(nibbles, scales, min_exp=DEFAULT_MIN_EXP, lut=None):
    """Pure-Python reference for one row: (e4m3fn bytes, k). TorchCodec.encode must agree byte for byte."""
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
                    f"block {b} elem {j}: {E2M1[n]} * 2^{s - E8M0_BIAS} is not an e4m3 normal in "
                    f"[2^{min_exp}, 2^{FP8_EXP_MAX}] after the row scale 2^{k}"
                )
            out.append(lut_b[i])
    return out, k


def unpack_row(packed: bytes) -> list[int]:
    """Packed U8 row to nibbles, element 0 in the low nibble."""
    out = []
    for byte in packed:
        out.append(byte & 0x0F)
        out.append(byte >> 4)
    return out


def pack_row(nibbles: list[int]) -> bytes:
    return bytes(nibbles[i] | (nibbles[i + 1] << 4) for i in range(0, len(nibbles), 2))


def synthetic_row(rng: random.Random, cols: int, spread: int, base: int | None = None):
    """Random row with an exact E8M0 spread, including all-zero and -0 blocks."""
    nb = cols // BLOCK
    base = rng.randint(100, 135) if base is None else base
    scales = [base + rng.randint(0, spread) for _ in range(nb)]
    scales[rng.randrange(nb)] = base + spread
    scales[rng.randrange(nb)] = base
    nib = [rng.randrange(16) for _ in range(cols)]
    for _ in range(max(1, nb // 16)):
        b = rng.randrange(nb)
        nib[b * BLOCK : (b + 1) * BLOCK] = [rng.choice((0, 8))] * BLOCK
    return nib, scales


def adversarial_row(cols: int, spread: int, base: int = 120):
    """6.0 at scale base+spread and -0.5 at base: the widest row a spread allows (spread + 3 binades)."""
    nb = cols // BLOCK
    scales = [base] * nb
    nib = [0] * cols
    scales[0] = base + spread
    nib[0:BLOCK] = [7] * BLOCK
    nib[BLOCK : 2 * BLOCK] = [9] * BLOCK
    return nib, scales


def parse_layers(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return sorted(out)


def ct_match(name: str, target: str) -> bool:
    """compressed-tensors target matching as vLLM does it: 're:' regex from the start, else equality."""
    if target.startswith("re:"):
        import re

        return re.match(target[3:], name) is not None
    return name == target


# --------------------------------------------------------------------------------------------- safetensors IO
class Shard:
    """Read-only raw view of one safetensors file (no torch, no dtype conversion)."""

    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "rb")  # noqa: SIM115 (kept open for the mmap)
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


ITEMSIZE = {"F8_E4M3": 1, "U8": 1, "BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4}


class ShardWriter:
    """Buffers tensors as bytes and writes model-<tag>-NNNNN.safetensors (tmp, fsync, rename) once the buffer
    passes the size limit. The name to file map is saved last, as the worker's completion marker."""

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
        if len(data) != math.prod(shape) * ITEMSIZE[dtype]:
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
        with open(os.path.join(self.dst, f"_stats-{self.tag}.json"), "w") as f:
            json.dump(stats, f, indent=1)
        with open(os.path.join(self.dst, f"_manifest-{self.tag}.json"), "w") as f:
            json.dump(self.written, f)


# ---------------------------------------------------------------------------------------------------- torch
class TorchCodec:
    """Vectorised twin of reference_encode_row (torch on the CPU)."""

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
        """w uint8 [n, rows, cols/2], s uint8 [n, rows, cols/32] to
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
        top = self.e2m1_exp[nib].amax(-1) + sl - E8M0_BIAS
        emax = top.amax(-1)
        zero = emax < NO_EXP // 2
        k = torch.where(zero, torch.zeros_like(emax), emax - FP8_EXP_MAX)
        if bool(((k < K_LO) | (k > K_HI)).any()):
            raise InexactError(f"row scale exponent outside [{K_LO}, {K_HI}]: {int(k.min())}..{int(k.max())}")
        di = (sl - E8M0_BIAS - k.unsqueeze(-1) - D_MIN).clamp(0, ND - 1)
        idx = nib * ND + di.unsqueeze(-1)
        ok = self.lut_ok[idx]
        if not bool(ok.all()):
            e, r, b, j = (~ok).nonzero()[0].tolist()
            raise InexactError(
                f"{int((~ok).sum())} elements not exact; first: expert+{e} row {r} block {b} elem {j} "
                f"nibble {int(nib[e, r, b, j])} scale {int(sl[e, r, b])} row-k {int(k[e, r])}"
            )
        q = self.lut_bytes[idx]
        # Independent re-decode in float64 (-0 == +0); any difference stops the conversion.
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
        }
        if stats["q_exp_min"] is not None and stats["q_exp_min"] < self.min_exp:
            raise InexactError(f"emitted exponent {stats['q_exp_min']} below {self.min_exp}")
        if bool(((q == 0x80) | ((qa >> 3) == 0xF)).any()):
            raise InexactError("emitted 0x80 or an exponent-15 code")
        return q, scale, stats
