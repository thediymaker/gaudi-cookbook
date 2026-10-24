#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent check of mxfp4_to_fp8.py output against its MXFP4 source.

Reads DST back from disk and compares by value with its own reader and decoders (nothing is imported from the
converter), so a bug shared with the converter's LUT cannot pass. Any failure exits 1.

  inventory  config.json parses as the expected compressed-tensors config; the FP8 target hits every routed expert
             and nothing else (vLLM match rules); the index holds exactly the expected names (6 per expert plus
             every non-expert source tensor); every shard header has the right dtype and shape; total_size matches;
             no unreferenced shards; aux files identical.
  rest       every non-expert tensor is byte-identical to the source (sha256).
  scales     every expert weight_scale (all layers in --layers, all experts, all rows) is a finite power of two 2^k
             with k <= smax_row - 132: the row tops out in e4m3's 2^7 binade, so a larger k would decode above the
             source row's ceiling of 6 * 2^(smax-127).
  sample     N random experts per layer (plus the first and last), element by element:
             e4m3fn(dst) * scale == e2m1(src) * 2^(s-127) in float64; no 0x80, no exponent-15 code; every nonzero
             row tops out in the 2^7 binade and bottoms at >= 2^--min-exp; torch's own float8_e4m3fn decode agrees.
             With --amd-src the same experts are also matched against amd/Kimi-K2.7-Code-MXFP4's per-expert
             gate/up/down tensors, which pins the gate/up split independently of the fused w13.
  full       `sample` over every expert of --layers (reads the whole range).

The parts are independent; scales, sample and full can be split by layer range across parallel processes.

Usage:
  validate_fp8.py --src SRC --dst DST --part inventory
  validate_fp8.py --src SRC --dst DST --part rest
  validate_fp8.py --src SRC --dst DST --part scales --layers 1-6
  validate_fp8.py --src SRC --dst DST --part sample --layers 1-6 --experts-per-layer 8 [--amd-src AMD_SNAPSHOT]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import random
import re
import struct
import sys

import torch

E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float64
)
FUSED = re.compile(r"^(?P<pfx>.+)\.layers\.(?P<L>\d+)\.mlp\.experts\.routed_experts\.(?P<w>\w+)$")
PROJ = ("gate_proj", "up_proj", "down_proj")


def fail(msg: str) -> None:
    print(f"VALIDATE_FAIL {msg}", flush=True)
    sys.exit(1)


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


class Raw:
    """Minimal safetensors reader of its own (header + mmap)."""

    def __init__(self, root: str):
        self.root = root
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            idx = json.load(f)
        self.weight_map: dict[str, str] = idx["weight_map"]
        self.total_size = idx.get("metadata", {}).get("total_size")
        self._h: dict[str, tuple[dict, int, mmap.mmap]] = {}

    def _file(self, fn: str):
        if fn not in self._h:
            f = open(os.path.join(self.root, fn), "rb")  # noqa: SIM115 (kept open for the mmap's lifetime)
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            hdr.pop("__metadata__", None)
            self._h[fn] = (hdr, 8 + n, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
        return self._h[fn]

    def header(self, fn: str) -> dict:
        return self._file(fn)[0]

    def meta(self, name: str) -> tuple[str, list[int]]:
        m = self._file(self.weight_map[name])[0][name]
        return m["dtype"], m["shape"]

    def raw(self, name: str, start: int = 0, length: int | None = None) -> bytes:
        hdr, base, mm = self._file(self.weight_map[name])
        a, b = hdr[name]["data_offsets"]
        length = b - a - start if length is None else length
        return mm[base + a + start : base + a + start + length]

    def u8(self, name: str, start: int = 0, length: int | None = None, shape=None) -> torch.Tensor:
        t = torch.frombuffer(bytearray(self.raw(name, start, length)), dtype=torch.uint8)
        return t.reshape(shape) if shape is not None else t

    def f32(self, name: str) -> torch.Tensor:
        dt, shape = self.meta(name)
        if dt != "F32":
            fail(f"{name} is {dt}, want F32")
        return torch.frombuffer(bytearray(self.raw(name)), dtype=torch.float32).reshape(shape)


def fp4_decode(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """MXFP4 rows (U8 [R, C/2] low nibble first, E8M0 [R, C/32]) -> float64 [R, C]."""
    v = E2M1[torch.stack([(w & 0x0F).long(), (w >> 4).long()], -1).reshape(w.shape[0], -1)]
    return v * torch.pow(2.0, s.double() - 127.0).repeat_interleave(32, 1)


def e4m3fn_decode(q: torch.Tensor) -> torch.Tensor:
    """e4m3fn bytes -> float64 by formula (bias 7, subnormals at exponent 0)."""
    q = q.long()
    sign = torch.where(q >= 128, -1.0, 1.0).double()
    e = (q >> 3) & 0xF
    m = (q & 7).double()
    normal = (1.0 + m / 8.0) * torch.pow(2.0, e.double() - 7.0)
    sub = (m / 8.0) * 2.0**-6
    return sign * torch.where(e == 0, sub, normal)


def source_layout(src: Raw):
    pfx, layers = set(), {}
    for n in src.weight_map:
        m = FUSED.match(n)
        if m and m.group("w") == "w13_weight":
            pfx.add(m.group("pfx"))
            E, R13, C13 = src.meta(n)[1]
            layers[int(m.group("L"))] = (E, R13 // 2, 2 * C13)  # experts, inter, hidden
    if len(pfx) != 1:
        fail(f"source expert prefix not unique: {sorted(pfx)}")
    return pfx.pop(), layers


def re_match(name: str, target: str) -> bool:
    return re.match(target[3:], name) is not None if target.startswith("re:") else name == target


# -------------------------------------------------------------------- parts
def part_inventory(src: Raw, dst: Raw, src_dir: str, dst_dir: str) -> None:
    with open(os.path.join(dst_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(src_dir, "config.json")) as f:
        scfg = json.load(f)
    qc = cfg.get("quantization_config") or {}
    if qc.get("quant_method") != "compressed-tensors" or qc.get("format") != "float-quantized":
        fail(f"quantization_config is not compressed-tensors/float-quantized: {qc}")
    if qc.get("fp8_dtype_flavor") != "float8_e4m3fnuz":
        fail("fp8_dtype_flavor missing: Gaudi 2 would apply scale_adjustment in the config process")
    groups = list(qc.get("config_groups", {}).values())
    if len(groups) != 1:
        fail(f"expected one config group, got {len(groups)}")
    g = groups[0]
    w, a = g.get("weights", {}), g.get("input_activations", {})
    if (w.get("num_bits"), w.get("type"), w.get("strategy"), w.get("dynamic"), w.get("symmetric")) != (
        8,
        "float",
        "channel",
        False,
        True,
    ):
        fail(f"weights args {w}")
    if (a.get("num_bits"), a.get("type"), a.get("strategy"), a.get("dynamic")) != (8, "float", "token", True):
        fail(f"input_activations args {a}")
    targets, ignore = g.get("targets", []), qc.get("ignore", [])
    tc = cfg.get("text_config", {})
    if isinstance(tc, dict) and tc.get("quantization_config"):
        fail("text_config still carries a quantization_config")
    strip = {k: v for k, v in cfg.items() if k != "quantization_config"}
    sstrip = {k: v for k, v in scfg.items() if k != "quantization_config"}
    if strip != sstrip:
        fail("config.json differs from the source beyond quantization_config")

    pfx, layers = source_layout(src)
    n_routed = tc.get("n_routed_experts") if isinstance(tc, dict) else None
    if n_routed is not None and {E for E, _, _ in layers.values()} != {n_routed}:
        fail(f"experts per layer {set(E for E, _, _ in layers.values())} != n_routed_experts {n_routed}")
    want: dict[str, tuple[str, list[int]] | None] = {}
    for L, (E, inter, hidden) in layers.items():
        shp = {"gate_proj": [inter, hidden], "up_proj": [inter, hidden], "down_proj": [hidden, inter]}
        for e in range(E):
            for p in PROJ:
                base = f"{pfx}.layers.{L}.mlp.experts.{e}.{p}"
                want[f"{base}.weight"] = ("F8_E4M3", shp[p])
                want[f"{base}.weight_scale"] = ("F32", [shp[p][0], 1])
    rest = [n for n in src.weight_map if not FUSED.match(n)]
    for n in rest:
        want[n] = src.meta(n)
    got = set(dst.weight_map)
    if got != set(want):
        miss, extra = sorted(set(want) - got), sorted(got - set(want))
        fail(f"index: {len(miss)} missing (e.g. {miss[:3]}), {len(extra)} unexpected (e.g. {extra[:3]})")

    files = sorted(set(dst.weight_map.values()))
    on_disk = sorted(f for f in os.listdir(dst_dir) if f.endswith(".safetensors"))
    if files != on_disk:
        fail(f"shards on disk != shards in index: {sorted(set(on_disk) ^ set(files))[:5]}")
    total = 0
    for fn in files:
        total += os.path.getsize(os.path.join(dst_dir, fn))
        for name, m in dst.header(fn).items():
            if dst.weight_map.get(name) != fn:
                fail(f"{name} in {fn} but index says {dst.weight_map.get(name)}")
            if (m["dtype"], m["shape"]) != tuple(want[name]):
                fail(f"{name}: {m['dtype']}{m['shape']} != {want[name]}")
    if dst.total_size != total:
        fail(f"index total_size {dst.total_size} != {total}")

    # Target and ignore under vLLM's rules, on checkpoint names and on the vLLM-side names KimiK25 builds (fused
    # shards, RoutedExperts' ".0.<proj>" probe, the mapper).
    n_t = 0
    for n in dst.weight_map:
        if not n.endswith(".weight"):
            continue
        mod = n[:-7]
        hit = any(re_match(mod, t) for t in targets)
        ign = any(re_match(mod, t) for t in ignore)
        if re.search(r"\.mlp\.experts\.\d+\.", n):
            if not hit or ign:
                fail(f"expert module {mod} not targeted or ignored")
            n_t += 1
        elif hit:
            fail(f"non-expert module {mod} matches the FP8 target")
    for L in sorted(layers)[:1] + sorted(layers)[-1:]:
        for p in PROJ:
            probe = f"{pfx}.layers.{L}.mlp.experts.0.{p}"
            if not any(re_match(probe, t) for t in targets):
                fail(f"RoutedExperts probe {probe} misses the target")
    for mod in (
        "mm_projector.linear_1",
        "mm_projector.linear_2",
        "language_model.lm_head",
        f"{pfx}.layers.1.self_attn.q_a_proj",
        f"{pfx}.layers.1.mlp.shared_experts.gate_proj",
        f"{pfx}.layers.0.mlp.gate_proj",
        f"{pfx}.layers.1.mlp.gate",
    ):
        if any(re_match(mod, t) for t in targets):
            fail(f"vLLM module {mod} matches the FP8 target")

    skip = {"config.json", "model.safetensors.index.json"}
    for fn in sorted(os.listdir(src_dir)):
        sp = os.path.join(src_dir, fn)
        if fn in skip or fn.endswith(".safetensors") or fn.startswith("_") or not os.path.isfile(sp):
            continue
        dp = os.path.join(dst_dir, fn)
        if not os.path.isfile(dp) or read_bytes(sp) != read_bytes(dp):
            fail(f"aux file {fn} missing or different")
    print(
        f"[validate] inventory OK: {len(want)} tensors ({n_t} FP8 expert weights, {len(rest)} "
        f"passthrough), {len(layers)} expert layers, {len(files)} shards, {total / 2**30:.1f} GiB",
        flush=True,
    )


def part_rest(src: Raw, dst: Raw) -> None:
    n = 0
    for name in sorted(src.weight_map):
        if FUSED.match(name):
            continue
        if src.meta(name) != dst.meta(name):
            fail(f"{name}: dtype/shape changed")
        if hashlib.sha256(src.raw(name)).digest() != hashlib.sha256(dst.raw(name)).digest():
            fail(f"{name}: bytes differ from the source")
        n += 1
    print(f"[validate] rest OK: {n} tensors byte-identical", flush=True)


def part_scales(src: Raw, dst: Raw, layers: list[int]) -> None:
    pfx, lay = source_layout(src)
    rows = top = 0
    for L in layers:
        E, inter, hidden = lay[L]
        base = f"{pfx}.layers.{L}.mlp.experts.routed_experts"
        s13_all = src.u8(f"{base}.w13_weight_scale", shape=src.meta(f"{base}.w13_weight_scale")[1])
        s2_all = src.u8(f"{base}.w2_weight_scale", shape=src.meta(f"{base}.w2_weight_scale")[1])
        smax13 = s13_all.amax(-1).long()  # [E, 2I]
        smax2 = s2_all.amax(-1).long()  # [E, H]
        for e in range(E):
            p = f"{pfx}.layers.{L}.mlp.experts.{e}"
            sc = torch.cat([dst.f32(f"{p}.gate_proj.weight_scale"), dst.f32(f"{p}.up_proj.weight_scale")]).flatten()
            for s, smax, what in (
                (sc, smax13[e], "w13"),
                (dst.f32(f"{p}.down_proj.weight_scale").flatten(), smax2[e], "w2"),
            ):
                if not bool(torch.isfinite(s).all()) or not bool((s > 0).all()):
                    fail(f"{p} {what}: non-finite or non-positive scale")
                mant, ex = torch.frexp(s.double())
                if not bool((mant == 0.5).all()):
                    fail(f"{p} {what}: scale is not a power of two")
                k = ex.long() - 1
                if bool((k > smax - 132).any()):
                    r = int((k > smax - 132).nonzero()[0])
                    fail(
                        f"{p} {what} row {r}: scale 2^{int(k[r])} > 2^(smax-132) "
                        f"(smax {int(smax[r])}): inconsistent with the source row"
                    )
                rows += k.numel()
                top += int((k == smax - 132).sum())
        print(f"[validate] scales layer {L}: {E} experts OK", flush=True)
    print(
        f"[validate] scales OK: {rows} rows, {100.0 * top / max(rows, 1):.3f}% at k = smax-132 "
        f"(the rest have a top block without a 4/6)",
        flush=True,
    )


def check_expert(src: Raw, dst: Raw, pfx: str, L: int, e: int, dims, min_exp: int, amd: Raw | None):
    E, inter, hidden = dims
    base = f"{pfx}.layers.{L}.mlp.experts.routed_experts"
    w13 = src.u8(f"{base}.w13_weight", e * 2 * inter * hidden // 2, 2 * inter * hidden // 2, (2 * inter, hidden // 2))
    s13 = src.u8(
        f"{base}.w13_weight_scale", e * 2 * inter * hidden // 32, 2 * inter * hidden // 32, (2 * inter, hidden // 32)
    )
    w2 = src.u8(f"{base}.w2_weight", e * hidden * inter // 2, hidden * inter // 2, (hidden, inter // 2))
    s2 = src.u8(f"{base}.w2_weight_scale", e * hidden * inter // 32, hidden * inter // 32, (hidden, inter // 32))
    ref13, ref2 = fp4_decode(w13, s13), fp4_decode(w2, s2)
    refs = {"gate_proj": ref13[:inter], "up_proj": ref13[inter:], "down_proj": ref2}
    p = f"{pfx}.layers.{L}.mlp.experts.{e}"
    for proj in PROJ:
        dt, shape = dst.meta(f"{p}.{proj}.weight")
        q = dst.u8(f"{p}.{proj}.weight", shape=shape)
        sc = dst.f32(f"{p}.{proj}.weight_scale").double()
        val = e4m3fn_decode(q) * sc
        if not torch.equal(val, refs[proj]):
            bad = (val != refs[proj]).nonzero()[0].tolist()
            fail(f"{p}.{proj} value mismatch at {bad}: {float(val[tuple(bad)])} vs {float(refs[proj][tuple(bad)])}")
        qa = q & 0x7F
        if bool((q == 0x80).any()) or bool(((qa >> 3) == 0xF).any()):
            fail(f"{p}.{proj}: emits 0x80 or an exponent-15 code (not finite on Gaudi 2)")
        nz = qa != 0
        ex = (qa >> 3).long() - 7
        row_nz = nz.any(1)
        top = torch.where(nz, ex, torch.full_like(ex, -99)).amax(1)
        low = torch.where(nz, ex, torch.full_like(ex, 99)).amin(1)
        if bool((top[row_nz] != 7).any()):
            fail(f"{p}.{proj}: a nonzero row does not reach the 2^7 binade")
        if bool((low[row_nz] < min_exp).any()):
            fail(f"{p}.{proj}: exponent below 2^{min_exp} (Gaudi 2's x0.5 load would emit subnormals)")
        if hasattr(torch, "float8_e4m3fn"):
            tv = q.view(torch.float8_e4m3fn).to(torch.float64) * sc
            if not torch.equal(tv, val):
                fail(f"{p}.{proj}: torch float8_e4m3fn decode disagrees with the formula decode")
        if amd is not None:
            an = f"{pfx}.layers.{L}.mlp.experts.{e}.{proj}"
            aw = amd.u8(f"{an}.weight", shape=amd.meta(f"{an}.weight")[1])
            asc = amd.u8(f"{an}.weight_scale", shape=amd.meta(f"{an}.weight_scale")[1])
            if not torch.equal(fp4_decode(aw, asc), val):
                fail(f"{p}.{proj} != AMD per-expert {an} (gate/up split or order wrong)")


def part_sample(src: Raw, dst: Raw, layers: list[int], n: int | None, seed: int, min_exp: int, amd: Raw | None) -> None:
    pfx, lay = source_layout(src)
    rng = random.Random(seed)
    done = 0
    for L in layers:
        E = lay[L][0]
        experts = list(range(E)) if n is None else sorted({0, E - 1, *rng.sample(range(E), min(n, E))})
        for e in experts:
            check_expert(src, dst, pfx, L, e, lay[L], min_exp, amd)
            done += 1
        print(
            f"[validate] layer {L}: experts {experts if n is not None else f'0..{E - 1}'} exact"
            f"{' + AMD match' if amd else ''}",
            flush=True,
        )
    print(f"[validate] {'full' if n is None else 'sample'} OK: {done} experts element-exact", flush=True)


def parse_layers(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="MXFP4 checkpoint the FP8 one was converted from")
    ap.add_argument("--dst", required=True, help="FP8 checkpoint to check")
    ap.add_argument("--part", required=True, choices=["inventory", "rest", "scales", "sample", "full"])
    ap.add_argument("--layers", default="1-60")
    ap.add_argument("--experts-per-layer", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--min-exp", type=int, default=-5)
    ap.add_argument("--amd-src", help="amd/Kimi-K2.7-Code-MXFP4 snapshot for the gate/up cross-check")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    src, dst = Raw(args.src), Raw(args.dst)
    if args.part == "inventory":
        part_inventory(src, dst, args.src, args.dst)
    elif args.part == "rest":
        part_rest(src, dst)
    elif args.part == "scales":
        part_scales(src, dst, parse_layers(args.layers))
    else:
        amd = Raw(args.amd_src) if args.amd_src else None
        n = None if args.part == "full" else args.experts_per_layer
        part_sample(src, dst, parse_layers(args.layers), n, args.seed, args.min_exp, amd)
    print(
        f"VALIDATE_OK part={args.part} layers={args.layers if args.part not in ('inventory', 'rest') else '-'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
