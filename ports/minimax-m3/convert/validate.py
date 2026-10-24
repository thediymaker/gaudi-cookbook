#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent check of a repack.py output against its MXFP4 source.

Reads DST back from disk and compares by VALUE with its own reader and decoders
(nothing is imported from repack.py or fp8_codec.py), so a bug shared with the converter's
LUT cannot pass. Any failure -> exit 1.

  inventory  config.json is the source config with only quantization_config
             replaced by the expected compressed-tensors config; the FP8 target
             hits every routed expert and nothing else under vLLM's matching rules
             (checkpoint names and the vLLM-side names M3 builds: the RoutedExperts
             ".0.<proj>" probe, fused qkv_proj/gate_up_proj shards, the VL mapper's
             vision_tower.* prefixes); the index holds exactly the expected names
             (6 per expert, 1 BF16 weight per shared-expert projection, every other
             source tensor); shard headers, total_size, no unreferenced shards, aux
             files byte-identical.
  rest       every passthrough tensor is byte-identical to the source (sha256), and
             every shared-expert BF16 weight equals the MXFP4 decode element by
             element.
  scales     every expert weight_scale (all experts of --layers, all rows) is a
             finite power of two 2^k with k <= smax_row - 132: the row tops out in
             e4m3's 2^7 binade, so a larger k would decode above the source row's
             ceiling of 6 * 2^(smax-127).
  sample     N random experts per layer (+ the first and last), element by element:
             e4m3fn(dst) * scale == e2m1(src) * 2^(s-127) in float64; no 0x80, no
             exponent-15 code; every nonzero row tops out in the 2^7 binade and
             bottoms at >= 2^--min-exp; torch's own float8_e4m3fn decode agrees.
  full       `sample` over every expert of --layers.

  validate.py --src SRC --dst DST --part inventory
  validate.py --src SRC --dst DST --part rest
  validate.py --src SRC --dst DST --part scales --layers 3-8
  validate.py --src SRC --dst DST --part sample --layers 3-8 --experts-per-layer 8
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
EXPERT = re.compile(
    r"^(?P<pfx>.+)\.layers\.(?P<L>\d+)\.block_sparse_moe\.experts\.(?P<E>\d+)\."
    r"(?P<w>w[123])\.(?P<k>weight|weight_scale)$"
)
SHARED = re.compile(
    r"^(?P<b>.+\.block_sparse_moe\.shared_experts\.(gate_proj|up_proj|down_proj))\."
    r"(?P<k>weight|weight_scale)$"
)
WS = ("w1", "w3", "w2")
# vLLM's packed_modules_mapping for M3 (HpuMiniMaxM3SparseForCausalLM) and the
# vision tower's fused qkv.
FUSED = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"]}


def fail(msg: str) -> None:
    print(f"VALIDATE_FAIL {msg}", flush=True)
    sys.exit(1)


class Raw:
    """Own minimal safetensors reader (header + mmap)."""

    def __init__(self, root: str):
        self.root = root
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            idx = json.load(f)
        self.weight_map: dict[str, str] = idx["weight_map"]
        self.total_size = idx.get("metadata", {}).get("total_size")
        self._h: dict[str, tuple[dict, int, mmap.mmap]] = {}

    def _file(self, fn: str):
        if fn not in self._h:
            f = open(os.path.join(self.root, fn), "rb")  # noqa: SIM115 (kept open for the mmap)
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

    def raw(self, name: str) -> bytes:
        hdr, base, mm = self._file(self.weight_map[name])
        a, b = hdr[name]["data_offsets"]
        return mm[base + a : base + b]

    def u8(self, name: str) -> torch.Tensor:
        dt, shape = self.meta(name)
        if dt not in ("U8", "F8_E4M3"):
            fail(f"{name} is {dt}, want U8/F8_E4M3")
        return torch.frombuffer(bytearray(self.raw(name)), dtype=torch.uint8).reshape(shape)

    def f32(self, name: str) -> torch.Tensor:
        dt, shape = self.meta(name)
        if dt != "F32":
            fail(f"{name} is {dt}, want F32")
        return torch.frombuffer(bytearray(self.raw(name)), dtype=torch.float32).reshape(shape)

    def bf16_as_f64(self, name: str) -> torch.Tensor:
        """BF16 bytes -> float64 by bit shift into an fp32 pattern (no torch bf16 path)."""
        dt, shape = self.meta(name)
        if dt != "BF16":
            fail(f"{name} is {dt}, want BF16")
        h = torch.frombuffer(bytearray(self.raw(name)), dtype=torch.int16).to(torch.int32) & 0xFFFF
        return (h << 16).view(torch.float32).to(torch.float64).reshape(shape)


def fp4_decode(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """MXFP4 rows (U8 [R, C/2] low nibble first, E8M0 [R, C/32]) -> float64 [R, C]."""
    if bool((s == 255).any()):
        fail("E8M0 NaN scale in the source")
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


def _read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def re_match(name: str, target: str) -> bool:
    return re.match(target[3:], name) is not None if target.startswith("re:") else name == target


def ignored(name: str, ignore: list[str]) -> bool:
    """vLLM should_ignore_layer: a fused module is ignored iff all its shards are,
    and mixed verdicts are an error."""
    proj = name.split(".")[-1]
    if proj in FUSED and name not in ignore:
        verdicts = {any(re_match(name.replace(proj, s), t) for t in ignore) for s in FUSED[proj]}
        if len(verdicts) != 1:
            fail(f"fused module {name}: shards disagree on ignore")
        return verdicts.pop()
    return any(re_match(name, t) for t in ignore)


def targeted(name: str, targets: list[str]) -> bool:
    """vLLM find_matched_target for regex targets: direct match, else all shards."""
    if any(re_match(name, t) for t in targets):
        return True
    proj = name.split(".")[-1]
    if proj in FUSED:
        return all(any(re_match(name.replace(proj, s), t) for t in targets) for s in FUSED[proj])
    return False


def source_layout(src: Raw):
    pfx, lay = set(), {}
    for n in src.weight_map:
        m = EXPERT.match(n)
        if m:
            pfx.add(m.group("pfx"))
            lay.setdefault(int(m.group("L")), set()).add(int(m.group("E")))
    if len(pfx) != 1:
        fail(f"source expert prefix not unique: {sorted(pfx)}")
    for L, es in lay.items():
        if es != set(range(len(es))):
            fail(f"source layer {L}: expert ids are not contiguous")
    return pfx.pop(), {L: len(es) for L, es in lay.items()}


def shared_bases(src: Raw) -> list[str]:
    out = set()
    for n in src.weight_map:
        m = SHARED.match(n)
        if m and src.meta(m.group("b") + ".weight")[0] == "U8":
            out.add(m.group("b"))
    return sorted(out)


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
        fail("fp8_dtype_flavor missing: Gaudi2 would apply scale_adjustment in the config process")
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
        fail("text_config carries a quantization_config")
    if {k: v for k, v in cfg.items() if k != "quantization_config"} != {
        k: v for k, v in scfg.items() if k != "quantization_config"
    }:
        fail("config.json differs from the source beyond quantization_config")

    pfx, lay = source_layout(src)
    n_routed = tc.get("num_local_experts") if isinstance(tc, dict) else None
    if n_routed is not None and set(lay.values()) != {n_routed}:
        fail(f"experts per layer {set(lay.values())} != num_local_experts {n_routed}")
    freq = tc.get("moe_layer_freq") if isinstance(tc, dict) else None
    if freq is not None and sorted(lay) != [i for i, f in enumerate(freq) if f]:
        fail(f"expert layers {sorted(lay)} != moe_layer_freq")

    want: dict[str, tuple[str, list[int]]] = {}
    for L, E in lay.items():
        for e in range(E):
            for wn in WS:
                b = f"{pfx}.layers.{L}.block_sparse_moe.experts.{e}.{wn}"
                dt, (rows, half) = src.meta(b + ".weight")
                want[b + ".weight"] = ("F8_E4M3", [rows, 2 * half])
                want[b + ".weight_scale"] = ("F32", [rows, 1])
    sb = shared_bases(src)
    for b in sb:
        rows, half = src.meta(b + ".weight")[1]
        want[b + ".weight"] = ("BF16", [rows, 2 * half])
    skip = set(want) | {b + ".weight_scale" for b in sb}
    rest = [n for n in src.weight_map if n not in skip]
    for n in rest:
        dt, shp = src.meta(n)
        if dt not in ("BF16", "F32"):
            fail(f"source passthrough tensor {n} is {dt}")
        want[n] = (dt, shp)
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

    # Target / ignore on checkpoint names ...
    n_t = 0
    for n in dst.weight_map:
        if not n.endswith(".weight"):
            continue
        mod = n[:-7]
        if EXPERT.match(n):
            if not targeted(mod, targets) or any(re_match(mod, t) for t in ignore):
                fail(f"expert module {mod} not targeted or ignored")
            n_t += 1
        elif targeted(mod, targets):
            fail(f"non-expert module {mod} matches the FP8 target")
    # ... and on the vLLM-side names the CT config is consulted with.
    for L in (min(lay), max(lay)):
        for p in ("gate_proj", "up_proj", "down_proj"):
            probe = f"{pfx}.layers.{L}.block_sparse_moe.experts.0.{p}"
            if not targeted(probe, targets) or ignored(probe, ignore):
                fail(f"RoutedExperts probe {probe} misses the target or is ignored")
    L0 = min(lay)
    for mod in (
        f"{pfx}.layers.{L0}.self_attn.qkv_proj",
        f"{pfx}.layers.{L0}.self_attn.o_proj",
        f"{pfx}.layers.{L0}.block_sparse_moe.shared_experts.gate_up_proj",
        f"{pfx}.layers.{L0}.block_sparse_moe.shared_experts.down_proj",
        f"{pfx}.layers.{L0}.block_sparse_moe.gate",
        f"{pfx}.layers.0.mlp.gate_up_proj",
        f"{pfx}.layers.0.mlp.down_proj",
        "language_model.lm_head",
        "vision_tower.vision_model.encoder.layers.0.self_attn.qkv_proj",
        "vision_tower.vision_model.encoder.layers.0.self_attn.out_proj",
        "vision_tower.vision_model.encoder.layers.0.fc1",
        "vision_tower.vision_model.encoder.layers.0.fc2",
        "vision_tower.multi_modal_projector.linear_1",
        "vision_tower.patch_merge_mlp.linear_2",
    ):
        if targeted(mod, targets):
            fail(f"vLLM module {mod} matches the FP8 target")
        if not ignored(mod, ignore):
            fail(f"vLLM module {mod} is not in ignore (belt and braces)")

    for fn in sorted(os.listdir(src_dir)):
        sp = os.path.join(src_dir, fn)
        if (
            fn in ("config.json", "model.safetensors.index.json")
            or fn.endswith(".safetensors")
            or fn.startswith((".", "_"))
            or not os.path.isfile(sp)
        ):
            continue
        dp = os.path.join(dst_dir, fn)
        if not os.path.isfile(dp) or _read(sp) != _read(dp):
            fail(f"aux file {fn} missing or different")
    print(
        f"[validate] inventory OK: {len(want)} tensors ({n_t} FP8 expert weights, {len(sb)} shared BF16, "
        f"{len(rest)} passthrough), {len(lay)} expert layers, {len(files)} shards, {total / 2**30:.1f} GiB",
        flush=True,
    )


def part_rest(src: Raw, dst: Raw) -> None:
    sb = set(shared_bases(src))
    n = ns = 0
    for name in sorted(src.weight_map):
        if EXPERT.match(name):
            continue
        m = SHARED.match(name)
        if m and m.group("b") in sb:
            if m.group("k") == "weight":
                ref = fp4_decode(src.u8(name), src.u8(m.group("b") + ".weight_scale"))
                val = dst.bf16_as_f64(name)
                if not torch.equal(val, ref):
                    bad = (val != ref).nonzero()[0].tolist()
                    fail(f"{name}: BF16 {float(val[tuple(bad)])} != MXFP4 {float(ref[tuple(bad)])} at {bad}")
                ns += 1
            continue
        if src.meta(name) != dst.meta(name):
            fail(f"{name}: dtype/shape changed")
        if hashlib.sha256(src.raw(name)).digest() != hashlib.sha256(dst.raw(name)).digest():
            fail(f"{name}: bytes differ from the source")
        n += 1
    print(
        f"[validate] rest OK: {n} tensors byte-identical, {ns} shared-expert BF16 weights equal the MXFP4 decode",
        flush=True,
    )


def part_scales(src: Raw, dst: Raw, layers: list[int]) -> None:
    pfx, lay = source_layout(src)
    rows = top = 0
    for L in layers:
        if L not in lay:
            fail(f"layer {L} has no experts in the source")
        for e in range(lay[L]):
            for wn in WS:
                b = f"{pfx}.layers.{L}.block_sparse_moe.experts.{e}.{wn}"
                smax = src.u8(b + ".weight_scale").amax(-1).long()
                s = dst.f32(b + ".weight_scale").flatten()
                if s.numel() != smax.numel():
                    fail(f"{b}: {s.numel()} scale rows vs {smax.numel()} source rows")
                if not bool(torch.isfinite(s).all()) or not bool((s > 0).all()):
                    fail(f"{b}: non-finite or non-positive scale")
                mant, ex = torch.frexp(s.double())
                if not bool((mant == 0.5).all()):
                    fail(f"{b}: scale is not a power of two")
                k = ex.long() - 1
                if bool((k > smax - 132).any()):
                    r = int((k > smax - 132).nonzero()[0])
                    fail(f"{b} row {r}: scale 2^{int(k[r])} > 2^(smax-132) (smax {int(smax[r])})")
                rows += k.numel()
                top += int((k == smax - 132).sum())
        print(f"[validate] scales layer {L}: {lay[L]} experts OK", flush=True)
    print(
        f"[validate] scales OK: {rows} rows, {100.0 * top / max(rows, 1):.3f}% at k = smax-132 "
        f"(the rest have a top block without a 4/6)",
        flush=True,
    )


def check_expert(src: Raw, dst: Raw, pfx: str, L: int, e: int, min_exp: int) -> None:
    for wn in WS:
        b = f"{pfx}.layers.{L}.block_sparse_moe.experts.{e}.{wn}"
        ref = fp4_decode(src.u8(b + ".weight"), src.u8(b + ".weight_scale"))
        q = dst.u8(b + ".weight")
        if dst.meta(b + ".weight")[0] != "F8_E4M3":
            fail(f"{b}.weight is not F8_E4M3")
        sc = dst.f32(b + ".weight_scale").double()
        val = e4m3fn_decode(q) * sc
        if not torch.equal(val, ref):
            bad = (val != ref).nonzero()[0].tolist()
            fail(f"{b} value mismatch at {bad}: {float(val[tuple(bad)])} vs {float(ref[tuple(bad)])}")
        qa = q & 0x7F
        if bool((q == 0x80).any()) or bool(((qa >> 3) == 0xF).any()):
            fail(f"{b}: emits 0x80 or an exponent-15 (Gaudi2 non-finite) code")
        nz = qa != 0
        ex = (qa >> 3).long() - 7
        row_nz = nz.any(1)
        hi = torch.where(nz, ex, torch.full_like(ex, -99)).amax(1)
        lo = torch.where(nz, ex, torch.full_like(ex, 99)).amin(1)
        if bool((hi[row_nz] != 7).any()):
            fail(f"{b}: a nonzero row does not reach the 2^7 binade")
        if bool((lo[row_nz] < min_exp).any()):
            fail(f"{b}: exponent below 2^{min_exp} (Gaudi2's x0.5 load would emit subnormals)")
        if bool((sc[~row_nz] != 1.0).any()):
            fail(f"{b}: an all-zero row has a scale other than 1")
        if hasattr(torch, "float8_e4m3fn"):
            tv = q.view(torch.float8_e4m3fn).to(torch.float64) * sc
            if not torch.equal(tv, val):
                fail(f"{b}: torch float8_e4m3fn decode disagrees with the formula decode")


def part_sample(src: Raw, dst: Raw, layers: list[int], n: int | None, seed: int, min_exp: int) -> None:
    pfx, lay = source_layout(src)
    rng = random.Random(seed)
    done = 0
    for L in layers:
        E = lay[L]
        experts = list(range(E)) if n is None else sorted({0, E - 1, *rng.sample(range(E), min(n, E))})
        for e in experts:
            check_expert(src, dst, pfx, L, e, min_exp)
            done += 1
        print(f"[validate] layer {L}: experts {experts if n is not None else f'0..{E - 1}'} exact", flush=True)
    print(f"[validate] {'full' if n is None else 'sample'} OK: {done} experts element-exact", flush=True)


def parse_layers(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--part", required=True, choices=["inventory", "rest", "scales", "sample", "full"])
    ap.add_argument("--layers", default="3-59")
    ap.add_argument("--experts-per-layer", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--min-exp", type=int, default=-5)
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
        n = None if args.part == "full" else args.experts_per_layer
        part_sample(src, dst, parse_layers(args.layers), n, args.seed, args.min_exp)
    print(
        f"VALIDATE_OK part={args.part} layers={args.layers if args.part not in ('inventory', 'rest') else '-'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
