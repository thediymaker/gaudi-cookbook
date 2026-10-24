#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent check of convert_to_hpu_mxfp4.py output against the source checkpoint.

Reads DST back from disk and compares it to SRC by VALUE, with its own decoders, so a bug shared with the converter's
byte transforms cannot pass:

  experts  every expert's w13 (split per --w13-layout) and w2 decode to exactly the source w1/w3/w2 values; scale
           bytes are equal; the output weight bytes contain no 0x8 nibble (-0 canonicalised).
  dense    every bf16 tensor equals e4m3fn(src) * 2^(scale - 127) elementwise; passthrough tensors are
           byte-identical.
  engram   each bucket's bytes hash-equal the source row range.

Prints VALIDATE_OK, or VALIDATE_FAILED and exits 1 on any mismatch.

  validate_conversion.py --src SRC --dst DST --part all
  validate_conversion.py --src SRC --dst DST --part experts --layers 0-4
  validate_conversion.py --src SRC --dst DST --part dense
  validate_conversion.py --src SRC --dst DST --part engram --layer 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_to_hpu_mxfp4 import (  # noqa: E402  (raw IO only)
    Shard,
    Source,
    _text_config,
    expert_count,
    mtp_count,
    parse_layers,
)

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def fp4_values(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    lo, hi = (w & 0x0F).long(), (w >> 4).long()
    v = E2M1[torch.stack([lo, hi], -1).reshape(w.shape[0], -1)]
    return v * torch.pow(2.0, s.float() - 127.0).repeat_interleave(32, 1)


def e4m3fn_values(w: torch.Tensor) -> torch.Tensor:
    """Independent e4m3fn decode (bias 7, no inf, 0x7F/0xFF NaN)."""
    w = w.long()
    sign = torch.where(w >= 128, -1.0, 1.0)
    e = (w >> 3) & 0xF
    m = (w & 7).float()
    normal = (1.0 + m / 8.0) * torch.pow(2.0, e.float() - 7.0)
    sub = (m / 8.0) * 2.0**-6
    return sign * torch.where(e == 0, sub, normal)


class Dst:
    def __init__(self, root: str):
        self.root = root
        self.shards: dict[str, Shard] = {}
        self.where: dict[str, str] = {}
        for fn in sorted(os.listdir(root)):
            if fn.endswith(".safetensors"):
                sh = Shard(os.path.join(root, fn))
                self.shards[fn] = sh
                for n in sh.header:
                    self.where[n] = fn

    def u8(self, name):
        sh = self.shards[self.where[name]]
        dt, shape = sh.meta(name)
        assert dt == "U8", (name, dt)
        return torch.frombuffer(bytearray(sh.raw(name)), dtype=torch.uint8).reshape(shape)

    def tensor(self, name):
        sh = self.shards[self.where[name]]
        dt, shape = sh.meta(name)
        td = {"BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8}[dt]
        return torch.frombuffer(bytearray(sh.raw(name)), dtype=td).reshape(shape)


def check_experts(src: Source, dst: Dst, prefix: str, layout: str) -> int:
    key = f"{prefix}.ffn.experts.routed_experts"
    w13, s13 = dst.u8(f"{key}.w13_weight"), dst.u8(f"{key}.w13_weight_scale")
    w2, s2 = dst.u8(f"{key}.w2_weight"), dst.u8(f"{key}.w2_weight_scale")
    bad = 0
    for e in range(w13.shape[0]):
        base = f"{prefix}.ffn.experts.{e}"
        if layout == "interleaved":
            g, u, gs, us = w13[e, 0::2], w13[e, 1::2], s13[e, 0::2], s13[e, 1::2]
        else:
            h = w13.shape[1] // 2
            g, u, gs, us = w13[e, :h], w13[e, h:], s13[e, :h], s13[e, h:]
        for got_w, got_s, wn in ((g, gs, "w1"), (u, us, "w3"), (w2[e], s2[e], "w2")):
            sw, ss = src.bytes_u8(f"{base}.{wn}.weight"), src.bytes_u8(f"{base}.{wn}.scale")
            ok = (
                torch.equal(got_s, ss)
                and torch.equal(fp4_values(got_w, got_s), fp4_values(sw, ss))
                and not bool(((got_w & 0x0F) == 8).any() or ((got_w >> 4) == 8).any())
            )
            if not ok:
                bad += 1
                print(f"MISMATCH {base}.{wn}")
    n_src = expert_count(src, prefix)
    if w13.shape[0] != n_src:
        bad += 1
        print(f"MISMATCH {prefix}: {w13.shape[0]} experts in dst, {n_src} in src")
    print(f"{prefix}: {w13.shape[0]} experts checked, {bad} mismatches", flush=True)
    return bad


def check_dense(src: Source, dst: Dst) -> int:
    bad = n = 0
    for name in sorted(dst.where):
        if ".routed_experts." in name or ".engram.embed.head" in name:
            continue
        dt, _ = src.meta(name)
        got = dst.tensor(name)
        if dt == "F8_E4M3":
            w, s = src.bytes_u8(name), src.bytes_u8(name[: -len(".weight")] + ".scale")
            N, K = w.shape
            scale = torch.pow(2.0, s.float() - 127.0).repeat_interleave(32, 0)[:N].repeat_interleave(32, 1)[:, :K]
            want = e4m3fn_values(w) * scale
            ok = got.dtype == torch.bfloat16 and torch.equal(got.float(), want)
        else:
            ok = bytes(dst.shards[dst.where[name]].raw(name)) == bytes(src.shard(name).raw(name))
        n += 1
        if not ok:
            bad += 1
            print(f"MISMATCH {name}")
    print(f"dense: {n} tensors checked, {bad} mismatches", flush=True)
    return bad


def check_engram(src: Source, dst: Dst, layer: int) -> int:
    with open(os.path.join(dst.root, "manifests", f"engram-l{layer}.json")) as f:
        m = json.load(f)["tensors"]
    bad = 0
    for name, e in sorted(m.items()):
        sname, row0, rows = e["src"]
        sh = src.shard(sname)
        _, shape = sh.meta(sname)
        rb = shape[1]
        a, _ = sh.header[sname]["data_offsets"]
        hs, hd = hashlib.sha256(), hashlib.sha256()
        base = sh._base + a
        step = 1 << 20
        for r in range(row0, row0 + rows, step):
            k = min(step, row0 + rows - r)
            hs.update(sh._mm[base + r * rb : base + (r + k) * rb])
        dsh = dst.shards[dst.where[name]]
        da, db = dsh.header[name]["data_offsets"]
        for off in range(da, db, step * rb):
            hd.update(dsh._mm[dsh._base + off : dsh._base + min(off + step * rb, db)])
        ok = hs.hexdigest() == hd.hexdigest() == e["sha256"]
        if not ok:
            bad += 1
            print(f"MISMATCH {name}")
    print(f"engram layer {layer}: {len(m)} buckets checked, {bad} mismatches", flush=True)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--part", required=True, choices=["all", "experts", "mtp", "dense", "engram"])
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--w13-layout", default="interleaved", choices=["interleaved", "concat"])
    args = ap.parse_args()
    src, dst = Source(args.src), Dst(args.dst)
    if args.part == "all":
        prefixes = [f"layers.{i}" for i in parse_layers(args.layers)] + [f"mtp.{i}" for i in range(mtp_count(src))]
        bad = sum(check_experts(src, dst, p, args.w13_layout) for p in prefixes)
        bad += check_dense(src, dst)
        bad += sum(check_engram(src, dst, layer) for layer in _text_config(args.src)["engram_layer_ids"])
    elif args.part == "experts":
        bad = sum(check_experts(src, dst, f"layers.{i}", args.w13_layout) for i in parse_layers(args.layers))
    elif args.part == "mtp":
        bad = sum(check_experts(src, dst, f"mtp.{i}", args.w13_layout) for i in range(mtp_count(src)))
    elif args.part == "dense":
        bad = check_dense(src, dst)
    else:
        bad = check_engram(src, dst, args.layer)
    print("VALIDATE_OK" if bad == 0 else f"VALIDATE_FAILED ({bad})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
