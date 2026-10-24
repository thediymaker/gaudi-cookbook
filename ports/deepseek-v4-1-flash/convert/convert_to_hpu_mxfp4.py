#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash -> the checkpoint this port serves: a lossless re-layout, not a quantization.

Source: a public DeepSeek-V4.1-Flash checkpoint (FP8 32x32 blocks with E8M0 scales, MXFP4 routed experts), for
example deepseek-ai/DeepSeek-V4.1-Flash-1017 (the one the recipe serves) or deepseek-ai/DeepSeek-V4.1-Flash. Both
use the same tensor names and formats, and this script converts either.

DeepSeek's routed experts are already OCP MXFP4 (E2M1 packed low nibble first, one E8M0 scale per 32 values), so they
are copied, not requantized. Per tensor class:

  routed experts  -> P.ffn.experts.routed_experts.{w13_weight,w13_weight_scale,w2_weight,w2_weight_scale}, all U8.
                     w13 rows are gate (w1) and up (w3) INTERLEAVED, the layout the clamp-capable
                     bias_mxfp4_fused_weights op and the port's kernels read. Nibble 0x8 (-0) becomes 0x0 (+0).
  FP8 32x32 E8M0  -> bf16, exactly (an e4m3 value times 2^k fits bf16 without rounding). Gaudi 2's FP8 E4M3 tops out
                     at 240, so e4m3fn values above that do not survive a device cast; no FP8 byte is cast there.
  bf16 / f32      -> byte passthrough.
  engram tables   -> 24 U8 bucket tensors per layer (one per prime hash column), bytes unchanged, so a
                     tensor-parallel rank reads only its own heads.

Tensors are read as raw bytes from the safetensors files. Nothing goes through a torch dtype conversion of I8 or
F8_E8M0: a copy_ of int8 into uint8 clamps where a byte view wraps, which would silently zero expert bytes.

Each part writes its shards plus a manifest (manifests/<part>.json, sha256 of every tensor), so parts can run in
parallel or on different machines; finalize merges them, checks that every source tensor went somewhere exactly once,
and writes the index and config.json. --part all runs everything in order.

  convert_to_hpu_mxfp4.py --src SRC --dst DST --part all
  convert_to_hpu_mxfp4.py --src SRC --dst DST --part experts --layers 0-4
  convert_to_hpu_mxfp4.py --src SRC --dst DST --part mtp
  convert_to_hpu_mxfp4.py --src SRC --dst DST --part dense
  convert_to_hpu_mxfp4.py --src SRC --dst DST --part engram --layer 1
  convert_to_hpu_mxfp4.py --src SRC --dst DST --part finalize --source deepseek-ai/DeepSeek-V4.1-Flash-1017@<rev>

validate_conversion.py checks the output against the source by value, with its own decoders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import shutil
import struct
import sys

import torch

H = 5120  # hidden_size
INTER = 2304  # moe_intermediate_size
BLOCK = 32  # MXFP4 group, and the FP8 weight block edge
FP8_E8M0_ONE = 127  # E8M0 byte for 2^0

# One-byte dtypes whose raw bytes are kept as U8.
_BYTE_DTYPES = {"I8", "U8", "F8_E4M3", "F8_E8M0"}
_PASSTHROUGH = {"BF16", "F32"}


# --------------------------------------------------------------------------- IO
class Shard:
    """Read-only raw view of one safetensors file."""

    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "rb")  # noqa: SIM115  (kept open for the mmap)
        n = struct.unpack("<Q", self._f.read(8))[0]
        self.header = json.loads(self._f.read(n))
        self.header.pop("__metadata__", None)
        self._base = 8 + n
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

    def meta(self, name: str) -> tuple[str, list[int]]:
        m = self.header[name]
        return m["dtype"], m["shape"]

    def raw(self, name: str) -> bytes:
        a, b = self.header[name]["data_offsets"]
        return self._mm[self._base + a : self._base + b]


class Source:
    def __init__(self, root: str):
        self.root = root
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self._shards: dict[str, Shard] = {}

    def shard(self, name: str) -> Shard:
        fn = self.weight_map[name]
        if fn not in self._shards:
            self._shards[fn] = Shard(os.path.join(self.root, fn))
        return self._shards[fn]

    def meta(self, name: str) -> tuple[str, list[int]]:
        return self.shard(name).meta(name)

    def bytes_u8(self, name: str) -> torch.Tensor:
        """A one-byte tensor as uint8 with its checkpoint shape (a copy, never a cast)."""
        dtype, shape = self.meta(name)
        assert dtype in _BYTE_DTYPES, f"{name}: {dtype} is not a one-byte dtype"
        buf = bytearray(self.shard(name).raw(name))
        return torch.frombuffer(buf, dtype=torch.uint8).reshape(shape)

    def tensor(self, name: str) -> torch.Tensor:
        """A bf16/f32 tensor, by bytes."""
        dtype, shape = self.meta(name)
        td = {"BF16": torch.bfloat16, "F32": torch.float32}[dtype]
        buf = bytearray(self.shard(name).raw(name))
        return torch.frombuffer(buf, dtype=td).reshape(shape)


_ST_DTYPE = {torch.uint8: "U8", torch.bfloat16: "BF16", torch.float32: "F32"}


def write_safetensors(path: str, tensors: dict[str, torch.Tensor]) -> dict[str, str]:
    """Write tensors (U8/BF16/F32 only) and return {name: sha256 of its bytes}."""
    header, blobs, off, digests = {}, [], 0, {}
    for name in sorted(tensors):
        t = tensors[name].contiguous()
        assert t.dtype in _ST_DTYPE, f"{name}: output dtype {t.dtype} not allowed"
        b = t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.uint8 else t.numpy().tobytes()
        header[name] = {"dtype": _ST_DTYPE[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
        digests[name] = hashlib.sha256(b).hexdigest()
        blobs.append(b)
        off += len(b)
    header["__metadata__"] = {"format": "pt"}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return digests


# -------------------------------------------------------------------- transforms
def canon_neg_zero(w: torch.Tensor) -> torch.Tensor:
    """Rewrite E2M1 nibble 0x8 (-0.0) as 0x0 (+0.0). Value-preserving."""
    lo = w & 0x0F
    hi = w >> 4
    lo = torch.where(lo == 0x8, torch.zeros_like(lo), lo)
    hi = torch.where(hi == 0x8, torch.zeros_like(hi), hi)
    return lo | (hi << 4)


def fuse_w13(gate: torch.Tensor, up: torch.Tensor, layout: str) -> torch.Tensor:
    """[rows, k] gate/up -> [2*rows, k]. Scales use the same fusion as weights."""
    if layout == "interleaved":
        return torch.stack([gate, up], dim=1).reshape(2 * gate.shape[0], *gate.shape[1:])
    if layout == "concat":
        return torch.cat([gate, up], dim=0)
    raise ValueError(layout)


def expert_count(src: Source, prefix: str) -> int:
    n = 0
    while f"{prefix}.ffn.experts.{n}.w1.weight" in src.weight_map:
        n += 1
    return n


def repack_experts(
    src: Source, prefix: str, layout: str = "interleaved", experts: list[int] | None = None, canon: bool = True
) -> dict[str, torch.Tensor]:
    """Routed experts of one MoE layer ('layers.3' or 'mtp.0') -> fused U8 tensors."""
    ids = experts if experts is not None else list(range(expert_count(src, prefix)))
    w13, s13, w2, s2 = [], [], [], []
    for e in ids:
        base = f"{prefix}.ffn.experts.{e}"
        g, u, d = (src.bytes_u8(f"{base}.{w}.weight") for w in ("w1", "w3", "w2"))
        gs, us, ds = (src.bytes_u8(f"{base}.{w}.scale") for w in ("w1", "w3", "w2"))
        assert g.shape == (INTER, H // 2) and u.shape == g.shape and d.shape == (H, INTER // 2), (
            base,
            g.shape,
            d.shape,
        )
        assert gs.shape == (INTER, H // BLOCK) and ds.shape == (H, INTER // BLOCK), (base, gs.shape, ds.shape)
        for s in (gs, us, ds):
            # E8M0 0xFF is NaN; 0 would be 2^-127. Neither appears in a sane checkpoint.
            assert int(s.max()) < 0xFF and int(s.min()) > 0, f"{base}: scale byte out of range"
        if canon:
            g, u, d = canon_neg_zero(g), canon_neg_zero(u), canon_neg_zero(d)
        w13.append(fuse_w13(g, u, layout))
        s13.append(fuse_w13(gs, us, layout))
        w2.append(d)
        s2.append(ds)
    out = f"{prefix}.ffn.experts.routed_experts"
    return {
        f"{out}.w13_weight": torch.stack(w13),
        f"{out}.w13_weight_scale": torch.stack(s13),
        f"{out}.w2_weight": torch.stack(w2),
        f"{out}.w2_weight_scale": torch.stack(s2),
    }


def _e4m3fn_lut() -> torch.Tensor:
    lut = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
    return lut


E4M3FN_LUT = _e4m3fn_lut()


def fp8_block_to_bf16(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """F8_E4M3 bytes [N, K] with E8M0 bytes [ceil(N/32), ceil(K/32)] -> bf16, exactly."""
    N, K = w.shape
    assert s.shape == ((N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK), (w.shape, s.shape)
    assert not bool(((w & 0x7F) == 0x7F).any()), "NaN e4m3fn code in weight"
    assert int(s.max()) < 0xFF, "NaN E8M0 scale"
    vals = E4M3FN_LUT[w.long()]
    scale = torch.pow(2.0, s.to(torch.float32) - FP8_E8M0_ONE)
    scale = scale.repeat_interleave(BLOCK, 0)[:N].repeat_interleave(BLOCK, 1)[:, :K]
    out = vals * scale
    back = out.to(torch.bfloat16)
    assert torch.equal(back.to(torch.float32), out), "fp8 -> bf16 was not exact"
    return back


# -------------------------------------------------------------------------- parts
def parse_layers(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def part_experts(src: Source, dst: str, prefixes: list[str], layout: str) -> None:
    if not prefixes:  # e.g. a checkpoint without MTP layers
        return
    manifest = {}
    for p in prefixes:
        tensors = repack_experts(src, p, layout)
        fn = f"experts-{p.replace('.', '-')}.safetensors"
        digests = write_safetensors(os.path.join(dst, fn), tensors)
        manifest.update({k: {"file": fn, "sha256": v} for k, v in digests.items()})
        print(f"{p}: {sum(t.numel() for t in tensors.values()) / 1e9:.3f} GB -> {fn}", flush=True)
    _write_manifest(dst, f"experts-{prefixes[0]}-{prefixes[-1]}", manifest, layout)


def _is_expert(name: str) -> bool:
    return ".ffn.experts." in name


def _is_engram_table(name: str) -> bool:
    return ".engram.embed." in name


def part_dense(src: Source, dst: str, shard_bytes: int = 5 << 30) -> None:
    """Everything that is not a routed expert or an engram table: FP8 -> bf16, rest by bytes."""
    names = sorted(
        n for n in src.weight_map if not _is_expert(n) and not _is_engram_table(n) and not n.endswith(".scale")
    )
    manifest, batch, size, k = {}, {}, 0, 0

    def flush():
        nonlocal batch, size, k
        if not batch:
            return
        fn = f"dense-{k:03d}.safetensors"
        digests = write_safetensors(os.path.join(dst, fn), batch)
        manifest.update({n: {"file": fn, "sha256": d} for n, d in digests.items()})
        print(f"{fn}: {len(batch)} tensors, {size / 1e9:.2f} GB", flush=True)
        batch, size, k = {}, 0, k + 1

    for n in names:
        dtype, _ = src.meta(n)
        if dtype == "F8_E4M3":
            scale = n[: -len(".weight")] + ".scale"
            t = fp8_block_to_bf16(src.bytes_u8(n), src.bytes_u8(scale))
        elif dtype in _PASSTHROUGH:
            t = src.tensor(n)
        else:
            raise SystemExit(f"unhandled dtype {dtype} for {n}")
        batch[n] = t
        size += t.numel() * t.element_size()
        if size >= shard_bytes:
            flush()
    flush()
    _write_manifest(dst, "dense", manifest, None)


# ------------------------------------------------------------------------- engram
def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 2**32, as in the reference inference/engram.py."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def engram_bucket_sizes(text_config: dict) -> dict[int, list[int]]:
    """Per engram layer, the 24 prime bucket sizes in hash-column order
    (n-gram size major, head minor), drawn exactly as the reference inference/engram.py draws them. With
    engram_num_embeddings in the config, each layer's sizes must add up to it."""
    seen: set[int] = set()
    out = {}
    for layer in text_config["engram_layer_ids"]:
        cols = []
        for _ in range(text_config["engram_max_ngram_size"] - 1):
            current = text_config["engram_vocab_size"] - 1
            for _ in range(text_config["engram_n_heads"]):
                current += 1
                while not _is_prime(current) or current in seen:
                    current += 1
                seen.add(current)
                cols.append(current)
        out[layer] = cols
    for layer, n in zip(text_config["engram_layer_ids"], text_config.get("engram_num_embeddings", []), strict=False):
        assert sum(out[layer]) == n, f"engram layer {layer}: primes sum {sum(out[layer])} != {n}"
    return out


def write_raw_safetensors(path: str, entries: list[tuple[str, str, list[int], list]]) -> dict[str, str]:
    """Stream tensors to disk. entries: (name, dtype, shape, [byte chunks...]).
    Returns {name: sha256}. Nothing larger than one chunk is held in memory."""
    header, off = {}, 0
    for name, dtype, shape, chunks in entries:
        n = sum(len(c) for c in chunks)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + n]}
        off += n
    header["__metadata__"] = {"format": "pt"}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    digests, tmp = {}, path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for name, _, _, chunks in entries:
            h = hashlib.sha256()
            for c in chunks:
                h.update(c)
                f.write(c)
            digests[name] = h.hexdigest()
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return digests


def _row_chunks(shard: Shard, name: str, row0: int, rows: int, row_bytes: int, chunk_rows: int = 1 << 20) -> list:
    a, _ = shard.header[name]["data_offsets"]
    base = shard._base + a
    out = []
    for r in range(row0, row0 + rows, chunk_rows):
        n = min(chunk_rows, row0 + rows - r)
        out.append(memoryview(shard._mm)[base + r * row_bytes : base + (r + n) * row_bytes])
    return out


def part_engram(src: Source, dst: str, layer: int, text_config: dict) -> None:
    """One engram table [N, 256] F8_E4M3 + scale [N, 8] F8_E8M0 -> 24 U8 bucket
    tensors layers.L.engram.embed.headHH.{weight,scale}. Bytes unchanged; the
    dtype becomes U8 because Gaudi2 decodes some e4m3fn codes as NaN and cannot
    allocate e8m0 at all. The HPU model decodes rows with a 256-entry LUT."""
    sizes = engram_bucket_sizes(text_config)[layer]
    wname, sname = f"layers.{layer}.engram.embed.weight", f"layers.{layer}.engram.embed.scale"
    (wd, wshape), (sd, sshape) = src.meta(wname), src.meta(sname)
    assert wd == "F8_E4M3" and sd == "F8_E8M0" and wshape[0] == sum(sizes) == sshape[0], (wd, sd, wshape, sshape)
    ws, ss = src.shard(wname), src.shard(sname)
    manifest, row0 = {}, 0
    for h, rows in enumerate(sizes):
        fn = f"engram-l{layer}-h{h:02d}.safetensors"
        pre = f"layers.{layer}.engram.embed.head{h:02d}"
        digests = write_raw_safetensors(
            os.path.join(dst, fn),
            [
                (f"{pre}.weight", "U8", [rows, wshape[1]], _row_chunks(ws, wname, row0, rows, wshape[1])),
                (f"{pre}.scale", "U8", [rows, sshape[1]], _row_chunks(ss, sname, row0, rows, sshape[1])),
            ],
        )
        manifest.update(
            {
                n: {"file": fn, "sha256": d, "src": [wname if n.endswith("weight") else sname, row0, rows]}
                for n, d in digests.items()
            }
        )
        print(f"engram layer {layer} head {h:02d}: rows [{row0}, {row0 + rows}) -> {fn}", flush=True)
        row0 += rows
    _write_manifest(dst, f"engram-l{layer}", manifest, None)


# ----------------------------------------------------------------------- finalize
_COPY_FILES = ["tokenizer.json", "tokenizer_config.json", "LICENSE", "README.md"]


def finalize(src: Source, dst: str, layout: str, source: str) -> None:
    """Merge manifests into model.safetensors.index.json, write config.json, and run
    the structural gate: every source tensor is accounted for exactly once."""
    mdir = os.path.join(dst, "manifests")
    weight_map, layouts = {}, set()
    for fn in sorted(os.listdir(mdir)):
        with open(os.path.join(mdir, fn)) as f:
            m = json.load(f)
        if m.get("w13_layout"):
            layouts.add(m["w13_layout"])
        for name, e in m["tensors"].items():
            assert name not in weight_map, f"{name} written twice"
            weight_map[name] = e["file"]
    assert layouts == {layout}, f"expert layouts in manifests: {layouts}"

    # Structural gate: map each source tensor to where it went.
    expected = set()
    for n in src.weight_map:
        if n.endswith(".scale") and not _is_engram_table(n):
            continue  # folded into bf16 (dense) or carried as w*_weight_scale (experts)
        if _is_expert(n):
            prefix = n.split(".ffn.experts.")[0]
            expected.update(
                f"{prefix}.ffn.experts.routed_experts.{w}"
                for w in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
            )
        elif _is_engram_table(n):
            continue  # checked by the engram bucket counts below
        else:
            expected.add(n)
    got = {n for n in weight_map if ".engram.embed.head" not in n}
    missing, extra = sorted(expected - got), sorted(got - expected)
    assert not missing and not extra, f"structural gate: missing {missing[:10]} extra {extra[:10]}"
    with open(os.path.join(src.root, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg["text_config"]
    for layer, sizes in engram_bucket_sizes(tc).items():
        heads = {n for n in weight_map if n.startswith(f"layers.{layer}.engram.embed.head")}
        assert len(heads) == 2 * len(sizes), f"engram layer {layer}: {len(heads)} bucket tensors"

    total = sum(os.path.getsize(os.path.join(dst, f)) for f in set(weight_map.values()))
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, f, indent=1)

    cfg["quantization_config"] = {
        "quant_method": "dsv41_hpu_mxfp4",
        "routed_experts": {
            "format": "mxfp4",
            "w13_layout": f"{layout}_gate_up",
            "scale_format": "e8m0_u8",
            "negative_zero": "canonicalized",
        },
        "dense": "bf16 (dequantized exactly from fp8 32x32 e8m0)",
        "engram": {"format": "e4m3fn_u8 rows + e8m0_u8 scale per 32", "buckets": "per hash column"},
    }
    cfg["hpu_conversion"] = {
        "source": source,
        "converter": "gaudi-cookbook ports/deepseek-v4-1-flash/convert/convert_to_hpu_mxfp4.py",
        "routed_experts": {"w13_layout": f"{layout}_gate_up"},
    }
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for fn in _COPY_FILES:
        p = os.path.join(src.root, fn)
        if os.path.exists(p):
            with open(p, "rb") as a, open(os.path.join(dst, fn), "wb") as b:
                b.write(a.read())
    for d in ("encoding", "inference"):
        sd = os.path.join(src.root, d)
        if os.path.isdir(sd):
            shutil.copytree(sd, os.path.join(dst, d), dirs_exist_ok=True)
    print(f"finalize: {len(weight_map)} tensors, {total / 1e9:.2f} GB, layout {layout}; structural gate OK")


def _write_manifest(dst: str, part: str, entries: dict, layout: str | None) -> None:
    path = os.path.join(dst, "manifests", f"{part}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"part": part, "w13_layout": layout, "tensors": entries}, f, indent=1)


def mtp_count(src: Source) -> int:
    n = 0
    while expert_count(src, f"mtp.{n}"):
        n += 1
    return n


def _text_config(root: str) -> dict:
    with open(os.path.join(root, "config.json")) as f:
        return json.load(f)["text_config"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--part", required=True, choices=["all", "experts", "mtp", "dense", "engram", "finalize"])
    ap.add_argument("--layer", type=int, help="engram layer for --part engram (1 or 14)")
    ap.add_argument("--layers", default="0-39", help="backbone MoE layers for --part experts, e.g. 0-4 or 3")
    ap.add_argument("--w13-layout", default="interleaved", choices=["interleaved", "concat"])
    ap.add_argument(
        "--source",
        default="deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170",
        help="repo@revision recorded in config.json hpu_conversion.source",
    )
    args = ap.parse_args()

    src = Source(args.src)
    os.makedirs(args.dst, exist_ok=True)
    if args.part == "all":
        part_experts(src, args.dst, [f"layers.{i}" for i in parse_layers(args.layers)], args.w13_layout)
        part_experts(src, args.dst, [f"mtp.{i}" for i in range(mtp_count(src))], args.w13_layout)
        part_dense(src, args.dst)
        tc = _text_config(args.src)
        for layer in tc["engram_layer_ids"]:
            part_engram(src, args.dst, layer, tc)
        finalize(src, args.dst, args.w13_layout, args.source)
    elif args.part == "experts":
        part_experts(src, args.dst, [f"layers.{i}" for i in parse_layers(args.layers)], args.w13_layout)
    elif args.part == "mtp":
        part_experts(src, args.dst, [f"mtp.{i}" for i in range(mtp_count(src))], args.w13_layout)
    elif args.part == "dense":
        part_dense(src, args.dst)
    elif args.part == "engram":
        part_engram(src, args.dst, args.layer, _text_config(args.src))
    else:
        finalize(src, args.dst, args.w13_layout, args.source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
