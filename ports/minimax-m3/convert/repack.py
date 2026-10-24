#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Repack amd/MiniMax-M3-MXFP4 into the per-channel FP8 checkpoint the port serves. Exact: a re-encoding.

Source: the public amd/MiniMax-M3-MXFP4 snapshot (AMD Quark, quant_method "quark", OCP MXFP4 weights in
blocks of 32 with E8M0 scales, packed low nibble first). Routed experts are stored per expert:

  <P>.layers.L.block_sparse_moe.experts.E.w1.weight        U8 [3072, 3072]   gate
  <P>.layers.L.block_sparse_moe.experts.E.w3.weight        U8 [3072, 3072]   up
  <P>.layers.L.block_sparse_moe.experts.E.w2.weight        U8 [6144, 1536]   down
  <P>.layers.L.block_sparse_moe.experts.E.w*.weight_scale  U8 [rows, cols/32]

57 MoE layers (3 to 59) x 128 experts. The shared experts are MXFP4 too; everything else is BF16 or F32.

Output, a compressed-tensors checkpoint that vllm-gaudi loads through HPUCompressedTensorsW8A8Fp8MoEMethod:

  routed experts  same names and layout; F8_E4M3 [rows, cols] weights with an F32 [rows, 1] scale that is
                  always a power of two (fp8_codec.py explains why this is exact)
  shared experts  dequantized to BF16, exact (an e2m1 value has at most 2 significant bits); scales dropped
  everything else copied byte for byte

The FP8 config sets fp8_dtype_flavor so that Gaudi 2 does not halve the weights at load in the process that
parses the config; the recipe sets VLLM_SCALE_ADJUSTMENT=false for the worker processes as well.

Every chunk is re-decoded in float64 and compared, or the worker exits 1; validate.py re-checks the written
files with its own reader and decoders. The work splits into independent workers, for example on several
machines or jobs sharing one output directory:

  repack.py --src SRC --dst DST --layers 3-8          one worker per expert-layer range
  repack.py --src SRC --dst DST --rest                shared experts and the byte-for-byte rest
  repack.py --src SRC --dst DST --finalize            after every worker: completeness gates, index, config
  repack.py --src SRC --self-test [--layers 3,59]     torch path vs the pure-Python reference

About 410 GiB of output; a worker needs torch (CPU) and about 4 GiB of memory per 8-expert chunk.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp8_codec import (  # noqa: E402
    BLOCK,
    DEFAULT_MIN_EXP,
    E2M1,
    E8M0_BIAS,
    E8M0_NAN,
    FP8_EXP_MIN_NORMAL,
    POW2_HI,
    POW2_LO,
    InexactError,
    Shard,
    ShardWriter,
    TorchCodec,
    adversarial_row,
    build_lut,
    ct_match,
    pack_row,
    parse_layers,
    reference_encode_row,
    synthetic_row,
    unpack_row,
)

# vLLM probes a RoutedExperts layer as "<prefix>.0.{gate,up,down}_proj"; the checkpoint names w1/w2/w3 are
# listed too so the gates below can check them.
EXPERT_TARGET = r"re:.*\.block_sparse_moe\.experts\.\d+\.(gate_proj|up_proj|down_proj|w1|w2|w3)$"
# The target already matches routed experts only; the ignore list keeps every other Linear unquantized even
# under vLLM's module names (fused qkv_proj and gate_up_proj are checked shard by shard).
IGNORE = [
    r"re:.*lm_head$",
    r"re:.*\.self_attn\..*",
    r"re:.*\.shared_experts\..*",
    r"re:.*\.mlp\.(gate_proj|up_proj|gate_up_proj|down_proj)$",
    r"re:.*\.block_sparse_moe\.gate$",
    r"re:^vision_tower\..*",
    r"re:^multi_modal_projector\..*",
    r"re:^patch_merge_mlp\..*",
]

EXPERT_RE = re.compile(
    r"^(?P<pfx>.+)\.layers\.(?P<L>\d+)\.block_sparse_moe\.experts\.(?P<E>\d+)\."
    r"(?P<w>w1|w2|w3)\.(?P<k>weight|weight_scale)$"
)
SHARED_RE = re.compile(
    r"^(?P<base>.+\.layers\.\d+\.block_sparse_moe\.shared_experts\.(?:gate_proj|up_proj|down_proj))"
    r"\.(?P<k>weight|weight_scale)$"
)
WEIGHTS = ("w1", "w3", "w2")


def build_quant_config() -> dict:
    """FP8 per-channel weights, FP8 dynamic per-token activations, routed experts only."""
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


def check_targets(names) -> tuple[int, int]:
    """Every routed-expert module hits the target and misses `ignore`; nothing else hits the target."""
    n_exp = n_other = 0
    for n in names:
        if not n.endswith(".weight"):
            continue
        mod = n[: -len(".weight")]
        hit = ct_match(mod, EXPERT_TARGET)
        if EXPERT_RE.match(n):
            if not hit or any(ct_match(mod, t) for t in IGNORE):
                raise SystemExit(f"FATAL: expert module {mod} not targeted (or ignored)")
            n_exp += 1
        elif hit:
            raise SystemExit(f"FATAL: non-expert module {mod} matches the FP8 target")
        else:
            n_other += 1
    return n_exp, n_other


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

    def raw(self, name: str, start: int = 0, length: int | None = None) -> bytes:
        return self.shard(name).raw(name, start, length)

    def u8(self, names: list[str]):
        """Stack same-shape U8 tensors into a uint8 torch tensor [n, *shape]."""
        import torch

        shape, bufs = None, []
        for n in names:
            dt, sh = self.meta(n)
            if dt != "U8" or (shape is not None and sh != shape):
                raise SystemExit(f"FATAL: {n} is {dt}{sh}, expected U8{shape or ''}")
            shape = sh
            bufs.append(torch.frombuffer(bytearray(self.raw(n)), dtype=torch.uint8).reshape(sh))
        return torch.stack(bufs)

    def expert_layout(self):
        """(text prefix, {layer: n_experts}, {w: (weight shape, scale shape)}); experts complete and 0..E-1."""
        pfx: set[str] = set()
        seen: dict[int, dict[int, set[str]]] = {}
        for n in self.weight_map:
            m = EXPERT_RE.match(n)
            if m:
                pfx.add(m.group("pfx"))
                seen.setdefault(int(m.group("L")), {}).setdefault(int(m.group("E")), set()).add(
                    f"{m.group('w')}.{m.group('k')}"
                )
        if len(pfx) != 1:
            raise SystemExit(f"FATAL: expected one expert prefix, got {sorted(pfx)}")
        p = pfx.pop()
        full = {f"{w}.{k}" for w in WEIGHTS for k in ("weight", "weight_scale")}
        layers, shapes = {}, {w: set() for w in WEIGHTS}
        for L, ex in seen.items():
            if sorted(ex) != list(range(len(ex))):
                raise SystemExit(f"FATAL: layer {L}: expert ids are not 0..{len(ex) - 1}")
            for e, parts in ex.items():
                if parts != full:
                    raise SystemExit(f"FATAL: layer {L} expert {e}: tensors {sorted(parts)}")
            layers[L] = len(ex)
            for w in WEIGHTS:
                base = f"{p}.layers.{L}.block_sparse_moe.experts.0.{w}"
                shapes[w].add((tuple(self.meta(base + ".weight")[1]), tuple(self.meta(base + ".weight_scale")[1])))
        out = {}
        for w, s in shapes.items():
            if len(s) != 1:
                raise SystemExit(f"FATAL: {w} shapes differ across layers: {s}")
            ((ws, ss),) = s
            if ws[1] * 2 != ss[1] * BLOCK or ws[0] != ss[0]:
                raise SystemExit(f"FATAL: {w} weight {ws} vs scale {ss}")
            out[w] = (list(ws), list(ss))
        inter, hidden = out["w1"][0][0], out["w1"][0][1] * 2
        if out["w3"][0] != out["w1"][0] or out["w2"][0] != [hidden, inter // 2]:
            raise SystemExit(f"FATAL: unexpected expert shapes {out}")
        return p, layers, out

    def shared_quantized(self) -> list[str]:
        """Base names (without .weight) of the MXFP4 shared-expert projections."""
        bases: dict[str, set[str]] = {}
        for n in self.weight_map:
            m = SHARED_RE.match(n)
            if m:
                bases.setdefault(m.group("base"), set()).add(m.group("k"))
        out = []
        for b, ks in sorted(bases.items()):
            dt = self.meta(b + ".weight")[0]
            if dt == "U8":
                if ks != {"weight", "weight_scale"} or self.meta(b + ".weight_scale")[0] != "U8":
                    raise SystemExit(f"FATAL: {b}: quantized shared expert without a U8 scale")
                out.append(b)
            elif "weight_scale" in ks:
                raise SystemExit(f"FATAL: {b}: {dt} weight with a weight_scale")
        return out


def dense_dims(shape_w: list[int]) -> tuple[int, int]:
    return shape_w[0], shape_w[1] * 2


class Bf16Codec:
    """MXFP4 to BF16, exact, with a float64 re-check."""

    def __init__(self):
        import torch

        self.torch = torch
        self.e2m1 = torch.tensor(E2M1, dtype=torch.float64)
        self.pow2 = torch.tensor([math.ldexp(1.0, e) for e in range(POW2_LO, POW2_HI + 1)], dtype=torch.float64)

    def decode64(self, w, s):
        torch = self.torch
        rows, half = w.shape
        if s.shape[0] != rows or s.shape[1] * BLOCK != 2 * half:
            raise ValueError(f"weight {tuple(w.shape)} vs scale {tuple(s.shape)}")
        if bool((s == E8M0_NAN).any()):
            raise InexactError("E8M0 NaN (0xFF) scale")
        nib = torch.stack((w & 0x0F, w >> 4), dim=-1).reshape(rows, 2 * half).long()
        return self.e2m1[nib] * self.pow2[s.long() - E8M0_BIAS - POW2_LO].repeat_interleave(BLOCK, dim=1)

    def encode(self, w, s) -> bytes:
        torch = self.torch
        v = self.decode64(w, s)
        bf = v.to(torch.float32).to(torch.bfloat16)
        if not torch.equal(bf.to(torch.float64), v):
            bad = (bf.to(torch.float64) != v).nonzero()[0].tolist()
            raise InexactError(f"bf16 not exact at {bad}: {float(v[tuple(bad)])}")
        return bf.view(torch.int16).numpy().tobytes()


def _merge(acc: dict, st: dict) -> None:
    for key, fn in (("k_min", min), ("k_max", max), ("q_exp_min", min), ("q_exp_max", max)):
        if st[key] is not None:
            acc[key] = st[key] if acc.get(key) is None else fn(acc[key], st[key])
    acc["zero_rows"] = acc.get("zero_rows", 0) + st["zero_rows"]


def convert_layers(src: Source, w: ShardWriter, layers: list[int], codec: TorchCodec, chunk: int) -> dict:
    pfx, lay, shapes = src.expert_layout()
    missing = [L for L in layers if L not in lay]
    if missing:
        raise SystemExit(f"FATAL: no routed experts for layers {missing}")
    stats: dict = {}
    for L in layers:
        t0 = time.time()
        st = {wn: {} for wn in WEIGHTS}
        for e0 in range(0, lay[L], chunk):
            e1 = min(lay[L], e0 + chunk)
            for wn in WEIGHTS:
                names = [f"{pfx}.layers.{L}.block_sparse_moe.experts.{e}.{wn}" for e in range(e0, e1)]
                q, scale, s = codec.encode(
                    src.u8([n + ".weight" for n in names]), src.u8([n + ".weight_scale" for n in names])
                )
                _merge(st[wn], s)
                rows, cols = dense_dims(shapes[wn][0])
                for j, n in enumerate(names):
                    w.emit(n + ".weight", "F8_E4M3", [rows, cols], q[j])
                    w.emit(n + ".weight_scale", "F32", [rows, 1], scale[j])
        stats[str(L)] = {"experts": lay[L], "secs": round(time.time() - t0, 1), **st}
        print(f"[fp8] layer {L}: {lay[L]} experts exact; {st}; {time.time() - t0:.0f}s", flush=True)
    return stats


def convert_rest(src: Source, w: ShardWriter) -> dict:
    shared = set(src.shared_quantized())
    bf = Bf16Codec() if shared else None
    n_pass = n_bf16 = 0
    for name in sorted(src.weight_map):
        if EXPERT_RE.match(name):
            continue
        m = SHARED_RE.match(name)
        if m and m.group("base") in shared:
            if m.group("k") == "weight":
                b = m.group("base")
                ws, ss = src.u8([b + ".weight"])[0], src.u8([b + ".weight_scale"])[0]
                w.emit(b + ".weight", "BF16", list(dense_dims(list(ws.shape))), bf.encode(ws, ss))
                n_bf16 += 1
            continue
        dtype, shape = src.meta(name)
        if dtype not in ("BF16", "F32"):
            # The FP8 config leaves every non-expert module unquantized: a quantized tensor here would load as
            # garbage into a bf16 Linear.
            raise SystemExit(f"FATAL: non-expert tensor {name} is {dtype}, expected BF16/F32")
        w.emit(name, dtype, shape, src.raw(name))
        n_pass += 1
    print(f"[fp8] rest: {n_pass} tensors copied, {n_bf16} shared-expert projections to BF16 (exact)", flush=True)
    return {"passthrough": n_pass, "shared_bf16": n_bf16}


def expected_names(src: Source):
    pfx, lay, shapes = src.expert_layout()
    experts = {}
    for L, E in lay.items():
        for e in range(E):
            for wn in WEIGHTS:
                rows, cols = dense_dims(shapes[wn][0])
                b = f"{pfx}.layers.{L}.block_sparse_moe.experts.{e}.{wn}"
                experts[b + ".weight"] = ("F8_E4M3", [rows, cols])
                experts[b + ".weight_scale"] = ("F32", [rows, 1])
    shared = {b + ".weight": ("BF16", list(dense_dims(src.meta(b + ".weight")[1]))) for b in src.shared_quantized()}
    shared_src = set(shared) | {n[: -len(".weight")] + ".weight_scale" for n in shared}
    rest = {n for n in src.weight_map if not EXPERT_RE.match(n) and n not in shared_src}
    return experts, shared, rest, lay


def aux_files(src_dir: str) -> list[str]:
    """Tokenizer, chat template, processor and remote-code files: every regular file that is not a shard."""
    skip = {"config.json", "model.safetensors.index.json"}
    return [
        fn
        for fn in sorted(os.listdir(src_dir))
        if fn not in skip
        and not fn.endswith(".safetensors")
        and not fn.startswith((".", "_"))
        and os.path.isfile(os.path.join(src_dir, fn))
    ]


def finalize(src_dir: str, dst: str) -> None:
    src = Source(src_dir)
    written: dict[str, str] = {}
    for mf in sorted(glob.glob(os.path.join(dst, "_manifest-*.json"))):
        with open(mf) as f:
            part = json.load(f)
        dup = written.keys() & part.keys()
        if dup:
            raise SystemExit(f"FATAL: {len(dup)} tensors written twice (e.g. {sorted(dup)[0]})")
        written.update(part)
    if not os.path.exists(os.path.join(dst, "_manifest-rest.json")):
        raise SystemExit("FATAL: _manifest-rest.json missing: the --rest worker did not finish")

    # Exactly the expected names: a missing worker would give a checkpoint that loads with absent experts.
    experts, shared, rest, lay = expected_names(src)
    want, got = set(experts) | set(shared) | rest, set(written)
    if got != want:
        miss, extra = sorted(want - got), sorted(got - want)
        raise SystemExit(f"FATAL: {len(miss)} missing (e.g. {miss[:3]}), {len(extra)} unexpected (e.g. {extra[:3]})")
    files = set(written.values())
    on_disk = {os.path.basename(p) for p in glob.glob(os.path.join(dst, "*.safetensors"))}
    if files - on_disk:
        raise SystemExit(f"FATAL: manifests reference missing files {sorted(files - on_disk)[:3]}")
    if on_disk - files:
        raise SystemExit(f"FATAL: stale shards not in any manifest: {sorted(on_disk - files)[:5]}")
    for fn in sorted(files):
        sh = Shard(os.path.join(dst, fn))
        for name in sh.header:
            got_m = sh.meta(name)
            want_m = experts.get(name) or shared.get(name) or tuple(src.meta(name))
            if (got_m[0], list(got_m[1])) != (want_m[0], list(want_m[1])):
                raise SystemExit(f"FATAL: {name} is {got_m}, want {want_m}")
            if written[name] != fn:
                raise SystemExit(f"FATAL: {name} in {fn} but its manifest says {written[name]}")
    n_exp, n_other = check_targets(written)

    total = sum(os.path.getsize(os.path.join(dst, f)) for f in files)
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(written.items()))}, f, indent=1)

    with open(os.path.join(src_dir, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config")
    if isinstance(tc, dict) and tc.get("quantization_config"):
        raise SystemExit("FATAL: source text_config carries a quantization_config")
    if (cfg.get("quantization_config") or {}).get("quant_method") != "quark":
        raise SystemExit("FATAL: source is not the Quark MXFP4 checkpoint")
    cfg["quantization_config"] = build_quant_config()
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    copied = aux_files(src_dir)
    for fn in copied:
        shutil.copy2(os.path.join(src_dir, fn), os.path.join(dst, fn))
    print(
        f"[fp8] FINALIZED {dst}: {total / 2**30:.1f} GiB, {len(written)} tensors ({n_exp} FP8 expert weights, "
        f"{n_other} other weights, {len(shared)} shared-expert projections BF16), {len(lay)} expert layers, "
        f"{len(files)} shards; aux copied: {copied}",
        flush=True,
    )


def self_test(src_dir: str | None, layers: list[int], min_exp: int) -> None:
    import torch

    codec, lut, rng = TorchCodec(min_exp), build_lut(min_exp), random.Random(20261004)

    def check(rows, label):
        wt = torch.tensor([list(pack_row(n)) for n, _ in rows], dtype=torch.uint8).unsqueeze(0)
        st = torch.tensor([sc for _, sc in rows], dtype=torch.uint8).unsqueeze(0)
        q, scale, s = codec.encode(wt, st)
        for r, (n, sc) in enumerate(rows):
            ref, k = reference_encode_row(n, sc, min_exp, lut)
            if q[0, r].tolist() != ref or float(scale[0, r, 0]) != math.ldexp(1.0, k):
                raise SystemExit(f"FATAL self-test {label}: torch != reference on row {r}")
        print(f"[self-test] {label}: {len(rows)} rows, torch == reference; {s}", flush=True)

    for cols in (6144, 3072):
        rows = [synthetic_row(rng, cols, rng.randint(0, 9), base=rng.randint(110, 130)) for _ in range(48)]
        rows += [adversarial_row(cols, 9), ([0] * cols, [127] * (cols // BLOCK)), ([8] * cols, [3] * (cols // BLOCK))]
        check(rows, f"synthetic cols={cols}")
        n, sc = adversarial_row(cols, 11)
        try:
            codec.encode(
                torch.tensor([[list(pack_row(n))]], dtype=torch.uint8), torch.tensor([[sc]], dtype=torch.uint8)
            )
            raise SystemExit("FATAL self-test: a spread-11 row passed")
        except InexactError:
            pass
    if src_dir:
        src = Source(src_dir)
        pfx, lay, shapes = src.expert_layout()
        for L in layers[:2] or [min(lay), max(lay)]:
            for wn in WEIGHTS:
                (R, C2), (_, NB) = shapes[wn]
                for e in (0, lay[L] - 1):
                    b = f"{pfx}.layers.{L}.block_sparse_moe.experts.{e}.{wn}"
                    rows = []
                    for r0 in (0, R // 2, R - 32):
                        wb = src.raw(b + ".weight", r0 * C2, 32 * C2)
                        sb = src.raw(b + ".weight_scale", r0 * NB, 32 * NB)
                        rows += [
                            (unpack_row(wb[i * C2 : (i + 1) * C2]), list(sb[i * NB : (i + 1) * NB])) for i in range(32)
                        ]
                    check(rows, f"real layer {L} {wn} expert {e}")
    print("SELF_TEST_OK", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="amd/MiniMax-M3-MXFP4 snapshot directory")
    ap.add_argument("--dst", help="output directory for the FP8 checkpoint")
    ap.add_argument("--layers", help="expert layers for this worker, e.g. 3-8")
    ap.add_argument("--rest", action="store_true", help="shared experts (BF16) and the byte-for-byte rest")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--min-exp", type=int, default=DEFAULT_MIN_EXP, help="lowest e4m3 exponent allowed")
    ap.add_argument("--expert-chunk", type=int, default=8, help="experts per torch batch")
    ap.add_argument("--shard-size-gb", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if not (FP8_EXP_MIN_NORMAL <= args.min_exp <= 0):
        ap.error("--min-exp must be in [-6, 0]")
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)
    if args.self_test:
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
    elif args.layers:
        layers = parse_layers(args.layers)
        w = ShardWriter(args.dst, f"L{layers[0]:02d}-{layers[-1]:02d}", args.shard_size_gb)
        w.clean_stale()
        stats = convert_layers(src, w, layers, TorchCodec(args.min_exp), args.expert_chunk)
    else:
        ap.error("pass --layers, --rest, --finalize or --self-test")
    w.flush()
    w.save_manifest(stats)
    print(f"[fp8] worker {w.tag} done: {len(w.written)} tensors", flush=True)


if __name__ == "__main__":
    main()
