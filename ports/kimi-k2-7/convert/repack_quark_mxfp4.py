#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Repack amd/Kimi-K2.7-Code-MXFP4 (AMD Quark, one tensor per expert) into vLLM's fused MXFP4 layout.

The output is what vllm-gaudi's native MXFP4 MoE method (HPUGptOssMxfp4MoEMethod, the stock
mixture_of_experts.mxfp4_fused_weights op) loads, and what the single-node recipe serves. Nothing is
re-quantized: the routed experts' FP4 bytes and E8M0 scales are copied as they are.

  routed experts   per layer, fused: <P>.layers.L.mlp.experts.routed_experts.
                     w13_weight        uint8 [384, 2 * 2048, 7168 / 2]   rows [gate; up], E2M1 low nibble first
                     w13_weight_scale  uint8 [384, 2 * 2048, 7168 / 32]  E8M0, one per 32 elements
                     w2_weight         uint8 [384, 7168, 2048 / 2]
                     w2_weight_scale   uint8 [384, 7168, 2048 / 32]
  other quantized  attention (FP8 with a per-channel scale), the dense and shared MLPs: dequantized to bf16,
                   because vLLM's MXFP4 config leaves every Linear outside the routed experts unquantized
  everything else  copied as is (router, norms, embeddings, lm_head, the vision tower and projector)

config.json gets quantization_config {"quant_method": "gpt_oss_mxfp4"}; tokenizer, chat template and the
remote-code files are copied. About 659 GB in 20 GB shards, CPU only.

Usage:
  repack_quark_mxfp4.py --src <amd/Kimi-K2.7-Code-MXFP4 snapshot> --dst <out dir> [--shard-size-gb 20]
  repack_quark_mxfp4.py --src <snapshot> --dst <out dir> --verify [--samples 24]
      re-reads the written checkpoint and checks sampled experts byte for byte against the source tensors
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import shutil
import sys
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
)
BLOCK = 32
FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)
EXPERT_RE = re.compile(r"^(?P<pfx>.*)\.layers\.(?P<L>\d+)\.mlp\.experts\.(?P<E>\d+)\.gate_proj\.weight$")
PROJ = (".gate_proj.", ".up_proj.", ".down_proj.")


def dequant_mxfp4(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """E2M1 two per byte (element 0 in the low nibble) with E8M0 block-32 scales -> float32."""
    nib = torch.stack([(w & 0x0F).long(), (w >> 4).long()], dim=-1).reshape(*w.shape[:-1], w.shape[-1] * 2)
    vals = E2M1[nib]
    scale = torch.pow(2.0, s.float() - 127.0).repeat_interleave(BLOCK, dim=-1)[..., : vals.shape[-1]]
    return vals * scale


def dequant_fp8(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """FP8 with a per-channel (or per-tensor) scale broadcast over the input dimension -> float32."""
    sf = s.float()
    while sf.ndim < w.ndim:
        sf = sf.unsqueeze(-1)
    return w.float() * sf


class Source:
    def __init__(self, src: str):
        shards = sorted(glob.glob(os.path.join(src, "*.safetensors")))
        if not shards:
            sys.exit(f"no safetensors files in {src}")
        self.where: dict[str, str] = {}
        for sp in shards:
            with safe_open(sp, framework="pt") as f:
                for n in f.keys():  # noqa: SIM118 (safe_open is not a dict)
                    self.where[n] = sp

    def get(self, name: str) -> torch.Tensor:
        with safe_open(self.where[name], framework="pt") as f:
            return f.get_tensor(name)

    def dtype(self, name: str) -> str:
        with safe_open(self.where[name], framework="pt") as f:
            return str(f.get_slice(name).get_dtype())

    def expert_layers(self) -> tuple[str, dict[int, int]]:
        """(key prefix, {layer: number of experts}) of the MXFP4 routed-expert layers."""
        prefixes, count = set(), defaultdict(int)
        for n in self.where:
            m = EXPERT_RE.match(n)
            if m:
                prefixes.add(m.group("pfx"))
                count[int(m.group("L"))] = max(count[int(m.group("L"))], int(m.group("E")) + 1)
        if len(prefixes) != 1:
            sys.exit(f"expected one expert key prefix, found {sorted(prefixes)}")
        pfx = prefixes.pop()
        layers = {
            L: n for L, n in count.items() if self.dtype(f"{pfx}.layers.{L}.mlp.experts.0.gate_proj.weight") == "U8"
        }
        if len(layers) != len(count):
            sys.exit(f"routed experts that are not MXFP4 in layers {sorted(set(count) - set(layers))}")
        return pfx, layers


def fused(src: Source, pfx: str, layer: int, n_exp: int) -> dict[str, torch.Tensor]:
    w13, s13, w2, s2 = [], [], [], []
    for e in range(n_exp):
        b = f"{pfx}.layers.{layer}.mlp.experts.{e}"
        w13.append(torch.cat([src.get(f"{b}.gate_proj.weight"), src.get(f"{b}.up_proj.weight")], 0))
        s13.append(torch.cat([src.get(f"{b}.gate_proj.weight_scale"), src.get(f"{b}.up_proj.weight_scale")], 0))
        w2.append(src.get(f"{b}.down_proj.weight"))
        s2.append(src.get(f"{b}.down_proj.weight_scale"))
    o = f"{pfx}.layers.{layer}.mlp.experts.routed_experts"
    return {
        f"{o}.w13_weight": torch.stack(w13),
        f"{o}.w13_weight_scale": torch.stack(s13),
        f"{o}.w2_weight": torch.stack(w2),
        f"{o}.w2_weight_scale": torch.stack(s2),
    }


def repack(args) -> None:
    src = Source(args.src)
    pfx, layers = src.expert_layers()
    print(f"routed experts: {len(layers)} MXFP4 layers under {pfx}.layers.L.mlp.experts.E", flush=True)
    os.makedirs(args.dst, exist_ok=True)
    limit = int(args.shard_size_gb * 1024**3)
    weight_map: dict[str, str] = {}
    buf: dict[str, torch.Tensor] = {}
    state = {"bytes": 0, "shard": 0}

    def flush():
        if not buf:
            return
        fn = f"model-{state['shard']:05d}.safetensors"
        save_file(buf, os.path.join(args.dst, fn), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(buf, fn))
        print(f"wrote {fn}: {len(buf)} tensors, {state['bytes'] / 1e9:.1f} GB", flush=True)
        buf.clear()
        state["bytes"] = 0
        state["shard"] += 1

    def emit(name: str, t: torch.Tensor):
        buf[name] = t.contiguous()
        state["bytes"] += t.numel() * t.element_size()
        if state["bytes"] >= limit:
            flush()

    for layer in sorted(layers):
        for name, t in fused(src, pfx, layer, layers[layer]).items():
            emit(name, t)
        print(f"layer {layer}: {layers[layer]} experts fused", flush=True)

    for n in src.where:
        if ".mlp.experts." in n and any(p in n for p in PROJ) and n.endswith((".weight", ".weight_scale")):
            continue  # routed experts, written above
        if n.endswith(".weight_scale") and n[: -len("_scale")] in src.where:
            continue  # folded into its weight below
        t = src.get(n)
        scale = n + "_scale"
        if n.endswith(".weight") and scale in src.where:
            s = src.get(scale)
            if t.dtype == torch.uint8:
                t = dequant_mxfp4(t, s)
            elif t.dtype in FP8:
                t = dequant_fp8(t, s)
            t = t.to(torch.bfloat16)
        elif t.dtype in FP8:
            t = t.to(torch.bfloat16)
        emit(n, t)
    flush()

    total = sum(os.path.getsize(os.path.join(args.dst, f)) for f in set(weight_map.values()))
    with open(os.path.join(args.dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=1)
    with open(os.path.join(args.src, "config.json")) as f:
        cfg = json.load(f)
    cfg["quantization_config"] = {"quant_method": "gpt_oss_mxfp4"}
    if isinstance(cfg.get("text_config"), dict):
        cfg["text_config"].pop("quantization_config", None)
    with open(os.path.join(args.dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for fn in os.listdir(args.src):
        sp = os.path.join(args.src, fn)
        if (
            os.path.isfile(sp)
            and not fn.endswith(".safetensors")
            and fn not in ("config.json", "model.safetensors.index.json")
        ):
            shutil.copy2(sp, os.path.join(args.dst, fn))
    print(f"done: {args.dst}, {total / 1e9:.1f} GB", flush=True)


def verify(args) -> int:
    """Sampled experts of the written checkpoint must equal the source's per-expert bytes exactly."""
    src = Source(args.src)
    pfx, layers = src.expert_layers()
    out = Source(args.dst)
    rng = random.Random(0)
    picks = [(L, rng.randrange(layers[L])) for L in rng.sample(sorted(layers), min(args.samples, len(layers)))]
    bad = 0
    for layer, e in picks:
        o = f"{pfx}.layers.{layer}.mlp.experts.routed_experts"
        b = f"{pfx}.layers.{layer}.mlp.experts.{e}"
        inter = src.get(f"{b}.gate_proj.weight").shape[0]
        checks = (
            ("w13_weight", slice(0, inter), "gate_proj.weight"),
            ("w13_weight", slice(inter, 2 * inter), "up_proj.weight"),
            ("w13_weight_scale", slice(0, inter), "gate_proj.weight_scale"),
            ("w13_weight_scale", slice(inter, 2 * inter), "up_proj.weight_scale"),
            ("w2_weight", slice(None), "down_proj.weight"),
            ("w2_weight_scale", slice(None), "down_proj.weight_scale"),
        )
        for fused_name, rows, src_name in checks:
            got = out.get(f"{o}.{fused_name}")[e, rows]
            if not torch.equal(got, src.get(f"{b}.{src_name}")):
                bad += 1
                print(f"MISMATCH layer {layer} expert {e} {src_name}")
    print(f"{'VERIFY_OK' if bad == 0 else 'VERIFY_FAIL'}: {len(picks)} experts, {bad} mismatches")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="amd/Kimi-K2.7-Code-MXFP4 snapshot directory")
    ap.add_argument("--dst", required=True, help="output directory")
    ap.add_argument("--shard-size-gb", type=float, default=20.0)
    ap.add_argument("--verify", action="store_true", help="check a written checkpoint against the source")
    ap.add_argument("--samples", type=int, default=24, help="experts to check with --verify")
    args = ap.parse_args()
    if args.verify:
        return verify(args)
    repack(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
