#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Repack amd/GLM-5.2-MXFP4 (AMD Quark, per-expert tensors) into the fused MXFP4 checkpoint the glm-5-mxfp4 port serves.

AMD's checkpoint holds plain OCP MXFP4 (E2M1 packed two per byte, element 0 in the low nibble; E8M0 scales per 32
input elements), so the routed experts are copied byte for byte, only rearranged:

  routed experts, layers 3-77      per-expert gate / up / down -> model.layers.L.mlp.experts.routed_experts.
                                   {w13,w2}_weight[_scale]: w13 = [gate; up] concatenated on the output dim (not
                                   interleaved), experts stacked. Bytes unchanged.
  routed experts, MTP layer 78     AMD left them in BF16: quantized with mxfp4.quantize_mxfp4_floor (the encoder of the
                                   served checkpoint, see mxfp4.py), then fused the same way
  shared experts, layers 3-77      MXFP4 -> BF16 (vllm-gaudi's MXFP4 method only covers the routed experts)
  everything else                  copied unchanged
  config.json                      quantization_config {"quant_method": "gpt_oss_mxfp4"}; num_nextn_predict_layers 1

Workers can split the layers like the GLM-5.3 converter:

  python3 glm52_quark_to_mxfp4.py --src AMD_DIR --dst OUT_DIR --layers 3-78
  python3 glm52_quark_to_mxfp4.py --src AMD_DIR --dst OUT_DIR --rest
  python3 glm52_quark_to_mxfp4.py --src AMD_DIR --dst OUT_DIR --finalize

--verify REF_DIR replaces --dst: nothing is written, every tensor is compared byte for byte with an existing
checkpoint, exit 1 on any difference. Output: about 399 GiB (1521 tensors). amd/GLM-5.2-MXFP4 is MIT licensed.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ckpt import Checkpoint, Sink, finalize, fused_names, parse_layers  # noqa: E402
from mxfp4 import dequant_mxfp4, quantize_mxfp4_floor  # noqa: E402

_PER_EXPERT = re.compile(r"\.mlp\.experts\.\d+\.(gate|up|down)_proj\.(weight|weight_scale)$")


def is_mxfp4(src: Checkpoint, name: str) -> bool:
    return src.dtype(name) in ("U8", "uint8")


def convert_experts(src: Checkpoint, out: Sink, layers: set[int]) -> None:
    n_exp = src.expert_layers()
    for layer in sorted(set(n_exp) & layers):
        packed = is_mxfp4(src, f"model.layers.{layer}.mlp.experts.0.gate_proj.weight")
        w13, w13s, w2, w2s = [], [], [], []
        for e in range(n_exp[layer]):
            base = f"model.layers.{layer}.mlp.experts.{e}"
            if packed:
                g, u, d = (
                    (src.get(f"{base}.{p}_proj.weight"), src.get(f"{base}.{p}_proj.weight_scale"))
                    for p in ("gate", "up", "down")
                )
            else:
                g, u, d = (quantize_mxfp4_floor(src.get(f"{base}.{p}_proj.weight")) for p in ("gate", "up", "down"))
            w13.append(torch.cat([g[0], u[0]], 0))
            w13s.append(torch.cat([g[1], u[1]], 0))
            w2.append(d[0])
            w2s.append(d[1])
        for name, parts in zip(fused_names(layer), (w13, w13s, w2, w2s), strict=True):
            out.emit(name, torch.stack(parts, 0))
        print(f"[repack] layer {layer}: {n_exp[layer]} experts ({'copied' if packed else 'quantized'})", flush=True)


def convert_rest(src: Checkpoint, out: Sink) -> None:
    for n in sorted(src.shard_of):
        if _PER_EXPERT.search(n) or (".mlp.shared_experts." in n and n.endswith(".weight_scale")):
            continue
        t = src.get(n)
        scale = n[: -len(".weight")] + ".weight_scale"
        if ".mlp.shared_experts." in n and n.endswith(".weight") and scale in src:
            t = dequant_mxfp4(t, src.get(scale)).to(torch.bfloat16)
        out.emit(n, t)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="amd/GLM-5.2-MXFP4 snapshot")
    to = ap.add_mutually_exclusive_group(required=True)
    to.add_argument("--dst", help="output directory")
    to.add_argument("--verify", metavar="REF_DIR", help="compare with an existing checkpoint instead of writing")
    job = ap.add_mutually_exclusive_group(required=True)
    job.add_argument("--layers", help="expert layers of this worker, e.g. 3-78")
    job.add_argument("--rest", action="store_true", help="the non-expert tensors")
    job.add_argument("--finalize", action="store_true", help="index, config and tokenizer files")
    ap.add_argument("--shard-gb", type=float, default=20.0)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    src = Checkpoint(args.src)
    if args.finalize:
        if not args.dst:
            raise SystemExit("--finalize needs --dst")
        finalize(args.src, args.dst, set(src.expert_layers()), mtp_layers=1)
        return 0
    if args.dst:
        os.makedirs(args.dst, exist_ok=True)
    tag = "rest" if args.rest else args.layers.replace(",", "_")
    out = Sink(args.dst, tag, args.shard_gb, args.verify)
    if args.rest:
        convert_rest(src, out)
        if out.ref is not None:
            fused = {n for layer in src.expert_layers() for n in fused_names(layer)}
            extra = sorted(set(out.ref.shard_of) - out.seen - fused)
            for n in extra:
                print(f"[verify] only in {args.verify}: {n}", flush=True)
            if extra:
                return 1
    else:
        convert_experts(src, out, parse_layers(args.layers))
    return out.close()


if __name__ == "__main__":
    sys.exit(main())
