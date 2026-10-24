#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert zai-org/GLM-5.3-BF16 into the MXFP4 checkpoint the glm-5-mxfp4 port serves.

There is no public MXFP4 build of GLM-5.3, so this script makes one from the BF16 release (not from the FP8 repo:
the FP8 weights are themselves a quantization of the BF16 ones, and MXFP4 from BF16 has the smaller error).

  routed experts (layers 3-77 and the MTP layer 78)  BF16 -> MXFP4 with the Quark rule (mxfp4.quantize_mxfp4),
                                                     w13 = [gate; up] concatenated on the output dim (not
                                                     interleaved), experts stacked:
                                                     model.layers.L.mlp.experts.routed_experts.{w13,w2}_weight[_scale]
  everything else                                    copied unchanged (BF16 / FP32): attention, DSA indexer,
                                                     shared experts, router, norms, embeddings, lm_head, the MTP
                                                     layer's own projections and norms
  config.json                                        quantization_config {"quant_method": "gpt_oss_mxfp4"};
                                                     num_nextn_predict_layers stays 1 (the MTP layer is kept)

The encode is CPU-bound (about 724 B routed-expert parameters). Split it over workers that each write their own shards,
then finalize:

  python3 glm53_bf16_to_mxfp4.py --src BF16_DIR --dst OUT_DIR --layers 3-12      # one worker per layer range
  ...
  python3 glm53_bf16_to_mxfp4.py --src BF16_DIR --dst OUT_DIR --layers 67-78
  python3 glm53_bf16_to_mxfp4.py --src BF16_DIR --dst OUT_DIR --rest             # the non-expert tensors
  python3 glm53_bf16_to_mxfp4.py --src BF16_DIR --dst OUT_DIR --finalize         # after every worker finished

--verify REF_DIR replaces --dst for a worker: nothing is written, every tensor the worker would write is compared
byte for byte with the same-named tensor of an existing checkpoint, and the worker exits 1 on any difference.

Output: about 399 GiB (1521 tensors). Peak memory per expert worker: about 10 GB for one layer plus one
shard (--shard-gb).

The output is a derivative of GLM-5.3: it is distributed under the GLM-5.3 License, whose LICENSE file the finalize
step copies next to the weights.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ckpt import MTP_LAYER, Checkpoint, Sink, finalize, fused_names, parse_layers  # noqa: E402
from mxfp4 import quantize_mxfp4  # noqa: E402

_PER_EXPERT = re.compile(r"\.mlp\.experts\.\d+\.")


def check_bf16(src: Checkpoint) -> None:
    fp8 = [n for n in src.shard_of if n.endswith(".weight_scale_inv")]
    if fp8:
        raise SystemExit(f"{src.path} has FP8 scales ({fp8[0]}): use the BF16 release zai-org/GLM-5.3-BF16")


def convert_experts(src: Checkpoint, out: Sink, layers: set[int]) -> None:
    n_exp = src.expert_layers()
    for layer in sorted(set(n_exp) & layers):
        w13, w13s, w2, w2s = [], [], [], []
        for e in range(n_exp[layer]):
            base = f"model.layers.{layer}.mlp.experts.{e}"
            gw, gs = quantize_mxfp4(src.get(f"{base}.gate_proj.weight"))
            uw, us = quantize_mxfp4(src.get(f"{base}.up_proj.weight"))
            dw, ds = quantize_mxfp4(src.get(f"{base}.down_proj.weight"))
            w13.append(torch.cat([gw, uw], 0))
            w13s.append(torch.cat([gs, us], 0))
            w2.append(dw)
            w2s.append(ds)
        for name, parts in zip(fused_names(layer), (w13, w13s, w2, w2s), strict=True):
            out.emit(name, torch.stack(parts, 0))
        print(f"[convert] layer {layer}: {n_exp[layer]} experts", flush=True)


def convert_rest(src: Checkpoint, out: Sink) -> None:
    for n in sorted(src.shard_of):
        if not _PER_EXPERT.search(n):
            out.emit(n, src.get(n))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="zai-org/GLM-5.3-BF16 snapshot")
    to = ap.add_mutually_exclusive_group(required=True)
    to.add_argument("--dst", help="output directory")
    to.add_argument("--verify", metavar="REF_DIR", help="compare with an existing checkpoint instead of writing")
    job = ap.add_mutually_exclusive_group(required=True)
    job.add_argument("--layers", help="expert layers of this worker, e.g. 3-12 or 67-78")
    job.add_argument("--rest", action="store_true", help="the non-expert tensors")
    job.add_argument("--finalize", action="store_true", help="index, config and tokenizer files")
    ap.add_argument("--shard-gb", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    src = Checkpoint(args.src)
    check_bf16(src)
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
        layers = parse_layers(args.layers)
        if MTP_LAYER in layers and MTP_LAYER not in src.expert_layers():
            raise SystemExit(f"no MTP layer {MTP_LAYER} in {args.src}")
        convert_experts(src, out, layers)
    return out.close()


if __name__ == "__main__":
    sys.exit(main())
