#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail closed unless DIR is a checkpoint written by convert/convert_to_hpu_mxfp4.py.

The port serves only the converted layout (interleaved MXFP4 experts, bf16 dense weights, engram buckets). The
original FP8 checkpoint has the same architecture name, so vLLM would start loading it and fail late, or worse,
read FP8 bytes as something else. This checks, without reading tensor data:

  - config.json has quantization_config.quant_method == "dsv41_hpu_mxfp4" and interleaved gate/up experts;
  - model.safetensors.index.json exists and every shard it names is present, non-empty, and has a readable header;
  - the routed experts of every layer, the dense weights and the engram buckets are all in the index.

  check_checkpoint.py DIR      exit 0 = servable, 1 = not (the reason is printed)
"""

from __future__ import annotations

import json
import os
import struct
import sys


def fail(msg: str) -> int:
    print(f"[dsv41 checkpoint] FAIL: {msg}", file=sys.stderr)
    return 1


def check(root: str) -> int:
    if not os.path.isdir(root):
        return fail(f"{root} is not a directory (set DSV41_CHECKPOINT to the output of convert/)")
    try:
        with open(os.path.join(root, "config.json")) as f:
            cfg = json.load(f)
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
    except (OSError, ValueError, KeyError) as e:
        return fail(f"cannot read config.json / model.safetensors.index.json in {root}: {e}")
    q = cfg.get("quantization_config") or {}
    if q.get("quant_method") != "dsv41_hpu_mxfp4":
        return fail(
            f"quant_method is {q.get('quant_method')!r}, not 'dsv41_hpu_mxfp4': this is not a converted checkpoint "
            "(run ports/deepseek-v4-1-flash/convert/convert_to_hpu_mxfp4.py on it first)"
        )
    layout = (q.get("routed_experts") or {}).get("w13_layout")
    if layout != "interleaved_gate_up":
        return fail(f"routed_experts.w13_layout is {layout!r}; the port reads interleaved_gate_up")
    tc = cfg.get("text_config") or {}
    n_layers = int(tc.get("num_hidden_layers", 0))
    if n_layers <= 0:
        return fail("config.json text_config has no num_hidden_layers")
    need = [f"layers.{i}.ffn.experts.routed_experts.w13_weight" for i in range(n_layers)]
    need += [f"layers.{i}.ffn.experts.routed_experts.w2_weight_scale" for i in range(n_layers)]
    need += ["embed.weight", "head.weight"]
    missing = [n for n in need if n not in weight_map]
    for lid in tc.get("engram_layer_ids", []):
        if not any(n.startswith(f"layers.{lid}.engram.embed.head") for n in weight_map):
            missing.append(f"layers.{lid}.engram.embed.head*")
    if missing:
        return fail(f"{len(missing)} expected tensors are not in the index, e.g. {missing[:3]}")
    shards = sorted(set(weight_map.values()))
    for fn in shards:
        p = os.path.join(root, fn)
        try:
            size = os.path.getsize(p)
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                if n <= 0 or 8 + n > size:
                    raise ValueError("bad header length")
                json.loads(f.read(n))
        except (OSError, ValueError, struct.error) as e:
            return fail(f"shard {fn}: {e}")
    print(f"[dsv41 checkpoint] OK: {root} ({len(shards)} shards, {len(weight_map)} tensors)")
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    return check(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())
