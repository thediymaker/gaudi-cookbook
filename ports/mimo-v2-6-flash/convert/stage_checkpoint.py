#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage the public MiMo-V2.6-Flash checkpoint for the port, without copying any weights.

    python3 stage_checkpoint.py XiaomiMiMo/MiMo-V2.6-Flash-RL OUT_DIR [--revision REV]
    python3 stage_checkpoint.py /path/to/local/checkpoint OUT_DIR

OUT_DIR gets a symlink to every file of the checkpoint except config.json, which is written with two edits:

- quantization_config is removed. The port dequantises the FP8 block weights (fused qkv_proj, the dense MLP of
  layer 0) to bf16 and re-encodes the MXFP4 experts itself, so vLLM must build plain bf16 linear layers. With the
  block left in, vLLM would build its FP8 linear methods.
- layer_types is derived from hybrid_layer_pattern (0 global, 1 sliding). vLLM keys its interleaved
  sliding-window handling on it; without it vLLM gives every layer the 128-token window, so the global layers
  would silently become sliding ones.

The original is kept as config.orig.json. Running it again refreshes the links and the config. A repository id is
resolved with huggingface_hub.snapshot_download, which downloads missing files (about 210 GB) unless
HF_HUB_OFFLINE=1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def resolve(source: str, revision: str | None) -> str:
    if os.path.isdir(source):
        return source
    from huggingface_hub import snapshot_download

    return snapshot_download(source, revision=revision)


def stage(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    for root, _dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            if rel == "." and f in ("config.json", "config.orig.json"):
                continue
            link = os.path.join(dst, rel, f)
            target = os.path.realpath(os.path.join(root, f))
            if os.path.islink(link) and os.readlink(link) == target:
                continue
            if os.path.lexists(link):
                os.unlink(link)
            os.symlink(target, link)

    with open(os.path.join(src, "config.json")) as fh:
        cfg = json.load(fh)
    if cfg.get("model_type") != "mimo_v2" or "hybrid_layer_pattern" not in cfg:
        sys.exit(f"{src}: not a MiMo-V2 checkpoint (model_type {cfg.get('model_type')!r})")
    with open(os.path.join(dst, "config.orig.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    cfg.pop("quantization_config", None)
    pattern = cfg["hybrid_layer_pattern"]
    cfg["layer_types"] = ["sliding_attention" if p == 1 else "full_attention" for p in pattern]
    with open(os.path.join(dst, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    with open(os.path.join(src, "model.safetensors.index.json")) as fh:
        shards = set(json.load(fh)["weight_map"].values())
    missing = sorted(s for s in shards if not os.path.exists(os.path.join(dst, s)))
    print(
        f"staged {src} -> {dst}: {len(shards)} shards, {pattern.count(1)} sliding and {pattern.count(0)} global "
        f"layers, {len(missing)} shards missing"
    )
    if missing:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="Hugging Face repository id or a local checkpoint directory")
    ap.add_argument("out_dir")
    ap.add_argument("--revision", help="checkpoint revision when SOURCE is a repository id")
    args = ap.parse_args()
    stage(resolve(args.source, args.revision), args.out_dir)


if __name__ == "__main__":
    main()
