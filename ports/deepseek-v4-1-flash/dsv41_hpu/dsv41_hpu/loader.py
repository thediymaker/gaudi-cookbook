# SPDX-License-Identifier: Apache-2.0
"""Load the converted DeepSeek-V4.1 checkpoint into DeepseekV41HPUForCausalLM.

vLLM's default loader would materialize every tensor on every rank: 8 x 518 GB of reads, with the 4.5 GB fused
expert tensors and 98 GB engram tables among them. This reads with safetensors slices instead, so a TP 8 / EP 8 rank
reads its own 48 experts per layer and its 3 engram buckets per table, ~77 GB in all.

Names are the converter's output (convert/convert_to_hpu_mxfp4.py in this port). Everything not listed here (mtp.*,
vision.*, aligner.*, image_*, the image routing bias) is skipped: the port serves text only, without speculative
decoding.
"""

from __future__ import annotations

import json
import logging
import os
import time

import torch
from safetensors import safe_open

logger = logging.getLogger("vllm.dsv41_hpu")  # under "vllm": its handlers print; a bare logger was silent


class _Ckpt:
    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            self.wm = json.load(f)["weight_map"]
        self.files = {}
        self.bytes = 0

    def _f(self, name):
        fn = self.wm[name]
        if fn not in self.files:
            self.files[fn] = safe_open(os.path.join(self.path, fn), framework="pt", device="cpu")
        return self.files[fn]

    def get(self, name: str, rows: slice | None = None) -> torch.Tensor:
        f = self._f(name)
        t = f.get_tensor(name) if rows is None else f.get_slice(name)[rows]
        self.bytes += t.numel() * t.element_size()
        return t


def _put(param: torch.nn.Parameter, t: torch.Tensor, *shard) -> None:
    wl = getattr(param, "weight_loader", None)
    if wl is not None:
        wl(param, t, *shard)
    else:
        assert param.shape == t.shape, (tuple(param.shape), tuple(t.shape))
        param.data.copy_(t.to(param.dtype))


def load_dsv41(model, path: str) -> None:
    t0 = time.time()
    ck = _Ckpt(path)
    r = model.tp_rank
    st = model.stack
    _put(st.embed.weight, ck.get("embed.weight"))
    _put(st.norm, ck.get("norm.weight"))
    _put(model.lm_head.weight, ck.get("head.weight"))
    for L, blk in enumerate(st.layers):
        p, a = f"layers.{L}", blk.attn
        for n in ("wq_a", "wq_b", "wkv", "wo_b"):
            _put(getattr(a, n).weight, ck.get(f"{p}.attn.{n}.weight"))
        G = a.n_local_groups * a.o_lora_rank
        _put(a.wo_a, ck.get(f"{p}.attn.wo_a.weight", slice(r * G, (r + 1) * G)))
        _put(a.q_norm, ck.get(f"{p}.attn.q_norm.weight"))
        _put(a.kv_norm, ck.get(f"{p}.attn.kv_norm.weight"))
        H = a.n_local_heads
        _put(a.attn_sink, ck.get(f"{p}.attn.attn_sink", slice(r * H, (r + 1) * H)))
        if a.role.is_kv_source:
            _put(a.c_wkv.weight, ck.get(f"{p}.attn.compressor.wkv.weight"))
            if a.role.ratio > 1:
                _put(a.c_wgate.weight, ck.get(f"{p}.attn.compressor.wgate.weight"))
            _put(a.c_norm, ck.get(f"{p}.attn.compressor.norm.weight"))
            _put(a.i_wk.weight, ck.get(f"{p}.attn.indexer.wk.weight"))
            _put(a.i_k_norm, ck.get(f"{p}.attn.indexer.k_norm.weight"))
        if a.role.is_index_source:
            _put(a.i_wq_b.weight, ck.get(f"{p}.attn.indexer.wq_b.weight"))
            _put(a.i_weights_proj.weight, ck.get(f"{p}.attn.indexer.weights_proj.weight"))
        _put(blk.attn_norm, ck.get(f"{p}.attn_norm.weight"))
        _put(blk.ffn_norm, ck.get(f"{p}.ffn_norm.weight"))
        for s in ("attn", "ffn"):
            for k in ("fn", "base", "scale"):
                _put(getattr(blk, f"hc_{s}_{k}"), ck.get(f"{p}.hc_{s}_{k}"))

        f = blk.ffn
        _put(f.gate_w, ck.get(f"{p}.ffn.gate.weight"))
        _put(f.gate_bias, ck.get(f"{p}.ffn.gate.bias"))
        ex = slice(f.e0, f.e0 + f.n_local)
        re = f"{p}.ffn.experts.routed_experts"
        for n in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
            _put(getattr(f, n), ck.get(f"{re}.{n}", ex))
        _put(f.shared_w13.weight, ck.get(f"{p}.ffn.shared_experts.w1.weight"), 0)
        _put(f.shared_w13.weight, ck.get(f"{p}.ffn.shared_experts.w3.weight"), 1)
        _put(f.shared_w2.weight, ck.get(f"{p}.ffn.shared_experts.w2.weight"))

        if blk.engram is not None:
            e = blk.engram
            off = 0
            for h in range(e.col0, e.col1):
                w = ck.get(f"{p}.engram.embed.head{h:02d}.weight")
                sc = ck.get(f"{p}.engram.embed.head{h:02d}.scale")
                e.table.data[off : off + w.shape[0]].copy_(w)
                e.table_scale.data[off : off + w.shape[0]].copy_(sc)
                off += w.shape[0]
                del w, sc
            assert off == e.table.shape[0], (off, e.table.shape)
            _put(e.wkv.weight, ck.get(f"{p}.engram.wkv.weight"))
            _put(e.q_weight, ck.get(f"{p}.engram.q_weight"))
            _put(e.k_weight, ck.get(f"{p}.engram.k_weight"))
        if L % 5 == 4:
            logger.warning(
                "dsv41 load: rank %d through layer %d, %.1f GB read, %.0fs", r, L, ck.bytes / 1e9, time.time() - t0
            )
    logger.warning("dsv41 load: rank %d done, %.1f GB read in %.0fs", r, ck.bytes / 1e9, time.time() - t0)
