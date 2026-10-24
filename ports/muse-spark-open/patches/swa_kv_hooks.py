#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hooks for window-sized KV groups (muse_spark_hpu/swa_kv.py) in the vllm-gaudi v0.30.0 runner and worker.

Stock vllm-gaudi gives every layer a full-length KV cache, so Muse Spark's 46 sliding-window layers
store context they never read. muse_spark_hpu/swa_kv.py hands them vLLM's SlidingWindowSpec, so the
hybrid KV cache manager forms window-sized groups. This script inserts the calls into
hpu_model_runner.py (spec conversion, one KV pair per slot shared by the groups, per-group block
tables and metadata) and hpu_worker.py (the profiling spec check). Every inserted branch tests
`self._swg is not None`, and the runner creates that helper only with MUSE_SPARK_SWA_KV=1, so with
the switch off the patched files behave like the stock ones.

Needs runner_window.py applied first (the per-group decode window lists use its arguments).

Usage: swa_kv_hooks.py [--check] [--root DIR_HOLDING_vllm_gaudi]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 source drifted (nothing
written; a half-patched runner would write sliding-window K/V into the full layers' blocks).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

MARKER = "[muse-spark-open:swa-kv-hooks]"
REQUIRES = "[muse-spark-open:runner-window]"

RUNNER_EDITS = [
    (
        "A1 trim-field list",
        "_TYPE_CACHE: dict[str, dict[str, Any]] = {}\n",
        "_TYPE_CACHE: dict[str, dict[str, Any]] = {}\n"
        f"# {MARKER} extra TrimmedAttentionMetadata fields; only muse_spark_hpu.swa_kv.create() fills it\n"
        "_SWG_TRIM_FIELDS: list = []\n",
    ),
    (
        "A2 trim_attn_metadata fields",
        "        'mamba_chunks_to_block_mapping', 'seqlens_offsets_for_blocks', 'image_seg_ids'\n    ])\n",
        "        'mamba_chunks_to_block_mapping', 'seqlens_offsets_for_blocks', 'image_seg_ids'\n"
        f"    ] + _SWG_TRIM_FIELDS)  # {MARKER}\n",
    ),
    (
        "A3 end of __init__",
        "        self.model_has_chunked_attention = False\n        self.is_causal = False\n",
        "        self.model_has_chunked_attention = False\n        self.is_causal = False\n"
        f"        # {MARKER} window-sized KV groups for Muse Spark's sliding-window layers\n"
        "        self._swg = None\n"
        '        if os.environ.get("MUSE_SPARK_SWA_KV", "0") == "1":\n'
        "            from muse_spark_hpu import swa_kv as _muse_swa_kv\n"
        "            self._swg = _muse_swa_kv.create(self)\n",
    ),
    (
        "A4 get_kv_cache_spec",
        "        return kv_cache_spec\n\n    def _update_states(",
        f"        if self._swg is not None:  # {MARKER}\n"
        "            kv_cache_spec = self._swg.convert_specs(self, kv_cache_spec)\n"
        "        return kv_cache_spec\n\n    def _update_states(",
    ),
    (
        "A5 _get_attention_group_id_for_hybrid",
        "    def _get_attention_group_id_for_hybrid(self):\n",
        "    def _get_attention_group_id_for_hybrid(self):\n"
        f"        if getattr(self, '_swg', None) is not None and self._swg.ga_gid is not None:  # {MARKER}\n"
        "            return self._swg.ga_gid\n",
    ),
    (
        "A6a _form_prefill_batch lengths",
        "        query_lens = [len(tids) for tids in contents.token_ids]\n        context_lens = contents.context_lens\n",
        "        query_lens = [len(tids) for tids in contents.token_ids]\n        context_lens = contents.context_lens\n"
        f"        # {MARKER} the per-request lengths, before align_and_pad pads these lists in place\n"
        "        _swg_lens = None if self._swg is None else (list(query_lens), list(context_lens))\n",
    ),
    (
        "A6 _form_prefill_batch return",
        "        return PrefillInputData(request_ids=[req_ids],\n",
        f"        if self._swg is not None:  # {MARKER}\n"
        "            self._swg.prefill(self, req_ids, _swg_lens[0], _swg_lens[1], attn_metadata, target_bs, target_seq)\n"
        "        return PrefillInputData(request_ids=[req_ids],\n",
    ),
    (
        "A7 _create_decode_input_data return",
        "        return DecodeInputData(num_decodes=num_decodes,\n                               token_ids=token_ids_device,\n",
        f"        if self._swg is not None:  # {MARKER}\n"
        "            self._swg.decode(self, attn_metadata, block_table_cpu_tensor, num_decodes, num_tokens,\n"
        "                             padded_batch_size, index, num_blocks, slot_mapping, decode_block_size)\n"
        "        return DecodeInputData(num_decodes=num_decodes,\n                               token_ids=token_ids_device,\n",
    ),
    (
        "A8 _add_dummy_request block ids",
        "for g in self.kv_cache_config.kv_cache_groups] if self.num_mamba_like_layers > 0 else [[block_id] *",
        "for g in self.kv_cache_config.kv_cache_groups] if (self.num_mamba_like_layers > 0 or self._swg is not None) "
        "else [[block_id] *",
    ),
    (
        "A9 warmup InputBatch block sizes",
        "                block_sizes=[self.block_size],\n                kernel_block_sizes=[self.block_size],\n"
        "                logitsprocs=build_logitsprocs(",
        "                block_sizes=[self.block_size] * (1 if self._swg is None else self._swg.n_groups),  # "
        + MARKER
        + "\n"
        "                kernel_block_sizes=[self.block_size] * (1 if self._swg is None else self._swg.n_groups),\n"
        "                logitsprocs=build_logitsprocs(",
    ),
    (
        "A10 after initialize_attn_backend",
        "        self.initialize_attn_backend(kv_cache_config)\n",
        "        self.initialize_attn_backend(kv_cache_config)\n"
        f"        if self._swg is not None:  # {MARKER}\n"
        "            self._swg.on_init_kv_cache(self, kv_cache_config)\n",
    ),
    (
        "A11 KV allocation",
        "        else:  # non-hybrid scenario\n"
        "            for kv_cache_tensor in kv_cache_config.kv_cache_tensors:\n"
        "                for layer_name in kv_cache_tensor.layers:\n",
        f"        elif self._swg is not None:  # {MARKER} one KV pair per slot, shared by the layers of every group\n"
        "            num_blocks = self._swg.alloc_all(self, kv_cache_config, kv_caches)\n"
        "        else:  # non-hybrid scenario\n"
        "            for kv_cache_tensor in kv_cache_config.kv_cache_tensors:\n"
        "                for layer_name in kv_cache_tensor.layers:\n",
    ),
    (
        "A12 after bind_kv_cache",
        "        bind_kv_cache(kv_caches, self.vllm_config.compilation_config.static_forward_context, self.kv_caches)\n",
        "        bind_kv_cache(kv_caches, self.vllm_config.compilation_config.static_forward_context, self.kv_caches)\n"
        f"        if self._swg is not None:  # {MARKER}\n"
        "            self._swg.after_bind(self, kv_caches)\n",
    ),
]

WORKER_EDITS = [
    (
        "W1 import SlidingWindowSpec",
        "from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig, KVCacheSpec, MambaSpec)\n",
        "from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig, KVCacheSpec, MambaSpec)\n"
        f"from vllm.v1.kv_cache_interface import SlidingWindowSpec  # {MARKER}\n",
    ),
    (
        "W2 profiling spec check",
        "            if isinstance(layer_spec, FullAttentionSpec):\n                dtype = layer_spec.dtype\n",
        f"            if isinstance(layer_spec, (FullAttentionSpec, SlidingWindowSpec)):  # {MARKER}\n"
        "                dtype = layer_spec.dtype\n",
    ),
]


class Drift(Exception):
    pass


def patch(src: str, edits, name: str) -> str:
    for label, old, new in edits:
        n = src.count(old)
        if n != 1:
            raise Drift(f"{name}: anchor {label!r} found {n} times, expected 1")
        src = src.replace(old, new)
    return src + f"\n# {MARKER} applied\n"


def package_dir(root: str | None) -> Path | None:
    if root:
        p = Path(root) / "vllm_gaudi"
        return p if p.is_dir() else None
    spec = importlib.util.find_spec("vllm_gaudi")
    locs = list(spec.submodule_search_locations or []) if spec else []
    return Path(locs[0]) if locs else None


def run(check: bool, root: str | None) -> int:
    pkg = package_dir(root)
    if pkg is None:
        print(f"{MARKER} vllm_gaudi not found", file=sys.stderr)
        return 2
    files = [(pkg / "v1/worker/hpu_model_runner.py", RUNNER_EDITS), (pkg / "v1/worker/hpu_worker.py", WORKER_EDITS)]
    texts = [p.read_text() for p, _ in files]
    done = [MARKER in t for t in texts]
    if all(done):
        print(f"{MARKER} applied")
        return 0
    if any(done):
        print(f"{MARKER} partly applied; restore the files and re-run", file=sys.stderr)
        return 2
    try:
        if REQUIRES not in texts[0]:
            raise Drift("runner_window.py is not applied")
        new = [patch(t, edits, p.name) for t, (p, edits) in zip(texts, files, strict=False)]
        for (p, _), t in zip(files, new, strict=False):
            compile(t, str(p), "exec", dont_inherit=True)
    except (Drift, SyntaxError) as e:
        print(f"{MARKER} {e}; source drifted, nothing written", file=sys.stderr)
        return 2
    if check:
        print(f"{MARKER} not applied (anchors match)")
        return 1
    for (p, _), t in zip(files, new, strict=False):
        tmp = p.with_name(p.name + ".muse.tmp")
        tmp.write_text(t)
        os.replace(tmp, p)
        for pyc in (p.parent / "__pycache__").glob(f"{p.stem}.*.pyc"):
            pyc.unlink(missing_ok=True)
        print(f"{MARKER} patched {p}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report the state without writing")
    ap.add_argument("--root", help="directory holding vllm_gaudi/ (default: the installed package)")
    args = ap.parse_args()
    return run(args.check, args.root)


if __name__ == "__main__":
    sys.exit(main())
