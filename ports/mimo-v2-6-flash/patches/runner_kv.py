#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Runner hooks for MiMo-V2's long-context switches in vllm-gaudi v0.26.0.

1. MIMO_GA_TILE=1. The metadata processor stops building the prefix-prefill bias ([chunk, context + chunk], 8.7 GB
   of bf16 at 512K context with 8192-token chunks, never profiled); MiMo's global layers build their masks per
   query tile (mimo_v2_hpu/attention.py).
2. MIMO_SWA_KV_GROUPS=1. Window-sized KV groups for the sliding layers (mimo_v2_hpu/swa_kv.py): 13 small anchors in
   hpu_model_runner.py and 2 in hpu_worker.py call into the helper.

Every hook is gated on its variable, and the window-sized KV helper also refuses any model type but mimo_v2. With
both variables unset the patched files behave like the unpatched ones: the extra trimmed-metadata field list stays
empty, every inserted branch tests `self._swg is not None`, and the worker change only widens an isinstance to a
spec type the stock runner never emits.

Both files are checked before either is written. Needs patches/runner_window.py first.

Usage: runner_kv.py [--check] [--root DIR_HOLDING_vllm_gaudi]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 source drifted (nothing written).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

MARKER = "[mimo-v2-6-flash:runner-kv]"
REQUIRES = "[mimo-v2-6-flash:runner-window]"

GA_TILE_EDIT = (
    "prefix-prefill bias",
    "        if attn_metadata.attn_bias is not None:\n"
    "            return attn_metadata\n"
    "\n"
    "        prefill_metadata = attn_metadata\n"
    "\n"
    "        seq_lens_t = prefill_metadata.seq_lens_tensor\n"
    '        assert seq_lens_t is not None, "seq_lens_tensor is required to build attn_bias"\n',
    "        if attn_metadata.attn_bias is not None:\n"
    "            return attn_metadata\n"
    f'        if os.environ.get("MIMO_GA_TILE", "0") == "1":  # {MARKER} MiMo builds its masks per query tile\n'
    "            return attn_metadata\n"
    "\n"
    "        prefill_metadata = attn_metadata\n"
    "\n"
    "        seq_lens_t = prefill_metadata.seq_lens_tensor\n"
    '        assert seq_lens_t is not None, "seq_lens_tensor is required to build attn_bias"\n',
)

# (label, old, new); every `old` must occur exactly once.
RUNNER_EDITS = [
    GA_TILE_EDIT,
    (
        "A1 trim-field list",
        "_TYPE_CACHE: dict[str, dict[str, Any]] = {}\n",
        "_TYPE_CACHE: dict[str, dict[str, Any]] = {}\n"
        f"# {MARKER} extra TrimmedAttentionMetadata fields; only mimo_v2_hpu.swa_kv.create() fills it\n"
        "_SWG_TRIM_FIELDS: list = []\n",
    ),
    (
        "A2 trim_attn_metadata fields",
        "        'mamba_chunks_to_block_mapping', 'seqlens_offsets_for_blocks'\n    ])\n",
        "        'mamba_chunks_to_block_mapping', 'seqlens_offsets_for_blocks'\n    ] + _SWG_TRIM_FIELDS)\n",
    ),
    (
        "A3 end of __init__",
        "        self.model_has_chunked_attention = False\n        self.is_causal = False\n",
        "        self.model_has_chunked_attention = False\n        self.is_causal = False\n"
        f"        # {MARKER} window-sized KV groups for MiMo-V2 sliding-window layers\n"
        "        self._swg = None\n"
        '        if os.environ.get("MIMO_SWA_KV_GROUPS", "0") == "1":\n'
        "            from mimo_v2_hpu import swa_kv\n"
        "            self._swg = swa_kv.create(self)\n",
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
        f"        if self._swg is not None and self._swg.ga_gid is not None:  # {MARKER}\n"
        "            return self._swg.ga_gid\n",
    ),
    (
        # align_and_pad pads these lists in place before the return, so take copies first
        "A6a _form_prefill_batch lengths",
        "        query_lens = [len(tids) for tids in contents.token_ids]\n"
        "        context_lens = contents.context_lens\n",
        "        query_lens = [len(tids) for tids in contents.token_ids]\n"
        "        context_lens = contents.context_lens\n"
        f"        # {MARKER} the per-request lengths, before align_and_pad pads these lists in place\n"
        "        _swg_lens = None if self._swg is None else (list(query_lens), list(context_lens))\n",
    ),
    (
        "A6 _form_prefill_batch return",
        "        return PrefillInputData(request_ids=[req_ids],\n",
        f"        if self._swg is not None:  # {MARKER}\n"
        "            self._swg.prefill(self, req_ids, _swg_lens[0], _swg_lens[1], attn_metadata, target_bs, "
        "target_seq)\n"
        "        return PrefillInputData(request_ids=[req_ids],\n",
    ),
    (
        "A7 _create_decode_input_data return",
        "        return DecodeInputData(num_decodes=num_decodes,\n"
        "                               token_ids=token_ids_device,\n",
        f"        if self._swg is not None:  # {MARKER}\n"
        "            self._swg.decode(self, attn_metadata, block_table_cpu_tensor, num_decodes, num_tokens,\n"
        "                             padded_batch_size, index, num_blocks, slot_mapping, decode_block_size)\n"
        "        return DecodeInputData(num_decodes=num_decodes,\n"
        "                               token_ids=token_ids_device,\n",
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
        "A11 non-hybrid KV allocation",
        "        else:  # non-hybrid scenario\n"
        "            for kv_cache_tensor in kv_cache_config.kv_cache_tensors:\n"
        "                for layer_name in kv_cache_tensor.shared_by:\n",
        "        else:  # non-hybrid scenario\n"
        "            for kv_cache_tensor in kv_cache_config.kv_cache_tensors:\n"
        "                for layer_name in kv_cache_tensor.shared_by:\n"
        f"                    if self._swg is not None:  # {MARKER} one KV pair per tensor, shared by its layers\n"
        "                        num_blocks = self._swg.alloc_tensor(self, kv_cache_tensor, kv_cache_config, "
        "kv_caches)\n"
        "                        break\n",
    ),
    (
        "A12 after bind_kv_cache",
        "        bind_kv_cache(kv_caches, self.vllm_config.compilation_config.static_forward_context, "
        "self.kv_caches)\n",
        "        bind_kv_cache(kv_caches, self.vllm_config.compilation_config.static_forward_context, "
        "self.kv_caches)\n"
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

# (file inside vllm_gaudi, edits, marker another patch must have left there first)
TARGETS = (
    ("v1/worker/hpu_model_runner.py", RUNNER_EDITS, REQUIRES),
    ("v1/worker/hpu_worker.py", WORKER_EDITS, None),
)


class Drift(Exception):
    pass


def patch(src: str, edits, requires: str | None) -> str | None:
    """The patched text, or None when the file already carries this patch."""
    if MARKER in src:
        missing = [label for label, _, new in edits if new not in src]
        if missing:
            raise Drift(f"partly patched, missing {missing}")
        return None
    if requires and requires not in src:
        raise Drift("patches/runner_window.py is not applied (install.sh runs it first)")
    for label, old, new in edits:
        n = src.count(old)
        if n != 1:
            raise Drift(f"anchor {label!r} found {n} times, expected 1")
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
    planned = []
    try:
        for rel, edits, requires in TARGETS:
            path = pkg / rel
            if not path.is_file():
                raise Drift(f"{rel} not found")
            new = patch(path.read_text(), edits, requires)
            if new is not None:
                compile(new, str(path), "exec", dont_inherit=True)
                planned.append((path, new))
    except (Drift, SyntaxError) as e:
        print(f"{MARKER} {e}; source drifted, nothing written", file=sys.stderr)
        return 2
    if not planned:
        print(f"{MARKER} applied")
        return 0
    if check:
        print(f"{MARKER} not applied (anchors match)")
        return 1
    for path, new in planned:
        tmp = path.with_name(path.name + ".mimo.tmp")
        tmp.write_text(new)
        os.replace(tmp, path)
        for pyc in (path.parent / "__pycache__").glob(f"{path.stem}.*.pyc"):
            pyc.unlink(missing_ok=True)
        print(f"{MARKER} patched {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report the state without writing")
    ap.add_argument("--root", help="directory holding vllm_gaudi/ (default: the installed package)")
    args = ap.parse_args()
    return run(args.check, args.root)


if __name__ == "__main__":
    sys.exit(main())
