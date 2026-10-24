#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""gdn-prefix-cache: correct prefix caching for Gated DeltaNet hybrids, keeping the full KV pool.

Qwen3.5 / Qwen3.6 / Qwen3-Next class models mix Gated DeltaNet (GDN) layers with full attention.
With --enable-prefix-caching vLLM puts the GDN layers in Mamba "align" mode: every KV cache group
uses one block size B (896 tokens on the 27B models), prefill chunks end on multiples of B, and a
full block must keep the recurrent state after its last token, so a hit at k * B tokens resumes
from block k - 1. vllm-gaudi v0.26.0 and releases/v0.30.0 break that contract:

  1. Compact GDN (VLLM_COMPACT_GDN=1, the default) keeps one state slot per request and never
     writes a state into a KV block, so a hit resumes from whatever the slot's previous owner left.
  2. vllm-gaudi resets the live cache_config.block_size to 128 after the KV cache and the scheduler
     were built with B, so the scheduler cuts chunks on 128-token multiples. On v0.30.0 the HPU
     async scheduler also aligns GDN chunks to the 128-token mamba chunk instead of B.
  3. The prefill caching branch is Mamba2's and asserts on a merged prefill batch.
  4. New attention blocks are not zeroed (vLLM lists them in new_block_ids_to_zero), and a GDN
     slot is cleared by multiplying with a mask, so a NaN left in memory stays (0 * NaN = NaN).

The fix keeps the compact slots and stores each cached block's GDN state as a checkpoint inside
the attention K/V pages of that block. A KV block belongs to one cache group at a time, so the
attention rows of a block owned by a GDN group are unused, and the page is already sized to hold
one GDN state. Checkpoints move as bit copies through int32 views, outside the compiled graphs:
after a step whose scheduled tokens end on a block boundary the slot is saved into that block;
before a step that starts from a hit (or a resumed preemption) the block is restored into the
slot. Preempted requests hand their slot back. New attention blocks, new requests' slots, the
spare kernel block and the two reserved slots are zeroed. The scheduler aligns chunks to B taken
from the KV cache config and refuses the opt-in Mamba prefix features this layout does not
implement. The KV pool keeps its full size (the non-compact layout scales it by about 0.53).

Inert unless prefix caching is on and the model has GDN layers. Refuses to serve (RuntimeError at
KV cache init) with VLLM_COMPACT_GDN=0, speculative decoding, the KV defragmenter, KV groups of
different block sizes, or a layout the checkpoints do not fit. Needs hybrid-decode-pad first.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "gdn-prefix-cache"
MARKER = f"[gaudi-fix:{FIX_ID}]"

# ---- runner: GDN state indices come from the fix when it is active

RUNNER_EDITS = (
    (
        "compact note",
        """                    logger.warning("Compact GDN mode does not support prefix caching.")
""",
        """                    logger.info("@M@ compact GDN with prefix caching: in-page GDN checkpoints")
""",
    ),
    (
        "prefill indices",
        """            req_indices = [self.input_batch.req_id_to_index[req_id] for req_id in contents.req_ids]
            if self.use_prefix_caching:
                load_state_indices_cpu = self.prepare_mamba_state_idxs(req_indices, block_idx_last_computed_token_cpu,
                                                                       target_bs)
""",
        """            req_indices = [self.input_batch.req_id_to_index[req_id] for req_id in contents.req_ids]
            if self.use_prefix_caching and getattr(self, "_gdn_pc", None):  # @M@
                load_state_indices_cpu = store_state_indices_cpu = _gdn_pc_indices(
                    self, req_indices, context_lens[:len(req_indices)], query_lens[:len(req_indices)], target_bs)
            elif self.use_prefix_caching:
                load_state_indices_cpu = self.prepare_mamba_state_idxs(req_indices, block_idx_last_computed_token_cpu,
                                                                       target_bs)
""",
    ),
    (
        "prefill caching range",
        """            if self.use_prefix_caching:
                assert len(contents.req_ids) == 1
""",
        """            if self.use_prefix_caching and not getattr(self, "_gdn_pc", None):  # @M@
                assert len(contents.req_ids) == 1
""",
    ),
    (
        "prefill caching range copy",
        """            if self.use_prefix_caching:
                blocks_caching_range = async_h2d_copy(all_blocks_caching_ranges_cpu, device=self.device)
""",
        """            if self.use_prefix_caching and not getattr(self, "_gdn_pc", None):  # @M@
                blocks_caching_range = async_h2d_copy(all_blocks_caching_ranges_cpu, device=self.device)
""",
    ),
    (
        "decode indices",
        """            req_indices = list(range(num_decodes))
            if self.use_prefix_caching:
                load_state_indices_cpu = self.prepare_mamba_state_idxs(req_indices, block_idx_last_computed_token_cpu,
                                                                       padded_batch_size)
""",
        """            req_indices = list(range(num_decodes))
            if self.use_prefix_caching and getattr(self, "_gdn_pc", None):  # @M@
                load_state_indices_cpu = store_state_indices_cpu = _gdn_pc_indices(
                    self, req_indices, [int(x) for x in context_lens[:num_decodes]],
                    [int(x) for x in num_scheduled_tokens[:num_decodes]], padded_batch_size)
            elif self.use_prefix_caching:
                load_state_indices_cpu = self.prepare_mamba_state_idxs(req_indices, block_idx_last_computed_token_cpu,
                                                                       padded_batch_size)
""",
    ),
)

# Names the appended helpers use; each must occur exactly once in the runner.
RUNNER_REQUIRED = (
    "class HPUModelRunner(",
    "\n_GDN_MAMBA_TYPES: ",
    "\nimport os\n",
    "    MambaSpec,\n",
    "    FullAttentionSpec,\n",
    "    def prepare_mamba_state_idxs(self, req_indices, block_table_offsets, target_bs):\n",
    "    def sample_tokens(self, grammar_output",
    "        self._gdn_req_to_base_slot: dict[str, int] = {}\n",
    "        self.warmup_mode: bool = False\n",
    "get_kv_cache_shape(num_kernel_blocks + 1,",
)

RUNNER_HELPERS = '''

# ---- @M@ in-page GDN checkpoints for prefix caching --------------------------------------
# Each cached block's GDN state (conv + SSM, every GDN layer) is stored inside the attention
# K/V pages of that block, which the attention group never reads while a GDN group owns the
# block. Copies are bit-exact int32 row copies, eager, outside the compiled graphs.
import math as _gdn_pc_math

_GDN_PC_STATS = {"saves": 0, "restores": 0, "zeroed_blocks": 0, "zeroed_slots": 0, "freed_slots": 0, "steps": 0}


def _gdn_pc_kv(runner, layer_name):
    kv = runner.vllm_config.compilation_config.static_forward_context[layer_name].kv_cache
    if isinstance(kv, list) and len(kv) == 1 and isinstance(kv[0], (tuple, list)):
        kv = kv[0]
    return kv


def _gdn_pc_rows(t, row_i32):
    """Contiguous tensor -> int32 view of its bytes, shaped [-1, row_i32] (a view, no copy)."""
    if not t.is_contiguous():
        raise RuntimeError("@M@ in-page GDN checkpoints need contiguous KV and state tensors")
    i32 = t.reshape(-1).view(torch.int32)
    n = (i32.numel() // row_i32) * row_i32
    return i32[:n].view(-1, row_i32)


def _gdn_pc_setup(runner):
    """After initialize_kv_cache: check the layout and map where each checkpoint goes."""
    runner._gdn_pc = None
    if not getattr(runner, "num_gdn", 0) or not runner.vllm_config.cache_config.enable_prefix_caching:
        return
    cfg = runner.kv_cache_config
    groups = cfg.kv_cache_groups
    gdn = [gid for gid, g in enumerate(groups)
           if isinstance(g.kv_cache_spec, MambaSpec) and g.kv_cache_spec.mamba_type in _GDN_MAMBA_TYPES]
    attn = [g for g in groups if isinstance(g.kv_cache_spec, FullAttentionSpec)]
    if not gdn:
        raise RuntimeError("@M@ prefix caching on a GDN model but no GDN KV cache group was found")
    if not runner._compact_gdn_group_ids:
        raise RuntimeError("@M@ needs VLLM_COMPACT_GDN=1: the in-page layout keeps the compact GDN slots")
    if set(runner._compact_gdn_group_ids) != set(gdn):
        raise RuntimeError("@M@ compact GDN groups %s differ from the GDN groups %s"
                           % (sorted(runner._compact_gdn_group_ids), gdn))
    if runner.vllm_config.speculative_config is not None:
        raise RuntimeError("@M@ speculative decoding is not supported with in-page GDN checkpoints")
    if getattr(runner, "defragmenter", None) is not None and getattr(runner.defragmenter, "enabled", False):
        raise RuntimeError("@M@ the KV defragmenter moves blocks without their GDN checkpoints; disable it")
    if {groups[g].kv_cache_spec.mamba_cache_mode for g in gdn} != {"align"}:
        raise RuntimeError("@M@ expected mamba_cache_mode 'align' for the GDN groups")
    sizes = {g.kv_cache_spec.block_size for g in groups}
    if len(sizes) != 1:
        raise RuntimeError("@M@ KV cache groups use different block sizes %s" % sorted(sizes))
    if len(attn) != 1 or len(attn) + len(gdn) != len(groups):
        raise RuntimeError("@M@ needs one attention cache group and GDN groups only")
    B = sizes.pop()

    # Layer p of every group shares one KV cache tensor slot: pair attention layer p with GDN
    # layer p of each group. The GDN layers at one position share their compact state tensors.
    nlayers = {len(groups[g].layer_names) for g in gdn}
    if len(nlayers) != 1 or len(attn[0].layer_names) < min(nlayers):
        raise RuntimeError("@M@ GDN groups differ in layer count, or there are fewer attention layers")
    entries = []
    for pos in range(nlayers.pop()):
        kc, vc = _gdn_pc_kv(runner, attn[0].layer_names[pos])[:2]
        conv, ssm = _gdn_pc_kv(runner, groups[gdn[0]].layer_names[pos])[:2]
        for g in gdn[1:]:
            o = _gdn_pc_kv(runner, groups[g].layer_names[pos])
            if o[0].data_ptr() != conv.data_ptr() or o[1].data_ptr() != ssm.data_ptr():
                raise RuntimeError("@M@ GDN layers at position %d do not share compact state tensors" % pos)
        entries.append((kc, vc, conv, ssm))

    def row_bytes(t):
        return t[0].numel() * t.element_size()

    kc, vc, conv, ssm = entries[0]
    page, ssm_b, conv_b = B * row_bytes(kc), row_bytes(ssm), row_bytes(conv)
    # Copy unit g bytes: the gcd of the page and state sizes, so one save or restore is one
    # index_select + index_copy_ per tensor pair.
    g = _gdn_pc_math.gcd(_gdn_pc_math.gcd(page, ssm_b), conv_b)
    if g % 4 or g < 256:
        raise RuntimeError("@M@ page %d B, SSM %d B and conv %d B share no copy row of 256 B or more"
                           % (page, ssm_b, conv_b))
    P, S, V = page // g, ssm_b // g, conv_b // g
    a = min(P, S)  # SSM rows in the K page; the rest of the SSM state and the conv state go to the V page
    if S - a + V > P:
        raise RuntimeError("@M@ a GDN state (%d B) does not fit the K and V pages of one block" % (ssm_b + conv_b))
    nb = cfg.num_blocks
    views = []
    for kc, vc, conv, ssm in entries:
        if kc.shape[0] < nb * B or vc.shape[0] < nb * B:
            raise RuntimeError("@M@ attention KV tensor has %d token rows for %d blocks of %d" % (kc.shape[0], nb, B))
        if row_bytes(kc) != row_bytes(vc) or row_bytes(kc) * B != page or row_bytes(ssm) != ssm_b \\
                or row_bytes(conv) != conv_b:
            raise RuntimeError("@M@ KV cache tensors differ in shape")
        views.append(tuple(_gdn_pc_rows(t, g // 4) for t in (kc, vc, ssm, conv)))

    # Decode padding must already sit in the spare kernel block after the pool (hybrid-decode-pad).
    kb = int(runner.attn_block_size)
    if runner._PAD_BLOCK_ID * kb < nb * B:
        raise RuntimeError("@M@ decode padding writes inside the KV pool; apply hybrid-decode-pad")
    # Memory no request owns but padded batches read: the spare kernel block after the pool and
    # compact slots 0 (unused) and -1 (padding). Zeroed after every step so a NaN cannot persist.
    n = int(ssm.shape[0])
    pad_views = []
    for kc2, vc2, ssm2, conv2 in views:
        pad_views += [kc2[nb * P:], vc2[nb * P:], ssm2[:S], ssm2[(n - 1) * S:], conv2[:V], conv2[(n - 1) * V:]]
    runner._gdn_pc = {"B": B, "groups": gdn, "P": P, "S": S, "V": V, "a": a, "C": g // 4, "views": views,
                      "pad_views": pad_views, "valid": {}, "pending": [], "seen": {}, "owner": {}, "warmup": False}
    logger.info("@M@ active: GDN groups %s, block %d tokens, %d KV blocks, %d layer positions; GDN state %d B "
                "per layer stored as %d B in the K page and %d B in the V page of a %d B page; %d compact slots",
                gdn, B, nb, len(entries), ssm_b + conv_b, a * g, (S - a + V) * g, page, n)


def _gdn_pc_copy(runner, pairs, to_slots):
    """pairs: [(compact slot, KV block)]. to_slots False saves slot -> block, True restores."""
    st = runner._gdn_pc
    P, S, V, a = st["P"], st["S"], st["V"], st["a"]
    bk, sk, bv1, sv1, bv2, sv2 = [], [], [], [], [], []
    for slot, blk in pairs:
        bk.extend(range(blk * P, blk * P + a))
        sk.extend(range(slot * S, slot * S + a))
        bv1.extend(range(blk * P, blk * P + S - a))
        sv1.extend(range(slot * S + a, slot * S + S))
        bv2.extend(range(blk * P + S - a, blk * P + S - a + V))
        sv2.extend(range(slot * V, slot * V + V))
    dev = st["views"][0][0].device
    bk, sk, bv1, sv1, bv2, sv2 = [torch.tensor(x, dtype=torch.long).to(dev) for x in (bk, sk, bv1, sv1, bv2, sv2)]
    for kc2, vc2, ssm2, conv2 in st["views"]:
        if to_slots:
            ssm2.index_copy_(0, sk, kc2.index_select(0, bk))
            if S > a:
                ssm2.index_copy_(0, sv1, vc2.index_select(0, bv1))
            conv2.index_copy_(0, sv2, vc2.index_select(0, bv2))
        else:
            kc2.index_copy_(0, bk, ssm2.index_select(0, sk))
            if S > a:
                vc2.index_copy_(0, bv1, ssm2.index_select(0, sv1))
            vc2.index_copy_(0, bv2, conv2.index_select(0, sv2))
    _GDN_PC_STATS["restores" if to_slots else "saves"] += len(pairs)


def _gdn_pc_zero(views):
    if views:
        try:
            torch._foreach_zero_(views)
        except (RuntimeError, NotImplementedError):
            for v in views:
                v.zero_()


def _gdn_pc_zero_blocks(runner, blocks):
    """Zero the K/V pages of KV blocks newly given to the attention group: decode reads whole
    kernel blocks and masks the tail, and a mask does not cancel NaN left by a checkpoint."""
    st = runner._gdn_pc
    P = st["P"]
    rows = []
    for blk in blocks:
        rows.extend(range(blk * P, (blk + 1) * P))
    dev = st["views"][0][0].device
    idx = torch.tensor(rows, dtype=torch.long).to(dev)
    zeros = torch.zeros(len(rows), st["C"], dtype=torch.int32, device=dev)
    for kc2, vc2, _, _ in st["views"]:
        kc2.index_copy_(0, idx, zeros)
        vc2.index_copy_(0, idx, zeros)
    _GDN_PC_STATS["zeroed_blocks"] += len(blocks)


def _gdn_pc_zero_new_slots(runner):
    """Zero the compact slots of requests that got a base slot since the last step: the GDN layers
    clear a new request's state by multiplying with 0, which keeps a NaN left by the slot's last owner."""
    st = runner._gdn_pc
    S, V, G = st["S"], st["V"], int(runner._num_gdn_groups)
    cur = runner._gdn_req_to_base_slot
    fresh = sorted(b for rid, b in cur.items() if st["owner"].get(b) != rid)
    views = []
    for b in fresh:
        for gi in range(G):
            s = b * G + gi + 1
            for _, _, ssm2, conv2 in st["views"]:
                views += [ssm2[s * S:(s + 1) * S], conv2[s * V:(s + 1) * V]]
    _gdn_pc_zero(views)
    _GDN_PC_STATS["zeroed_slots"] += len(fresh)
    st["owner"] = {b: rid for rid, b in cur.items()}


def _gdn_pc_indices(runner, req_indices, ctx_lens, sched_lens, target_bs):
    """Compact slot per request and GDN group, used for both load and store. Restores now, before
    the forward, every slot that does not hold the state after ctx tokens; queues a save for every
    request whose scheduled tokens end on a block boundary (written after the forward)."""
    st = runner._gdn_pc
    B = st["B"]
    idx = runner.prepare_mamba_state_idxs(req_indices, [0] * len(req_indices), target_bs)
    if st["warmup"]:
        return idx
    bt = runner.input_batch.block_table
    restores = []
    for i, r in enumerate(req_indices):
        req_id = runner.input_batch.req_ids[r]
        c, n = int(ctx_lens[i]), int(sched_lens[i])
        if st["seen"].get(req_id) == (c, n):
            continue  # the same request prepared twice in one step
        st["seen"][req_id] = (c, n)
        if c > 0 and st["valid"].get(req_id) != c:
            if c % B:
                raise RuntimeError("@M@ request %s resumes at %d tokens, not a multiple of %d, and its GDN slot "
                                   "does not hold that state" % (req_id, c, B))
            for gid in st["groups"]:
                blk = int(bt[gid].get_cpu_tensor()[r, c // B - 1])
                if blk == 0:
                    raise RuntimeError("@M@ request %s resumes at %d tokens from the null block" % (req_id, c))
                restores.append((int(idx[gid, i]), blk))
        end = c + n
        if end > 0 and end % B == 0:
            for gid in st["groups"]:
                blk = int(bt[gid].get_cpu_tensor()[r, end // B - 1])
                if blk == 0:
                    raise RuntimeError("@M@ request %s: the block holding token %d is the null block"
                                       % (req_id, end - 1))
                st["pending"].append((int(idx[gid, i]), blk))
        st["valid"][req_id] = end
    if restores:
        _gdn_pc_copy(runner, restores, True)
    return idx


def _gdn_pc_wrap_init_kv(orig):
    def initialize_kv_cache(self, kv_cache_config):
        out = orig(self, kv_cache_config)
        _gdn_pc_setup(self)
        return out
    return initialize_kv_cache


def _gdn_pc_wrap_update_states(orig):
    def _update_states(self, scheduler_output):
        st = getattr(self, "_gdn_pc", None)
        if not st:
            return orig(self, scheduler_output)
        if getattr(scheduler_output, "kv_cache_block_copies", None):
            raise RuntimeError("@M@ the scheduler asked for KV block copies (a partial-block Mamba hit), "
                               "which this layout never produces")
        scheduled = scheduler_output.num_scheduled_tokens
        for req_id in sorted(getattr(scheduler_output, "preempted_req_ids", None) or ()):
            st["valid"].pop(req_id, None)
            if req_id in scheduled:
                continue  # resumed in this same step: keeps its slot and is restored from its checkpoint
            slot = self._gdn_req_to_base_slot.pop(req_id, None)
            if slot is not None:  # hand the slot back so preemption cannot run the slots dry
                self._gdn_slot_free_list.append(slot)
                _GDN_PC_STATS["freed_slots"] += 1
        for req_id in scheduler_output.finished_req_ids:
            st["valid"].pop(req_id, None)
        new_blocks = getattr(scheduler_output, "new_block_ids_to_zero", None)
        if new_blocks:
            with torch.inference_mode():
                _gdn_pc_zero_blocks(self, new_blocks)
        out = orig(self, scheduler_output)
        with torch.inference_mode():
            _gdn_pc_zero_new_slots(self)
        return out
    return _update_states


def _gdn_pc_wrap_sample_tokens(orig):
    def sample_tokens(self, *args, **kwargs):
        st = getattr(self, "_gdn_pc", None)
        if not st:
            return orig(self, *args, **kwargs)
        st["seen"], st["pending"] = {}, []
        st["warmup"] = bool(self.warmup_mode)
        out = orig(self, *args, **kwargs)
        pending, st["pending"] = st["pending"], []
        with torch.inference_mode():
            if pending and not st["warmup"]:
                # Same stream as the forward that just ran, so the copies see its final states.
                _gdn_pc_copy(self, pending, False)
            _gdn_pc_zero(st["pad_views"])
        _GDN_PC_STATS["steps"] += 1
        if _GDN_PC_STATS["steps"] in (1, 100, 10000, 1000000):
            logger.info("@M@ stats %s", _GDN_PC_STATS)
        return out
    return sample_tokens


HPUModelRunner.initialize_kv_cache = _gdn_pc_wrap_init_kv(HPUModelRunner.initialize_kv_cache)
HPUModelRunner._update_states = _gdn_pc_wrap_update_states(HPUModelRunner._update_states)
HPUModelRunner.sample_tokens = _gdn_pc_wrap_sample_tokens(HPUModelRunner.sample_tokens)
'''

# ---- vLLM scheduler: align chunks to the GDN block size from the KV cache config

SCHED_EDITS = (
    (
        "scheduler init",
        """        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
""",
        """        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        # @M@ GDN chunks end on the KV cache block size: vllm-gaudi resets the live
        # cache_config.block_size to 128 after the KV cache was built with larger blocks.
        self._gdn_pc_block = _gdn_pc_block_size(kv_cache_config) if self.need_mamba_block_aligned_split else None
        if self._gdn_pc_block is not None:
            logger.info("@M@ scheduler: GDN prefill chunks end on %d-token blocks", self._gdn_pc_block)
""",
    ),
    (
        "module helper",
        """logger = init_logger(__name__)
""",
        """logger = init_logger(__name__)


def _gdn_pc_block_size(kv_cache_config):
    \"\"\"@M@ The KV block size of the GDN cache groups, or None for models without GDN layers.\"\"\"
    sizes = {
        g.kv_cache_spec.block_size
        for g in kv_cache_config.kv_cache_groups
        if getattr(getattr(g.kv_cache_spec, "mamba_type", None), "name", None) in ("GDN_ATTN", "LINEAR")
    }
    return sizes.pop() if len(sizes) == 1 else None
""",
    ),
    (
        "scheduler features",
        """        # Counts of non-empty steps scheduled / processed. update_from_output
""",
        """        if self._gdn_pc_block is not None:  # @M@ refuse what the in-page layout does not implement
            _gdn_bad = [
                _n
                for _n in (
                    "mamba_partial_cache_hit",
                    "mamba_has_prefill_checkpoint_blocks",
                    "mamba_fine_grained_prefix_cache",
                )
                if getattr(self, _n, False)
            ]
            if getattr(self, "hash_block_size", self._gdn_pc_block) != self._gdn_pc_block:
                _gdn_bad.append("hash block size %s" % self.hash_block_size)
            if _gdn_bad:
                raise RuntimeError("@M@ unsupported with in-page GDN checkpoints: %s" % _gdn_bad)
        # Counts of non-empty steps scheduled / processed. update_from_output
""",
    ),
    (
        "scheduler split",
        """        block_size = self.cache_config.block_size
        # The last block-aligned position whose state can be cached. With
""",
        """        block_size = self._gdn_pc_block or self.cache_config.block_size  # @M@
        # The last block-aligned position whose state can be cached. With
""",
    ),
)

# ---- vllm-gaudi async scheduler: its chunk override must defer to the split above. On v0.30.0 it
# aligns GDN chunks to the 128-token mamba chunk; on v0.26.0 it already falls through for GDN.

ASYNC_SPLIT_DOC = (
    '''        the runner assert on an unaligned context_lens.
        """
''',
    '''        memory-layout reasons, we substitute mamba_chunk_size here.
        """
''',
)
ASYNC_GUARD = """        if getattr(self, "_gdn_pc_block", None):  # @M@ chunks end on the GDN block (a multiple of 128)
            return super()._mamba_block_aligned_split(request, num_new_tokens, num_new_local_computed_tokens,
                                                      num_external_computed_tokens)
"""


class Drift(Exception):
    """The installed source does not match what this fix expects."""


def replace_once(src: str, old: str, new: str, label: str) -> str:
    n = src.count(old)
    if n != 1:
        raise Drift(f"anchor {label!r} found {n} times, expected 1")
    return src.replace(old, new, 1)


def require_once(src: str, needle: str, label: str) -> None:
    n = src.count(needle)
    if n != 1:
        raise Drift(f"{label}: {needle.strip()!r} found {n} times, expected 1")


def _m(text: str) -> str:
    return text.replace("@M@", MARKER)


def patch_runner(src: str, ctx: dict[str, str]) -> str:
    if "[gaudi-fix:hybrid-decode-pad]" not in src and "self._PAD_BLOCK_ID = num_blocks * split" not in src:
        raise Drift("hybrid-decode-pad is not applied; apply it first (apply_all.py does)")
    for needle in RUNNER_REQUIRED:
        require_once(src, needle, "hpu_model_runner.py")
    require_once(src, "\nlogger = init_logger()\n", "hpu_model_runner.py")
    for label, old, new in RUNNER_EDITS:
        src = replace_once(src, old, _m(new), label)
    return src.rstrip("\n") + "\n" + _m(RUNNER_HELPERS)


def patch_scheduler(src: str, ctx: dict[str, str]) -> str:
    for label, old, new in SCHED_EDITS:
        src = replace_once(src, old, _m(new), label)
    return src


def patch_async_scheduler(src: str, ctx: dict[str, str]) -> str:
    require_once(src, "    def _mamba_block_aligned_split(\n", "hpu_async_scheduler.py")
    found = [doc for doc in ASYNC_SPLIT_DOC if src.count(doc) == 1]
    if len(found) != 1:
        raise Drift("hpu_async_scheduler.py: _mamba_block_aligned_split docstring end not found exactly once")
    return src.replace(found[0], found[0] + _m(ASYNC_GUARD), 1)


# No exact upstream sentinel: upstream's prefix caching for these models is the non-compact layout.
UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm_gaudi", "v1/worker/hpu_model_runner.py", patch_runner, "runner"),
    ("vllm", "v1/core/sched/scheduler.py", patch_scheduler, "scheduler"),
    ("vllm_gaudi", "v1/core/sched/hpu_async_scheduler.py", patch_async_scheduler, "async_scheduler"),
)


# ---- common driver (identical in every fix) --------------------------------------------------


def locate(pkg: str, rel: str, root: str | None) -> Path | None:
    """Find <pkg>/<rel> under --root, or in the installed package without importing it."""
    if root:
        p = Path(root) / pkg / rel
        return p if p.is_file() else None
    spec = importlib.util.find_spec(pkg)
    for loc in (spec.submodule_search_locations or []) if spec else []:
        p = Path(loc) / rel
        if p.is_file():
            return p
    return None


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".gaudi-fix.tmp")
    tmp.write_text(text)
    os.chmod(tmp, path.stat().st_mode & 0o7777)
    os.replace(tmp, path)
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(f"{path.stem}.*.pyc"):
            pyc.unlink(missing_ok=True)


def run(check: bool, root: str | None) -> int:
    ctx: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for pkg, rel, _, key in TARGETS:
        path = locate(pkg, rel, root)
        if path is None:
            print(f"{MARKER} {pkg}/{rel} not found", file=sys.stderr)
            return 2
        paths[key] = path
        ctx[key] = path.read_text()
    if UPSTREAM_FIXED and all(MARKER not in ctx[k] and needle in ctx[k] for k, needle in UPSTREAM_FIXED):
        print(f"{MARKER} not needed: the upstream fix is present")
        return 0
    done, todo = [], []
    try:
        for _pkg, _rel, fn, key in TARGETS:
            if fn is None:
                continue
            src = ctx[key]
            if MARKER in src:
                done.append(paths[key])
                continue
            new = fn(src, ctx)
            if MARKER not in new:
                raise Drift("patched text lacks the marker")
            compile(new, str(paths[key]), "exec", dont_inherit=True)
            todo.append((paths[key], new))
    except (Drift, SyntaxError) as e:
        print(f"{MARKER} {e}; source drifted, nothing written", file=sys.stderr)
        return 2
    if done and todo:
        print(f"{MARKER} partly applied ({', '.join(map(str, done))}); restore the files and re-run", file=sys.stderr)
        return 2
    if not todo:
        print(f"{MARKER} applied")
        return 0
    if check:
        print(f"{MARKER} not applied (anchors match)")
        return 1
    for path, new in todo:
        write_atomic(path, new)
        print(f"{MARKER} patched {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report the state without writing (0 applied, 1 not applied)")
    ap.add_argument("--root", help="directory holding vllm/ and vllm_gaudi/ (default: the installed packages)")
    args = ap.parse_args()
    return run(args.check, args.root)


if __name__ == "__main__":
    sys.exit(main())
