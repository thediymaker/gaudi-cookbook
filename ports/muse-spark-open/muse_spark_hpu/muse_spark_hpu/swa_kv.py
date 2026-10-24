# SPDX-License-Identifier: Apache-2.0
"""Window-sized KV cache for Muse Spark's 46 sliding-window layers on vllm-gaudi v0.30.0.

MUSE_SPARK_SWA_KV=1 (needs patches/swa_kv_hooks.py, which install.sh applies). With the switch off the runner
never creates the helper and runs as stock.

Stock vllm-gaudi emits FullAttentionSpec for every layer, so the window-2048 layers hold a full-length cache like
the 16 global layers (31,744 B per token per card at TP 8). With this module:

* get_kv_cache_spec emits SlidingWindowSpec(2048) for the sliding layers (convert_specs). vLLM core then builds
  4 KV cache groups [global x16, sliding x16, 15, 15] over one backing allocation in which layer l of every group
  aliases the same region, frees blocks that fall out of the window, and applies the sliding-window prefix-cache
  rules and preemption. KV per token drops to the global layers' 8,192 B per card, so a block id costs 1 MiB per
  card and the pool holds 3.875x more tokens.
* initialize_kv_cache allocates ONE (k, v) pair per layer slot (16 at TP 8) and binds it to the layer at that
  slot in every group (alloc_all). The stock loop would give each of the 62 layers its own pair.
* The persistent batch gets one block table per group (on_init_kv_cache).
* Prefill and decode metadata carry, for each sliding group, the slot mapping and the window block list built
  from that group's block table, in two extra metadata fields indexed by group id (None for the global group).
  Shapes and dtypes equal the base tensors, so no new buckets or graph variants.
* The model hands each sliding layer a metadata view with its group's tensors (layer_views); the attention code
  is unchanged.
"""

from __future__ import annotations

import logging
import os
import sys

import torch

logger = logging.getLogger("vllm.muse_spark_hpu")

ENV = "MUSE_SPARK_SWA_KV"
MODEL_TYPE = "muse_spark"
RUNNER_MODULE = "vllm_gaudi.v1.worker.hpu_model_runner"
TRIM_FIELDS = ("kv_group_slot_mapping", "kv_group_window_block_list")
DUMMY_REQ_ID = "-1"  # _create_dummy_prefill_batch_contents (DP padding batches)

# set by after_bind() in the worker; read by the model's layer loop
_ACTIVE = {"on": False, "ga_gid": None}


def requested() -> bool:
    return os.environ.get(ENV, "0") == "1"


def active() -> bool:
    return _ACTIVE["on"]


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


# --------------------------------------------------------------------------
# KV cache specs
# --------------------------------------------------------------------------
def to_sliding_window_spec(spec, window: int):
    """The runner's FullAttentionSpec for a sliding layer, as a SlidingWindowSpec."""
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    return SlidingWindowSpec(
        block_size=spec.block_size,
        num_kv_heads=spec.num_kv_heads,
        head_size=spec.head_size,
        head_size_v=spec.head_size_v,
        dtype=spec.dtype,
        kv_quant_mode=spec.kv_quant_mode,
        page_size_padded=spec.page_size_padded,
        sliding_window=window,
    )


def convert_spec_dict(kv_cache_spec: dict, windows: dict) -> dict:
    """FullAttentionSpec -> SlidingWindowSpec for every layer with a window.

    windows: layer name -> sliding window, or None for a full layer. Raises unless the result
    is a layout the HPU side supports: plain FullAttentionSpec in, at least one full and one
    sliding layer, one window that is a multiple of the block size, and one page size for all
    layers (core would otherwise rescale a group's block size away from the 128-token HPU
    kernel block).
    """
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    out = {}
    for name, spec in kv_cache_spec.items():
        if type(spec) is not FullAttentionSpec:
            raise ValueError(f"{ENV}: layer {name} has {type(spec).__name__}; expected FullAttentionSpec")
        w = windows.get(name)
        out[name] = spec if w is None else to_sliding_window_spec(spec, w)
    slid = [n for n in out if windows.get(n) is not None]
    if not slid or len(slid) == len(out):
        raise ValueError(f"{ENV}: need both full and sliding-window layers, got {len(slid)} sliding of {len(out)}")
    ws = {windows[n] for n in slid}
    bss = {s.block_size for s in out.values()}
    if len(ws) != 1 or len(bss) != 1:
        raise ValueError(f"{ENV}: one window and one block size expected, got windows {ws}, block sizes {bss}")
    (w,), (bs,) = ws, bss
    if w % bs:
        raise ValueError(f"{ENV}: window {w} is not a multiple of block size {bs}")
    pages = {s.page_size_bytes for s in out.values()}
    if len(pages) != 1:
        raise ValueError(f"{ENV}: KV page sizes differ across layers ({sorted(pages)} B)")
    return out


# --------------------------------------------------------------------------
# Per-group metadata builders (pure; the same layout the runner uses for the full group)
# --------------------------------------------------------------------------
def prefill_group_tensors(
    rows: torch.Tensor,
    row_idx,
    ctx_lens,
    q_lens,
    target_bs: int,
    target_seq: int,
    block_size: int,
    win_blocks: int,
    with_window: bool,
):
    """Slot mapping and window block list of one KV group for a prefill batch.

    rows: the group's block table [R, M] (CPU int32). row_idx[i]: request i's row, or None for
    a DP padding request (block 0, like the runner's dummy). Request i computes positions
    [ctx, ctx + q). Matches _form_prefill_batch (runner_window.py window gate on):
      slots[i, :q] = blk[pos // bs] * bs + pos % bs, rest -1     -> [target_bs, target_seq] int64
      win[i] = blk[:cdiv(ctx, bs)][-win_blocks:], right-padded -1 -> [target_bs, win_blocks] int32
    """
    n = len(q_lens)
    assert len(row_idx) == n == len(ctx_lens) and n <= target_bs, (len(row_idx), n, len(ctx_lens), target_bs)
    assert max(q_lens, default=0) <= target_seq, (q_lens, target_seq)
    slots = torch.full((target_bs, target_seq), -1, dtype=torch.int64)
    win = torch.full((target_bs, win_blocks), -1, dtype=torch.int32) if with_window else None
    for i, (r, c, q) in enumerate(zip(row_idx, ctx_lens, q_lens, strict=False)):
        nb = cdiv(c + q, block_size)
        if r is None:
            blk = torch.zeros(nb, dtype=torch.int64)
        else:
            assert nb <= rows.shape[1], (nb, rows.shape)
            blk = rows[r, :nb].to(torch.int64)
        pos = torch.arange(c, c + q, dtype=torch.int64)
        slots[i, :q] = blk[pos // block_size] * block_size + pos % block_size
        if win is not None:
            w = blk[: cdiv(c, block_size)][-win_blocks:]
            win[i, : w.numel()] = w.to(torch.int32)
    return slots, win


def decode_group_tensors(
    table: torch.Tensor,
    num_decodes: int,
    index: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_blocks,
    block_size: int,
    win_blocks: int,
):
    """Slot mapping and window block tables of one KV group for a decode batch.

    table: the group's block table; rows 0..num_decodes-1 are the decodes (CPU int32).
    index: [num_decodes, 1] int64 positions. slot_mapping: the base [padded_bs, 1] int64 slots
    (padding rows hold the runner's dummy slots and are kept). num_blocks[i] = cdiv(ctx_i + 1, bs).
    Matches _create_decode_input_data: window tables = block_table[-win_blocks:] per request.
    """
    assert index.shape == (num_decodes, 1) and slot_mapping.shape[1] == 1, (index.shape, slot_mapping.shape)
    assert len(num_blocks) == num_decodes <= slot_mapping.shape[0]
    sm = slot_mapping.clone()
    if num_decodes:
        bn = torch.gather(table[:num_decodes], 1, index // block_size).to(torch.int64)
        sm[:num_decodes] = bn * block_size + index % block_size
    tables = [table[i, max(0, n - win_blocks) : n].tolist() for i, n in enumerate(num_blocks)]
    return sm, tables


def window_usage(tables, slots, block_size: int, full_len: int):
    """block_usage per decode window table, exactly as get_habana_paged_attn_buffers computes it
    after patches/runner_window.py (head_mask_full_len = full_len, drop_masked_head): a full-length list whose
    last block is full drops its oldest block; otherwise the oldest block of a full-length list stores -lbu
    (slots [lbu, bs) valid). Returns (tables, usage). For tests and the CPU harness."""
    lbu = [s[0] % block_size + 1 for s in slots]
    tables = [
        bt[1:] if (len(bt) == full_len and len(bt) > 1 and i < len(lbu) and lbu[i] == block_size) else bt
        for i, bt in enumerate(tables)
    ]
    usage = [[block_size] * (len(t) - 1) + [u] for t, u in zip(tables, lbu, strict=False) if t]
    usage = [([-u[-1]] + u[1:]) if len(u) == full_len and len(u) > 1 else u for u in usage]
    return tables, usage


def _h2d(t: torch.Tensor, device) -> torch.Tensor:
    if torch.device(device).type == "cpu":
        return t
    from vllm_gaudi.utils import async_h2d_copy

    return async_h2d_copy(t, device=device)


# --------------------------------------------------------------------------
# Model side: per-layer metadata views
# --------------------------------------------------------------------------
def layer_views(base):
    """One metadata view per KV group id (index = gid), or None when groups are not active or the
    metadata carries no per-group fields (e.g. a profile run). The full group's entry is the base
    metadata itself; a sliding group's entry is the base with that group's slot mapping and window
    block list. Every view clears the two group fields, so each layer sees today's structure."""
    if not _ACTIVE["on"] or base is None or isinstance(base, dict):
        return None
    sm = getattr(base, "kv_group_slot_mapping", None)
    if sm is None:
        return None
    wl = getattr(base, "kv_group_window_block_list", None)
    from vllm_gaudi.v1.worker.hpu_model_runner import custom_tuple_replace

    views = []
    for g in range(len(sm)):
        if sm[g] is None:
            views.append(base)
        else:
            views.append(
                custom_tuple_replace(
                    base,
                    "TrimmedAttentionMetadata",
                    slot_mapping=sm[g],
                    window_block_list=None if wl is None else wl[g],
                    kv_group_slot_mapping=None,
                    kv_group_window_block_list=None,
                )
            )
    return views


def set_active(on: bool, ga_gid=None) -> None:
    _ACTIVE["on"] = bool(on)
    _ACTIVE["ga_gid"] = ga_gid


# --------------------------------------------------------------------------
# Runner state and hooks
# --------------------------------------------------------------------------
def _is_lazy() -> bool:
    try:
        import habana_frameworks.torch as htorch

        return bool(htorch.utils.internal.is_lazy())
    except Exception:
        return True


def unsupported(runner) -> list[str]:
    """Runner features that would bypass or break per-group KV."""
    vc = runner.vllm_config
    bad = []
    if runner.speculative_config is not None:
        bad.append("speculative decoding (the drafter reads block_table[0])")
    if runner.use_merged_prefill:
        bad.append("merged prefill")
    if runner.use_contiguous_pa:
        bad.append("contiguous PA / KV defragmenter (on when prefix caching is off; set VLLM_CONTIGUOUS_PA=false)")
    if vc.kv_transfer_config is not None:
        bad.append("KV transfer / KV connectors")
    if runner.num_mamba_like_layers:
        bad.append("Mamba/GDN layers")
    if vc.parallel_config.pipeline_parallel_size > 1:
        bad.append("pipeline parallelism")
    if vc.scheduler_config.disable_hybrid_kv_cache_manager:
        bad.append("--disable-hybrid-kv-cache-manager (the hybrid KV cache manager builds the groups)")
    if runner.model_config.use_mla:
        bad.append("MLA")
    if not _is_lazy():
        bad.append(
            "torch.compile mode (PT_HPU_LAZY_MODE=0): the per-layer metadata swap is validated in lazy mode only"
        )
    return bad


class SwaKvGroups:
    """Runner-side state of window-sized KV groups. One per HPUModelRunner."""

    def __init__(self, window: int, block_size: int):
        if window % block_size:
            raise ValueError(f"{ENV}: window {window} is not a multiple of block size {block_size}")
        self.window = window
        self.block_size = block_size
        self.win_blocks = window // block_size + 1  # = the runner's sliding_block_size (+1 for muse_spark)
        self.ga_gid = None
        self.sw_gids: tuple = ()
        self.n_groups = 1
        self.layer_gid: dict = {}
        self.layer_slot: dict = {}
        self.layer_spec: dict = {}
        self.n_slots = 0
        self.kv_cache_config = None
        self.bytes_allocated = 0
        self._specs_logged = False

    # get_kv_cache_spec ---------------------------------------------------
    def convert_specs(self, runner, kv_cache_spec: dict) -> dict:
        fwd = runner.vllm_config.compilation_config.static_forward_context
        windows = {name: getattr(fwd.get(name), "sliding_window", None) for name in kv_cache_spec}
        out = convert_spec_dict(kv_cache_spec, windows)
        got = {w for w in windows.values() if w is not None}
        if got != {self.window}:
            raise ValueError(f"{ENV}: layer windows {got} != model sliding window {self.window}")
        if not self._specs_logged:
            self._specs_logged = True
            n_sw = sum(w is not None for w in windows.values())
            page = next(iter(out.values())).page_size_bytes
            logger.warning(
                "muse_spark swa_kv: %d sliding-window layers (window %d) -> SlidingWindowSpec, %d full "
                "layers stay FullAttentionSpec, page %d B",
                n_sw,
                self.window,
                len(out) - n_sw,
                page,
            )
        return out

    # initialize_kv_cache, after initialize_attn_backend -------------------
    def on_init_kv_cache(self, runner, kv_cache_config) -> None:
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        groups = kv_cache_config.kv_cache_groups
        kinds = [type(g.kv_cache_spec) for g in groups]
        ga = [i for i, k in enumerate(kinds) if k is FullAttentionSpec]
        sw = [i for i, k in enumerate(kinds) if k is SlidingWindowSpec]
        if len(groups) < 2 or len(ga) != 1 or len(ga) + len(sw) != len(groups):
            raise RuntimeError(
                f"{ENV}: expected 1 full-attention group plus sliding-window groups, got "
                f"{[k.__name__ for k in kinds]} (is the hybrid KV cache manager on?)"
            )
        bad = unsupported(runner)
        if bad:
            raise RuntimeError(f"{ENV}: unsupported with window-sized KV groups: " + "; ".join(bad))
        specs = [g.kv_cache_spec for g in groups]
        s0 = specs[0]
        bs = runner.attn_block_size
        if bs != self.block_size:
            raise RuntimeError(f"{ENV}: kernel block size {bs} != {self.block_size}")
        for s in specs:
            if (s.block_size, s.page_size_bytes, s.num_kv_heads, s.head_size, s.dtype) != (
                bs,
                s0.page_size_bytes,
                s0.num_kv_heads,
                s0.head_size,
                s0.dtype,
            ):
                raise RuntimeError(f"{ENV}: KV groups differ in layout: {specs}")
        for g in sw:
            if specs[g].sliding_window != self.window:
                raise RuntimeError(f"{ENV}: group {g} window {specs[g].sliding_window} != {self.window}")

        layer_gid = {ln: g for g, grp in enumerate(groups) for ln in grp.layer_names}
        if len(layer_gid) != sum(len(grp.layer_names) for grp in groups):
            raise RuntimeError(f"{ENV}: a layer is in more than one KV cache group")
        # v30 core: one KVCacheTensor per group over ONE backing allocation; layer l of a group's
        # tensor sits at offset + l * layer_stride, so layer l of every group aliases slot l.
        page, nb = s0.page_size_bytes, kv_cache_config.num_blocks
        layer_slot = {}
        sizes = set()
        for t in kv_cache_config.kv_cache_tensors:
            names = list(t.layers)
            gids = {layer_gid[ln] for ln in names}
            if len(gids) != 1:
                raise RuntimeError(f"{ENV}: a KV tensor spans groups {sorted(gids)}")
            if t.block_stride != page or t.layer_stride != page * nb:
                raise RuntimeError(
                    f"{ENV}: unexpected KV layout: block_stride {t.block_stride}, layer_stride "
                    f"{t.layer_stride} (page {page}, {nb} blocks); layer-outermost expected"
                )
            base = t.offset // (page * nb)
            if t.offset % (page * nb):
                raise RuntimeError(f"{ENV}: KV tensor offset {t.offset} not a multiple of a layer region")
            for i, ln in enumerate(names):
                if ln in layer_slot:
                    raise RuntimeError(f"{ENV}: layer {ln} in two KV tensors")
                layer_slot[ln] = base + i
            sizes.add(t.size)
        if set(layer_slot) != set(layer_gid):
            raise RuntimeError(f"{ENV}: layers without a KV tensor: {sorted(set(layer_gid) - set(layer_slot))[:4]}")
        n_slots = max(layer_slot.values()) + 1
        for g in range(len(groups)):
            slots_g = [layer_slot[ln] for ln in groups[g].layer_names]
            if len(set(slots_g)) != len(slots_g):
                raise RuntimeError(f"{ENV}: two layers of group {g} share a KV slot")
        if sizes != {page * nb * n_slots}:
            raise RuntimeError(f"{ENV}: backing size {sizes} != {n_slots} slots x {nb} blocks x {page} B")

        self.ga_gid, self.sw_gids, self.n_groups = ga[0], tuple(sw), len(groups)
        self.layer_gid = layer_gid
        self.layer_slot = layer_slot
        self.n_slots = n_slots
        self.layer_spec = {ln: specs[g] for ln, g in layer_gid.items()}
        self.kv_cache_config = kv_cache_config
        runner.may_reinitialize_input_batch(kv_cache_config, [bs] * len(groups))
        logger.warning(
            "muse_spark swa_kv: %d KV groups (full gid %d: %d layers; sliding gids %s: %s layers), %d shared KV slots, "
            "%d blocks = %.2f GiB per rank (%d B/token for the full layers)",
            len(groups),
            self.ga_gid,
            len(groups[self.ga_gid].layer_names),
            list(self.sw_gids),
            [len(groups[g].layer_names) for g in self.sw_gids],
            n_slots,
            nb,
            n_slots * (nb + 1) * page / 2**30,
            n_slots * page // bs,
        )

    # initialize_kv_cache: replaces the per-layer allocation loop -----------
    def alloc_all(self, runner, kv_cache_config, kv_caches: dict) -> int:
        """One zeroed (k, v) pair per KV slot, bound to every layer at that slot. Returns
        num_blocks: the runner derives _PAD_BLOCK_ID and the bucketing block count from it."""
        nb = kv_cache_config.num_blocks
        s0 = next(iter(self.layer_spec.values()))
        page = s0.page_size_bytes
        shape = tuple(runner.attn_backend.get_kv_cache_shape(nb + 1, s0.block_size, s0.num_kv_heads, s0.head_size))
        elem = torch.empty((), dtype=s0.dtype).element_size()
        n_elem = 1
        for d in shape:
            n_elem *= d
        # k + v of one block must be exactly one page, or the core's num_blocks is wrong.
        if 2 * n_elem * elem != (nb + 1) * page:
            raise RuntimeError(f"{ENV}: KV shape {shape} x2 is {2 * n_elem * elem} B, expected {(nb + 1) * page} B")
        pairs = []
        for _ in range(self.n_slots):
            key_cache = torch.zeros(shape, dtype=s0.dtype, device=runner.device)
            value_cache = torch.zeros(shape, dtype=s0.dtype, device=runner.device)
            pairs.append((key_cache, value_cache, None, None))
            self.bytes_allocated += 2 * n_elem * elem
        for ln, slot in self.layer_slot.items():
            kv_caches[ln] = pairs[slot]
        return nb

    # initialize_kv_cache, after bind_kv_cache -----------------------------
    def after_bind(self, runner, kv_caches: dict) -> None:
        by_slot: dict = {}
        for ln, slot in self.layer_slot.items():
            by_slot.setdefault(slot, set()).add(id(kv_caches[ln]))
        if any(len(v) != 1 for v in by_slot.values()):
            raise RuntimeError(f"{ENV}: layers of one KV slot do not share one KV pair")
        ptrs = {kv_caches[next(ln for ln, s in self.layer_slot.items() if s == slot)][0].data_ptr() for slot in by_slot}
        if len(ptrs) != len(by_slot):
            raise RuntimeError(f"{ENV}: two KV slots got the same storage")
        fwd = runner.vllm_config.compilation_config.static_forward_context
        for ln, g in self.layer_gid.items():
            fwd[ln]._muse_kv_gid = g
        set_active(True, self.ga_gid)
        logger.warning(
            "muse_spark swa_kv: bound %d KV pairs (%.2f GiB) to %d layers; per-group metadata on",
            len(by_slot),
            self.bytes_allocated / 2**30,
            len(self.layer_gid),
        )

    # _form_prefill_batch, before return ----------------------------------
    def prefill(
        self, runner, req_ids, query_lens, context_lens, attn_metadata, target_bs: int, target_seq: int
    ) -> None:
        """query_lens / context_lens: copies taken at the top of _form_prefill_batch (its
        align_and_pad pads these lists in place to the bucket)."""
        req_ids = list(req_ids)
        n = len(req_ids)
        assert len(query_lens) == len(context_lens) == n, (len(query_lens), len(context_lens), n)
        q_lens = [int(q) for q in query_lens[:n]]
        ctx_lens = [int(c) for c in context_lens[:n]]
        index = runner.input_batch.req_id_to_index
        row_idx = []
        for r in req_ids:
            i = index.get(r)
            if i is None and r != DUMMY_REQ_ID:
                raise KeyError(f"{ENV}: prefill request {r} is not in the persistent batch")
            row_idx.append(i)
        assert tuple(attn_metadata.slot_mapping.shape) == (target_bs, target_seq), (
            tuple(attn_metadata.slot_mapping.shape),
            target_bs,
            target_seq,
        )
        base_win = attn_metadata.window_block_list
        with_window = base_win is not None
        if with_window:
            assert base_win.numel() == target_bs * self.win_blocks, (base_win.shape, target_bs, self.win_blocks)
        sms = [None] * self.n_groups
        wls = [None] * self.n_groups
        for g in self.sw_gids:
            rows = runner.input_batch.block_table[g].get_cpu_tensor()
            slots, win = prefill_group_tensors(
                rows, row_idx, ctx_lens, q_lens, target_bs, target_seq, self.block_size, self.win_blocks, with_window
            )
            sms[g] = _h2d(slots, runner.device)
            if with_window:
                wls[g] = _h2d(win.flatten(), runner.device)
        attn_metadata.kv_group_slot_mapping = tuple(sms)
        attn_metadata.kv_group_window_block_list = tuple(wls) if with_window else None

    # _create_decode_input_data, before return -----------------------------
    def decode(
        self,
        runner,
        attn_metadata,
        block_table_cpu_tensor,
        num_decodes: int,
        num_tokens: int,
        padded_batch_size: int,
        index,
        num_blocks,
        slot_mapping,
        decode_block_size: int,
    ) -> None:
        if num_tokens != 1:
            raise RuntimeError(f"{ENV}: multi-token decode (spec decode) is not supported")
        assert decode_block_size == self.block_size
        ga_table = runner.input_batch.block_table[self.ga_gid].get_cpu_tensor()
        # The DP padding batch passes its own all-zero table; use it for every group.
        dummy = block_table_cpu_tensor is not ga_table
        base_win = attn_metadata.window_block_list
        sm_list = slot_mapping.tolist() if base_win is not None else None
        sms = [None] * self.n_groups
        wls = [None] * self.n_groups
        for g in self.sw_gids:
            table = block_table_cpu_tensor if dummy else runner.input_batch.block_table[g].get_cpu_tensor()
            sm_g, tables = decode_group_tensors(
                table, num_decodes, index, slot_mapping, num_blocks, decode_block_size, self.win_blocks
            )
            sms[g] = _h2d(sm_g, runner.device)
            if base_win is not None:
                wl = runner.get_habana_paged_attn_buffers(
                    tables,
                    sm_list,
                    padded_batch_size * num_tokens,
                    block_size=decode_block_size,
                    force_non_contiguous=True,
                    head_mask_full_len=self.win_blocks,
                    drop_masked_head=True,
                )[0]
                assert tuple(wl.shape) == tuple(base_win.shape), (wl.shape, base_win.shape)
                wls[g] = _h2d(wl, runner.device)
        attn_metadata.kv_group_slot_mapping = tuple(sms)
        attn_metadata.kv_group_window_block_list = tuple(wls) if base_win is not None else None


def create(runner, runner_module=None):
    """End of HPUModelRunner.__init__ (only reached with MUSE_SPARK_SWA_KV=1)."""
    model_type = runner._get_model_type()
    if model_type != MODEL_TYPE:
        raise RuntimeError(f"{ENV}=1 is only for model_type {MODEL_TYPE!r}, this model is {model_type!r}")
    if not runner.interleaved_sliding_window or not runner.sliding_window:
        raise RuntimeError(f"{ENV}: the model reports no interleaved sliding window")
    bad = unsupported(runner)
    if bad:
        raise RuntimeError(f"{ENV}: unsupported with window-sized KV groups: " + "; ".join(bad))
    mod = runner_module if runner_module is not None else sys.modules.get(RUNNER_MODULE)
    trim = getattr(mod, "_SWG_TRIM_FIELDS", None)
    if trim is None:
        raise RuntimeError(f"{ENV}: {RUNNER_MODULE} is not patched (patches/swa_kv_hooks.py)")
    trim[:] = list(TRIM_FIELDS)
    from vllm_gaudi.v1.attention.backends.hpu_attn import HPUAttentionMetadataV1

    for f in TRIM_FIELDS:
        if not hasattr(HPUAttentionMetadataV1, f):
            setattr(HPUAttentionMetadataV1, f, None)
    swg = SwaKvGroups(int(runner.sliding_window), int(runner.attn_block_size))
    logger.warning(
        "muse_spark swa_kv: %s=1: window-sized KV groups for the sliding-window layers (window %d, block %d)",
        ENV,
        swg.window,
        swg.block_size,
    )
    return swg


__all__ = [
    "ENV",
    "requested",
    "active",
    "layer_views",
    "set_active",
    "SwaKvGroups",
    "create",
    "convert_spec_dict",
    "prefill_group_tensors",
    "decode_group_tensors",
    "window_usage",
    "to_sliding_window_spec",
]
