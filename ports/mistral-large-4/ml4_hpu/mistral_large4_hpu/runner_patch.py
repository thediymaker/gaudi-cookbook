"""HPUModelRunner hook: per-request addressing tensors for Mistral Large 4 (the DeepSeek-V4.1 port's runner_patch,
without engram, SWA rings, decoder replay and speculative layouts).

The HPU runner hands a model one attention-metadata object with flattened block lists. ML4's paged caches (one per
(layer, kind), paged.cache_layout) are addressed per request, so every prefill and decode batch also carries, as
tensors (never scalars inside graphs: see trim_attn_metadata's note on graph hashing):

  ml4_block_table     [B, W] int32   the request's physical blocks, -1 past its end (W: _bt_width)
  ml4_req_uid         [B] int32      never reused; -1 for padding rows
  ml4_block_epoch     [B, W] int32   allocation number of each table entry, -1 past its end (the tag owner)
  ml4_req_slot        [B] int32      request slot (0 .. max_num_seqs-1), -1 for padding: DSpark's per-request rings
                                     are addressed by it, as DSV's
  ml4_host_max_pos    int            the batch's largest position, known on the host: <= 2047 lets every indexer skip
                                     its scores (dense top-k), indexer.py
  ml4_verify_rows     int | None     a DSpark verify step's rows per request T (spec._install_verify_layout; the
                                     rows are flat, request b in rows b*T .. b*T+T-1), None on every other step

Request slots: one per running request, at most max_num_seqs. A request keeps its slot until it finishes; one that is
not scheduled in a step (preempted) can lose it to a newcomer, and a request that (re)enters gets a new uid, so a
slot's contents are never trusted across owners (the consumer checks uid, as DSV's rings).

Cache rows are tagged with their block's allocation number (paged.py). The scheduler reports, for each new or resumed
request, its prefix-cache hits (whole blocks, the first num_computed_tokens // block) and its block list; every other
block it hands out is a fresh allocation. Fresh blocks get a new number, hits keep theirs, so a request reads rows its
prefix-sharer wrote but never rows left by a block's previous owner. That holds with prefix caching off too: then
every block is fresh. DSpark's rings are per request: a hit's ring rows read as masked (prefix.py).

The runner fixes DSV41 needed on vllm-gaudi v0.30 are kept as they are: one sampled position per prompt row,
ascending-context prompt order (same-step prefix hits), int32-first logprob gathers, prompt logprobs in row slices,
and no attention-bias masks / metadata processing (this model's attention never reads them).
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

logger = logging.getLogger("vllm.mistral_large4_hpu")

FIELDS = ["ml4_block_table", "ml4_req_uid", "ml4_block_epoch", "ml4_req_slot", "ml4_host_max_pos", "ml4_verify_rows"]
# Block-table width: the smallest power of two >= the blocks the batch's highest position needs, at least
# ML4_BT_MIN_BLOCKS, at most max_model_len // block. The unabsorbed attention's key range (and its fp32 transient)
# then follows the batch's real context; each width is its own graph. 0 = always full.
BT_MIN = int(os.environ.get("ML4_BT_MIN_BLOCKS", "16"))
# ML4_BT_PROMPT_HALF=1 (default; control key bt_prompt_half): prompt chunks (more than 8 rows per request) may also take
# 3/4 of that power of two (24, 48, 96, ... blocks): their indexer and attention cost follows the width, and a chunk
# just past a power of two ran at twice its context (a 34K prompt's last chunk at 64K keys).
# Decode widths stay powers of two (their HPU graphs are keyed by width).
BT_PROMPT_HALF = os.environ.get("ML4_BT_PROMPT_HALF", "1") == "1"
FAULT_MULTI_LOGITS = os.environ.get("ML4_FAULT_MULTI_LOGITS", "0") == "1"
FAULT_ROW_ORDER = os.environ.get("ML4_FAULT_ROW_ORDER", "0") == "1"
# Tests only (the control): prefix-cache hit blocks get a new allocation number like fresh
# ones, so their rows (written under the old number) read as foreign: ML4_TAG_CHECK=count must count every hit row.
FAULT_HIT_FRESH = os.environ.get("ML4_FAULT_HIT_FRESH", "0") == "1"
NEED_ATTN_METADATA = os.environ.get("ML4_ATTN_METADATA", "0") == "1"
PLP_ROWS = int(os.environ.get("ML4_PLP_ROWS", "512"))
# Prompt query buckets below the runner's smallest (its block size, 128): a prompt of <= 64 tokens then runs as 64 rows,
# which the MoE's mxk decode kernels take in one call. "" = the runner's buckets only.
PROMPT_QUERY_EXTRA = [int(v) for v in os.environ.get("ML4_PROMPT_QUERY_EXTRA", "64").split(",") if v.strip()]
_state = {
    "n_blocks": None,
    "block_size": None,
    "uids": {},
    "next_uid": 1,
    "next_epoch": 1,
    "installed": False,
    "n_slots": 0,
    "slots": {},
    "sched": set(),
}


def configure(max_model_len: int, block_size: int, n_slots: int = 0) -> None:
    _state["n_blocks"] = -(-max_model_len // block_size)
    _state["block_size"] = block_size
    _state["n_slots"] = n_slots


def _num_blocks(runner) -> int:
    kc = getattr(runner, "kv_cache_config", None)
    return int(getattr(kc, "num_blocks", 0) or runner.vllm_config.cache_config.num_gpu_blocks or 0)


def _epochs(runner, need: int = 0) -> np.ndarray:
    """Host table of block allocation numbers, one per physical block id; only ever grows (DSV)."""
    e = _state.get("epoch")
    n = max(need, _num_blocks(runner) + 2)
    if e is None or e.shape[0] < n:
        grown = np.zeros(n, dtype=np.int32)
        if e is not None:
            grown[: e.shape[0]] = e
        e = _state["epoch"] = grown
    return e


def _fresh(runner, blocks, start: int = 0) -> None:
    blocks = [int(b) for b in list(blocks)[start:]]
    e = _epochs(runner, max(blocks, default=-1) + 1)
    for b in blocks:
        if b >= 0:
            e[b] = _state["next_epoch"]
            _state["next_epoch"] = _state["next_epoch"] % 2_000_000_000 + 1


def _uid(req_id: str) -> int:
    u = _state["uids"].get(req_id)
    if u is None:
        u = _state["next_uid"]
        _state["next_uid"] += 1
        _state["uids"][req_id] = u
    return u


def _slot(rid: str) -> int:
    slots = _state["slots"]
    if rid in slots:
        return slots[rid]
    used = set(slots.values())
    free = [i for i in range(_state["n_slots"]) if i not in used]
    if not free:
        victim = next((r for r in slots if r not in _state["sched"]), None)
        assert victim is not None, "ml4: more scheduled requests than request slots"
        free = [slots.pop(victim)]
    slots[rid] = free[0]
    return free[0]


def _bt_width(positions: np.ndarray) -> int:
    nb = _state["n_blocks"]
    if BT_MIN <= 0:
        return nb
    need = (int(positions.max()) + 1 + _state["block_size"] - 1) // _state["block_size"] if positions.size else 1
    w = BT_MIN
    while w < need:
        w *= 2
    if positions.ndim == 2 and positions.shape[1] > 8 and w >= 2 * BT_MIN and need <= w * 3 // 4:
        from .control import get as ctl_get

        if ctl_get("bt_prompt_half", BT_PROMPT_HALF):
            w = w * 3 // 4
    return min(w, nb)


def _attach(runner, md, req_ids, blocks, positions: np.ndarray) -> None:
    B, S = positions.shape
    nb = _bt_width(positions)
    bt = np.full((B, nb), -1, dtype=np.int32)
    uid = np.full((B,), -1, dtype=np.int32)
    slot = np.full((B,), -1, dtype=np.int32)
    for b, rid in enumerate(req_ids):
        if rid is None or rid == "-1":  # padding rows and the runner's dummy batches
            continue
        blk = list(blocks[b])[:nb]
        bt[b, : len(blk)] = blk
        uid[b] = _uid(rid)
        if _state["n_slots"]:
            slot[b] = _slot(rid)
    dev = runner.device
    md.ml4_block_table = torch.from_numpy(bt).to(dev, non_blocking=True)
    md.ml4_req_uid = torch.from_numpy(uid).to(dev, non_blocking=True)
    e = _epochs(runner, int(bt.max(initial=-1)) + 1)
    ep = np.where(bt >= 0, e[np.maximum(bt, 0)], -1).astype(np.int32)
    md.ml4_block_epoch = torch.from_numpy(ep).to(dev, non_blocking=True)
    md.ml4_req_slot = torch.from_numpy(slot).to(dev, non_blocking=True)
    real = positions[uid >= 0] if (uid >= 0).any() else positions
    md.ml4_host_max_pos = int(real.max()) if real.size else 0


def install() -> None:
    if _state["installed"]:
        return
    from vllm_gaudi.v1.worker import hpu_model_runner as hmr

    runner_cls = hmr.HPUModelRunner

    # 1. trim_attn_metadata keeps a fixed field list; extend the CACHED namedtuple type with ours (DSV41 step 1).
    import collections

    orig_trim = hmr.trim_attn_metadata
    tname = "TrimmedAttentionMetadata"

    def _extend_cached_type():
        entry = hmr._TYPE_CACHE[tname]
        if set(FIELDS) <= set(entry["fields"]):
            return entry
        fields = [f for f in entry["type"]._fields if f not in FIELDS] + FIELDS
        entry["type"] = collections.namedtuple(tname, fields, defaults=[None] * len(FIELDS))
        entry["fields"] = set(fields)
        return entry

    def trim_attn_metadata(metadata):
        base = orig_trim(metadata)
        if base is None:
            return None
        entry = _extend_cached_type()
        if not isinstance(base, entry["type"]):
            base = entry["type"](**base._asdict())
        return base._replace(**{f: getattr(metadata, f, None) for f in FIELDS})

    hmr.trim_attn_metadata = trim_attn_metadata

    # 2. prefill batches: one sampled position per request (DSV41: vllm-gaudi's multi-position sampling duplicated
    #    output after preemption), positions rebuilt on the host, each request's full block list.
    orig_form = runner_cls._form_prefill_batch

    def _form_prefill_batch(self, contents):
        if getattr(contents, "logits_positions", None) and not FAULT_MULTI_LOGITS:
            contents.logits_positions = [lp[-1:] for lp in contents.logits_positions]
        spans = [(int(c), len(t)) for c, t in zip(contents.context_lens, contents.token_ids)]
        out = orig_form(self, contents)
        if not out.attn_metadata:
            return out
        md = out.attn_metadata[0]
        shape = tuple(out.position_ids[0].shape)
        if len(shape) == 2 and shape[0] >= len(spans) and all(n <= shape[1] for _, n in spans):
            pos = np.full(shape, -1, dtype=np.int64)
            for b, (c, n) in enumerate(spans):
                pos[b, :n] = np.arange(c, c + n)
        else:
            p = out.position_ids[0]
            pos = np.asarray(p.cpu() if torch.is_tensor(p) else p, dtype=np.int64)
        rids = list(contents.req_ids) + [None] * (pos.shape[0] - len(contents.req_ids))
        blocks = list(contents.blocks) + [[]] * (pos.shape[0] - len(contents.blocks))
        _attach(self, md, rids, blocks, pos)
        return out

    runner_cls._form_prefill_batch = _form_prefill_batch

    # 3. decode batches: rows 0..num_decodes-1 of the input batch, one token each.
    orig_decode = runner_cls._create_decode_input_data

    def _create_decode_input_data(
        self, num_decodes, num_scheduled_tokens, context_lens, block_table_cpu_tensor, scheduler_output=None
    ):
        out = orig_decode(
            self, num_decodes, num_scheduled_tokens, context_lens, block_table_cpu_tensor, scheduler_output
        )
        B = out.token_ids.shape[0]
        assert out.token_ids.shape[1] == 1, "ml4: decode rows carry one token each (no speculative layout yet)"
        real = list(self.input_batch.req_ids[:num_decodes]) if scheduler_output is not None else []
        bt = block_table_cpu_tensor[:num_decodes].tolist()
        blocks = [[b for b in row if b >= 0] for row in bt]
        pos = np.full((B, 1), -1, dtype=np.int64)
        pos[:num_decodes, 0] = np.asarray(context_lens[:num_decodes], dtype=np.int64)
        rids = real + [None] * (B - len(real))
        blocks = blocks + [[]] * (B - num_decodes)
        _attach(self, out.attn_metadata, rids, blocks, pos)
        return out

    runner_cls._create_decode_input_data = _create_decode_input_data

    # 4. a fresh uid when a request (re)enters; fresh allocation numbers for blocks past the prefix-cache hits.
    orig_update = runner_cls._update_states

    def _update_states(self, scheduler_output):
        bs = _state["block_size"]
        _state["sched"] = set(scheduler_output.num_scheduled_tokens)
        for r in scheduler_output.scheduled_new_reqs:
            _state["uids"].pop(r.req_id, None)
            _state["slots"].pop(r.req_id, None)
            _fresh(self, r.block_ids[0], 0 if FAULT_HIT_FRESH else r.num_computed_tokens // bs)
        cached = scheduler_output.scheduled_cached_reqs
        resumed = getattr(cached, "resumed_req_ids", ()) or ()
        for rid in resumed:
            _state["uids"].pop(rid, None)
            _state["slots"].pop(rid, None)
        for i, rid in enumerate(cached.req_ids):
            ids = cached.new_block_ids[i]
            if not ids:
                continue
            hit = cached.num_computed_tokens[i] // bs if rid in resumed and not FAULT_HIT_FRESH else 0
            _fresh(self, ids[0], hit)
        for rid in scheduler_output.finished_req_ids:
            _state["uids"].pop(rid, None)
            _state["slots"].pop(rid, None)
        return orig_update(self, scheduler_output)

    runner_cls._update_states = _update_states

    # 4b. prompts in ascending context order (a same-step prefix hit must run after its writer; DSV41 4b).
    orig_decodes_first = runner_cls._ensure_decodes_first

    def _ensure_decodes_first(self, scheduler_output):
        orig_decodes_first(self, scheduler_output)
        if FAULT_ROW_ORDER:
            return
        b = self.input_batch
        n = b.num_reqs
        first = next((i for i in range(n) if self._is_prompt(i, scheduler_output)), n)
        for i in range(first, n):
            j = min(range(i, n), key=lambda k: (int(b.num_computed_tokens_cpu[k]), k))
            if j != i:
                b.swap_states(i, j)

    runner_cls._ensure_decodes_first = _ensure_decodes_first

    # 5. logprobs on HPU lazy: cast ids to int32 BEFORE the cat (vllm-gaudi PR #1823); host tensors for prompts.
    from vllm.v1.outputs import LogprobsTensors
    import vllm.v1.sample.sampler as sampler_mod

    def _gather_ids_first_int32(logprobs, k, token_ids):
        topk_logprobs, topk_indices = torch.topk(logprobs, k, dim=-1)
        tok = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, tok)
        ids = torch.cat((tok.to(torch.int32), topk_indices.to(torch.int32)), dim=1)
        return ids, torch.cat((token_logprobs, topk_logprobs), dim=1), token_logprobs

    def _hpu_gather_logprobs(logprobs, num_logprobs, token_ids):
        ids, lps, token_logprobs = _gather_ids_first_int32(logprobs, num_logprobs, token_ids)
        ranks = sampler_mod.batched_count_greater_than(logprobs, token_logprobs)
        return LogprobsTensors(ids, lps, ranks)

    sampler_mod.Sampler.gather_logprobs = staticmethod(_hpu_gather_logprobs)

    def _gather_prompt_logprobs_hpu(self, logprobs, num_prompt_logprobs, token_ids):
        ids, lps, token_logprobs = _gather_ids_first_int32(logprobs, num_prompt_logprobs, token_ids)
        ranks = (logprobs >= token_logprobs).sum(-1)
        return LogprobsTensors(ids.cpu(), lps.cpu(), ranks.cpu())

    runner_cls._gather_prompt_logprobs_hpu = _gather_prompt_logprobs_hpu

    # 5b. prompt logprobs in slices of PLP_ROWS rows ([chunk, 204800] fp32 logits at once ran DSV out of memory).
    def _get_prompt_logprobs_dict(self, prefill_hidden_states, scheduler_output):
        want = self.input_batch.num_prompt_logprobs
        if not want:
            return {}
        in_progress = self.input_batch.in_progress_prompt_logprobs_cpu
        out, done = {}, []
        for req_id, k in want.items():
            if req_id not in prefill_hidden_states:
                continue
            request = self.requests[req_id]
            if request.prompt_token_ids is None:
                continue
            n_prompt = len(request.prompt_token_ids)
            acc = in_progress.get(req_id)
            if not acc:
                acc = LogprobsTensors.empty_cpu(n_prompt - 1, k + 1)
                in_progress[req_id] = acc
            start = request.num_computed_tokens
            n_tok = scheduler_output.num_scheduled_tokens[req_id]
            n = n_tok if n_tok <= n_prompt - start - 1 else n_prompt - start - 1
            if n_tok > n_prompt - start - 1:
                done.append(req_id)
                out[req_id] = acc
            if n <= 0:
                continue
            hs = prefill_hidden_states[req_id]
            hs = hs.squeeze(0) if hs.dim() == 3 else hs
            tgt = torch.tensor(request.prompt_token_ids[start + 1 : start + 1 + n], dtype=torch.int64)
            for a in range(0, n, PLP_ROWS):
                b = min(a + PLP_ROWS, n)
                logprobs = self.sampler.compute_logprobs(self.model.compute_logits(hs[a:b]))
                g = self._gather_prompt_logprobs_hpu(logprobs, k, tgt[a:b].to(self.device))
                del logprobs
                rows = slice(start + a, start + b)
                acc.logprob_token_ids[rows].copy_(g.logprob_token_ids)
                acc.logprobs[rows].copy_(g.logprobs)
                acc.selected_token_ranks[rows].copy_(g.selected_token_ranks)
        for req_id in done:
            del want[req_id]
            in_progress.pop(req_id, None)
        return out

    runner_cls._get_prompt_logprobs_dict = _get_prompt_logprobs_dict

    # 6. no attention-bias masks or metadata processing: ML4's attention never reads them, and the prompt mask alone
    #    is [B, 1, q, ctx + q] bf16 (DSV41: within 0.7 GB of the card's limit at 64K).
    def _no_attn_bias(self, attn_metadata, *args, **kwargs):
        return attn_metadata

    runner_cls._set_attn_bias = _no_attn_bias
    if hasattr(runner_cls, "set_attn_bias"):
        runner_cls.set_attn_bias = _no_attn_bias
    proc_cls = getattr(hmr, "HPUAttentionMetadataProcessor", None)
    if proc_cls is not None and not NEED_ATTN_METADATA:

        def _no_metadata(self, attn_metadata, *args, **kwargs):
            return attn_metadata

        proc_cls.process_metadata = _no_metadata
    # 7. no value caches: vllm-gaudi allocates a v tensor the size of k for every FullAttentionSpec unless
    #    model_config.use_mla (false here: ML4's MLA lives in its own modules). ML4KVCache binds k only, so the v half
    #    of the KV pool (6.44 GiB per card at 640 blocks, measured) is dead weight. It is never
    #    allocated: model_config answers use_mla for the allocation call only (initialize_kv_cache reads it there and
    #    nowhere else). Freeing it after the fact left the start-up peak at k + v, so KV_BLOCKS 2100 (12 GiB of k)
    #    failed to start on the 21 GiB a card has after the weights.
    orig_init_kv = runner_cls.initialize_kv_cache

    def initialize_kv_cache(self, kv_cache_config):
        mc = self.model_config
        cls = type(mc)
        mc.__class__ = type(cls.__name__, (cls,), {"use_mla": property(lambda _self: True)})
        has_kvt = hmr.has_kv_transfer_group
        hmr.has_kv_transfer_group = lambda: False  # 9: the connector gets the caches after 7b, below
        try:
            orig_init_kv(self, kv_cache_config)
        finally:
            mc.__class__ = cls
            hmr.has_kv_transfer_group = has_kvt
        freed = 0
        for i, t in enumerate(self.kv_caches):
            if isinstance(t, tuple) and len(t) > 1 and torch.is_tensor(t[1]):
                freed += t[1].numel() * t[1].element_size()
                self.kv_caches[i] = (t[0], None) + tuple(t[2:])
        # 7b. each ML4 cache as its own 2-D base tensor. vllm-gaudi allocates [rows, 1, W] and ML4KVCache binds a
        #     reshaped view; a decode HPU graph (graphs.py) that reads a cache last written by ANOTHER graph of the
        #     same step (the KV-shared layers read layer 35's) fails at capture on such a view ("Empty tensor optional
        # .. Strided Params", the decode-graph harness). One layer's cache extra at a time.
        n_rebased = 0
        ctx = self.vllm_config.compilation_config.static_forward_context
        for mod in ctx.values():
            t = getattr(mod, "kv_cache", None)
            if type(mod).__name__ != "ML4KVCache" or not torch.is_tensor(t) or t._base is None:
                continue
            ptr = t.data_ptr()
            new = torch.zeros(t.shape, dtype=t.dtype, device=t.device)
            mod.kv_cache = new
            for i, e in enumerate(self.kv_caches):
                if isinstance(e, tuple) and torch.is_tensor(e[0]) and e[0].data_ptr() == ptr:
                    self.kv_caches[i] = (new,) + tuple(e[1:])
                elif torch.is_tensor(e) and e.data_ptr() == ptr:
                    self.kv_caches[i] = new
            del t
            n_rebased += 1
        import gc

        gc.collect()
        logger.warning(
            "ml4: released unused value caches, %.2f GiB per card; %d caches rebased to 2-D tensors",
            freed / 2**30,
            n_rebased,
        )
        _state["runner"] = self
        if has_kvt():
            from . import offload  # 9. KV offloading (offload.py), registered after 7b's 2-D base tensors

            offload.register(self, _state["block_size"])

    runner_cls.initialize_kv_cache = initialize_kv_cache
    from . import offload

    offload.install(lambda: _state.get("runner"), _fresh, _epochs, lambda: _state["block_size"])

    # 8. extra small prompt query buckets (PROMPT_QUERY_EXTRA): each copies the smallest query bucket's batch / context
    #    entries.
    from vllm_gaudi.extension.bucketing.common import HPUBucketingManager

    orig_gen = HPUBucketingManager.generate_prompt_buckets

    def generate_prompt_buckets(self):
        orig_gen(self)
        if PROMPT_QUERY_EXTRA and self.prompt_buckets:
            base = min(q for _, q, _ in self.prompt_buckets)
            add = {(b, q, c) for b, qq, c in self.prompt_buckets if qq == base for q in PROMPT_QUERY_EXTRA if q < base}
            self.prompt_buckets = sorted(set(self.prompt_buckets) | add)
            logger.warning(
                "ml4: prompt query buckets %s added below %d (%d prompt buckets)",
                sorted({q for _, q, _ in add}),
                base,
                len(self.prompt_buckets),
            )

    HPUBucketingManager.generate_prompt_buckets = generate_prompt_buckets
    from . import spec

    spec.install_runner(runner_cls)  # ML4_SPEC=dspark: DSpark drafts, flat verify rows, verify sampler
    from . import asyncspec

    asyncspec.install_resume_fix(runner_cls)  # ML4_RESUME_FIX=1, sync path only (outermost _update_states wrapper)
    _state["installed"] = True
    logger.warning("ml4: HPU runner hook installed (block tables, request uids, block epochs, request slots)")


def batch_ctx_from_metadata(md, positions: torch.Tensor, block_size: int):
    """The model side: paged.BatchCtx from the trimmed metadata of this forward."""
    from .paged import BatchCtx

    uid = md.ml4_req_uid
    pos = positions.long()
    if pos.dim() == 1:
        pos = pos.view(uid.shape[0], -1)
    pos = torch.where(uid.view(-1, 1) >= 0, pos, torch.full_like(pos, -1))
    slot = getattr(md, "ml4_req_slot", None)
    return BatchCtx(
        positions=pos,
        block_table=md.ml4_block_table.long(),
        req_uid=uid,
        block_size=block_size,
        block_epoch=getattr(md, "ml4_block_epoch", None),
        host_max_pos=getattr(md, "ml4_host_max_pos", None),
        req_slot=slot.long() if slot is not None else None,
    )


__all__ = ["install", "configure", "batch_ctx_from_metadata", "FIELDS"]
