# SPDX-License-Identifier: Apache-2.0
"""HPUModelRunner hook: per-request addressing tensors for DeepSeek-V4.1.

The HPU runner hands a model one attention-metadata object with flattened block lists. V4.1's paged caches are
addressed per request, so every prefill and decode batch also carries, as tensors (never Python scalars, which
would become part of an HPU graph's key):

  dsv41_block_table   [B, W] int32                         the request's physical blocks, -1 past its end
  dsv41_req_uid       [B] int32                            -1 for padding rows
  dsv41_engram_rows   [B, S, n_engram_layers, n_cols] int32  hash rows, computed on the host
  dsv41_block_epoch   [B, W] int32                         allocation number of each table entry, -1 past its end
  dsv41_replay        int (prompt batches only)            decoder replay length (replay.py); 0 = full stack, also
                                                           for any batch with a prompt_logprobs request

W, the block-table width, is the smallest power of two >= the blocks the batch's highest position needs, at least
BT_MIN, at most max_model_len // block. Everything sized by the table (the indexer's entry range, the candidate
stage) then follows the batch's real context instead of max_model_len (at 128K the indexer had been scoring 131,072
entries per query whatever the prompt length); each width is its own decode graph.

Cache rows are tagged with their block's allocation number (paged.py). The scheduler reports, for each new or
resumed request, its prefix-cache hits (whole blocks, the first num_computed_tokens // block) and its block list;
every other block it hands out is a fresh allocation. Fresh blocks get a new number, hits keep theirs, so a request
reads rows its prefix sharer wrote but never rows left by a block's previous owner. That holds with prefix caching
off too: then every block is fresh.

Engram hashing needs each request's token history up to the token being processed, so async scheduling must be off
(the host then has every sampled token).

Generic runner bugs this model depends on are fixed by the cookbook's fixes (the recipe's required_fixes), not
here: prompt-order (a prefix-cache reader can run before its writer in the same step), resumed-prefill-logits (a
request resumed after preemption re-samples its generated tail), gather-logprobs-int32 and prompt-logprobs
(including row-chunked prompt logits, which a 2304-block KV pool needs).
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger("vllm.dsv41_hpu")  # under "vllm", so vLLM's handlers print it

FIELDS = ["dsv41_block_table", "dsv41_req_uid", "dsv41_engram_rows", "dsv41_block_epoch", "dsv41_replay"]
BT_MIN = 64
_state = {"hasher": None, "n_blocks": None, "uids": {}, "next_uid": 1, "next_epoch": 1, "installed": False}


def configure(hasher, max_model_len: int, block_size: int) -> None:
    _state["hasher"] = hasher
    _state["n_blocks"] = -(-max_model_len // block_size)
    _state["block_size"] = block_size


def _num_blocks(runner) -> int:
    """KV blocks of this worker. kv_cache_config is the runner's own; cache_config.num_gpu_blocks is set only in the
    engine process, and in tensor-parallel workers it stays None."""
    kc = getattr(runner, "kv_cache_config", None)
    return int(getattr(kc, "num_blocks", 0) or runner.vllm_config.cache_config.num_gpu_blocks or 0)


def _epochs(runner, need: int = 0) -> np.ndarray:
    """Host table of block allocation numbers, one per physical block id (vllm-gaudi also uses id num_blocks as its
    pad block). It only grows and keeps every number: a table smaller than a block id would give every block past it
    one shared number, so handing out one block would invalidate every cached row."""
    e = _state.get("epoch")
    n = max(need, _num_blocks(runner) + 2)
    if e is None or e.shape[0] < n:
        grown = np.zeros(n, dtype=np.int32)
        if e is not None:
            grown[: e.shape[0]] = e
        e = _state["epoch"] = grown
    return e


def _fresh(runner, blocks, start: int = 0) -> None:
    """Blocks handed out fresh (not prefix-cache hits): a new allocation number each."""
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


def _engram_rows(runner, req_ids, positions: np.ndarray, S: int) -> torch.Tensor:
    """positions [B, S] with -1 padding -> [B, S, Lg, n_cols] int32 (0 where padded)."""
    h = _state["hasher"]
    B = positions.shape[0]
    out = np.zeros((B, S, len(h.layer_ids), h.n_cols), dtype=np.int32)
    for b, rid in enumerate(req_ids):
        idx = runner.input_batch.req_id_to_index.get(rid) if rid is not None else None
        if idx is None:
            continue
        pos = positions[b]
        ok = pos >= 0
        if not ok.any():
            continue
        end = int(pos[ok].max()) + 1
        hist = runner.input_batch.token_ids_cpu[idx, :end]
        out[b, ok] = h.hash_rows(hist, pos[ok])
    return torch.from_numpy(out)


def _bt_width(positions: np.ndarray) -> int:
    """From positions, not block lists: the runner's decode rows may carry zero padding."""
    nb = _state["n_blocks"]
    need = (int(positions.max()) + 1 + _state["block_size"] - 1) // _state["block_size"] if positions.size else 1
    w = BT_MIN
    while w < need:
        w *= 2
    return min(w, nb)


def _attach(runner, md, req_ids, blocks, positions: np.ndarray) -> None:
    B, S = positions.shape
    nb = _bt_width(positions)
    bt = np.full((B, nb), -1, dtype=np.int32)
    uid = np.full((B,), -1, dtype=np.int32)
    for b, rid in enumerate(req_ids):
        if rid is None or rid == "-1":  # padding rows and the runner's dummy batches
            continue
        blk = list(blocks[b])[:nb]
        bt[b, : len(blk)] = blk
        uid[b] = _uid(rid)
    dev = runner.device
    md.dsv41_block_table = torch.from_numpy(bt).to(dev, non_blocking=True)
    md.dsv41_req_uid = torch.from_numpy(uid).to(dev, non_blocking=True)
    md.dsv41_engram_rows = _engram_rows(runner, req_ids, positions, S).to(dev, non_blocking=True)
    e = _epochs(runner, int(bt.max(initial=-1)) + 1)
    ep = np.where(bt >= 0, e[np.maximum(bt, 0)], -1).astype(np.int32)
    md.dsv41_block_epoch = torch.from_numpy(ep).to(dev, non_blocking=True)


# Fixes the model cannot run correctly without (a missing one gives cross-request reads or duplicated output, not
# an error), checked by their markers in the installed runner sources.
REQUIRED_FIXES = ("prompt-order", "resumed-prefill-logits", "gather-logprobs-int32", "prompt-logprobs")


def missing_fixes(*modules) -> list[str]:
    import inspect

    src = "".join(inspect.getsource(m) for m in modules)
    return [f for f in REQUIRED_FIXES if f"[gaudi-fix:{f}]" not in src]


def install() -> None:
    if _state["installed"]:
        return
    import collections

    from vllm_gaudi.v1.worker import hpu_input_batch, hpu_model_runner as hmr

    missing = missing_fixes(hmr, hpu_input_batch)
    if missing:
        raise RuntimeError(
            f"dsv41: required cookbook fixes not applied: {', '.join(missing)}. Run "
            "python3 fixes/apply_all.py --profile v0.30.0 before starting the server (the recipe's fix step)."
        )

    runner_cls = hmr.HPUModelRunner

    # 1. trim_attn_metadata keeps a fixed field list, and _set_block_mapping later rebuilds the tuple through
    #    custom_tuple_replace(..., "TrimmedAttentionMetadata") from the CACHED type's fields. So the cached type
    #    itself is extended with our fields (defaults None, appended last), and every trim and rebuild carries them.
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
        if not isinstance(base, entry["type"]):  # built from the pre-extension type
            base = entry["type"](**base._asdict())
        extra = {f: getattr(metadata, f, None) for f in FIELDS}
        return base._replace(**extra)

    hmr.trim_attn_metadata = trim_attn_metadata

    # 2. prefill batches: contents.blocks is each request's full block list.
    orig_form = runner_cls._form_prefill_batch

    def _form_prefill_batch(self, contents):
        spans = [(int(c), len(t)) for c, t in zip(contents.context_lens, contents.token_ids, strict=False)]
        out = orig_form(self, contents)
        if not out.attn_metadata:
            return out
        md = out.attn_metadata[0]
        # Positions rebuilt on the host, one row per request as vllm-gaudi lays them out (context_len + i, -1
        # padding): reading back its device copy would sync the device in the middle of input preparation.
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
        # decoder replay keeps only each row's last position real; prompt logprobs need every row
        from .replay import KEEP

        plp = getattr(self.input_batch, "num_prompt_logprobs", None) or {}
        md.dsv41_replay = 0 if any(r in plp for r in contents.req_ids) else KEEP
        return out

    runner_cls._form_prefill_batch = _form_prefill_batch

    # 3. decode batches: rows 0 .. num_decodes-1 of the input batch, one token each.
    orig_decode = runner_cls._create_decode_input_data

    def _create_decode_input_data(
        self, num_decodes, num_scheduled_tokens, context_lens, block_table_cpu_tensor, scheduler_output=None
    ):
        out = orig_decode(
            self, num_decodes, num_scheduled_tokens, context_lens, block_table_cpu_tensor, scheduler_output
        )
        B = out.token_ids.shape[0]
        assert out.token_ids.shape[1] == 1, "dsv41: speculative decoding is not supported"
        pos = np.full((B, 1), -1, dtype=np.int64)
        pos[:num_decodes, 0] = np.asarray(context_lens[:num_decodes], dtype=np.int64)
        # decode requests occupy the first num_decodes rows of the input batch; the runner's dummy decode
        # (profiling, padding) has no real request behind it
        real = list(self.input_batch.req_ids[:num_decodes]) if scheduler_output is not None else []
        rids = real + [None] * (B - len(real))
        bt = block_table_cpu_tensor[:num_decodes].tolist()
        blocks = [[b for b in row if b >= 0] for row in bt] + [[]] * (B - num_decodes)
        _attach(self, out.attn_metadata, rids, blocks, pos)
        return out

    runner_cls._create_decode_input_data = _create_decode_input_data

    # 4. block allocation numbers and uids when a request enters (new, or resumed from preemption).
    orig_update = runner_cls._update_states

    def _update_states(self, scheduler_output):
        bs = _state["block_size"]
        for r in scheduler_output.scheduled_new_reqs:
            _state["uids"].pop(r.req_id, None)
            # blocks past the prefix-cache hits are fresh allocations
            _fresh(self, r.block_ids[0], r.num_computed_tokens // bs)
        cached = scheduler_output.scheduled_cached_reqs
        resumed = getattr(cached, "resumed_req_ids", ()) or ()
        for rid in resumed:
            _state["uids"].pop(rid, None)
        for i, rid in enumerate(cached.req_ids):
            ids = cached.new_block_ids[i]
            if not ids:
                continue
            # resumed: ids replace the block list, hits first; running: ids are appended, all fresh
            _fresh(self, ids[0], cached.num_computed_tokens[i] // bs if rid in resumed else 0)
        for rid in scheduler_output.finished_req_ids:
            _state["uids"].pop(rid, None)
        return orig_update(self, scheduler_output)

    runner_cls._update_states = _update_states

    # 5. no attention metadata. For every prompt chunk with context, vllm-gaudi builds a dense bf16
    #    [B, 1, query, context + query] mask for vLLM's attention backends (~570 MB at a 4096-token chunk over 64K
    #    of context, ~2 GB at 128K, plus a copy while building it), and for decode a block mapping and bias. V4.1's
    #    attention is its own and reads none of it. Left on, it costs that HBM and puts vllm-gaudi's context-block
    #    bucket into the first layer's lazy graph, so every (query bucket x context bucket) pair compiles on its
    #    own on first use. The runner's own builders and the forward path's HPUAttentionMetadataProcessor
    #    (vllm-gaudi 058f107 builds the prompt mask there) both pass the metadata through unchanged.
    def _no_attn_bias(self, attn_metadata, *args, **kwargs):
        return attn_metadata

    runner_cls._set_attn_bias = _no_attn_bias
    if hasattr(runner_cls, "set_attn_bias"):
        runner_cls.set_attn_bias = _no_attn_bias
    proc_cls = getattr(hmr, "HPUAttentionMetadataProcessor", None)
    if proc_cls is not None:
        proc_cls.process_metadata = _no_attn_bias
    _state["installed"] = True
    logger.warning("dsv41: HPU runner hook installed (block tables, block epochs, engram rows)")


def batch_ctx_from_metadata(md, positions: torch.Tensor, block_size: int):
    """The model side: BatchCtx from the trimmed metadata of this forward."""
    from .paged import BatchCtx

    uid = md.dsv41_req_uid
    pos = positions.long()
    pos = torch.where(uid.view(-1, 1) >= 0, pos, torch.full_like(pos, -1))
    return BatchCtx(
        positions=pos,
        block_table=md.dsv41_block_table.long(),
        req_uid=uid,
        block_size=block_size,
        block_epoch=getattr(md, "dsv41_block_epoch", None),
    )


__all__ = ["install", "configure", "batch_ctx_from_metadata", "FIELDS"]
