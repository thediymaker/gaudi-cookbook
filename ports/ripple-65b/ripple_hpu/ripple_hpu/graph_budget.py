# SPDX-License-Identifier: Apache-2.0
"""A bound on the device memory that decode HPU graphs keep.

vllm-gaudi wraps the model in one HPU graph cache (habana_frameworks wrap_in_hpu_graph) that never evicts: every
new decode shape (rows, block-list length, window block-list length) captured at run time (VLLM_SKIP_WARMUP) keeps
its own device memory until the server stops. Ripple's decode graphs keep, at TP 4 (fit over 38 decode shapes,
exact to 2 MiB):

    0.312 MiB x blocks + 1.251 MiB x window blocks + 0.203 MiB x rows

about 20 KiB per block per layer: the 16 full layers read the block list, the 64 sliding layers the window list.
One graph for 32 requests at 64K context each keeps about 5.5 GiB, so a long-running server keeps capturing until
a device allocation fails. The failure lands on one TP rank first; the others wait in the next all-reduce until
the device reports "No progress" about 10 minutes later, which looks like a hang.

Before a forward that will capture a new decode graph, this module predicts that graph's memory with the formula
above (scaled by 4 / TP) and, when the graphs it already tracks plus the new one would exceed the budget, drops
the least recently used decode graphs from the wrapper's cache first (what the wrapper's own clear_cache does for
all of them). A shape whose single graph exceeds the budget runs without a graph. The prediction depends only on
the shape, which every TP rank sees the same, so the ranks evict the same graphs. Prompt graphs are not managed
(they keep under 0.25 GiB each). Nothing here changes a forward's result: an evicted shape is captured again the
next time it runs (about 1 s).

The generic fix hpu-graph-lru bounds the number of graphs, not their size; Ripple's decode graphs range from
tens of MiB to many GiB, so a count cap does not bound memory here, and the two must not be combined (both
manage the same wrapper cache).

  RIPPLE_GRAPH_BUDGET_GIB   GiB for decode graphs, "auto" (default: device memory free at the first decode
                            forward minus RIPPLE_GRAPH_MARGIN_GIB), or "off" (no bound, vllm-gaudi's behaviour)
  RIPPLE_GRAPH_MARGIN_GIB   [8] what "auto" leaves for everything else. The largest measured user: a 250K
                            prompt's eager prefill keeps a workspace that grows to 2 GiB and peaks 3.6 GiB above
                            it in its last 4096-token chunks; at a 6 GiB margin that left 0.43 GiB.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger("vllm.ripple_hpu")

MIB_PER_BLOCK, MIB_PER_WIN_BLOCK, MIB_PER_ROW = 0.312, 1.251, 0.203  # TP 4, see the module docstring
BUDGET = os.environ.get("RIPPLE_GRAPH_BUDGET_GIB", "auto").strip().lower()
MARGIN = float(os.environ.get("RIPPLE_GRAPH_MARGIN_GIB", "8"))

_st: dict = {
    "installed": False,
    "budget": None,
    "scale": None,
    "graphs": {},
    "step": 0,
    "used": 0.0,
    "repeat_logged": 0,
    "evicted": 0,
    "bypass": False,
    "rank": None,
}


def graph_gib(rows: int, blocks: int, win: int, tp: int = 4) -> float:
    """Predicted device memory (GiB) one decode graph of this shape keeps."""
    return (4.0 / tp) * (MIB_PER_BLOCK * blocks + MIB_PER_WIN_BLOCK * win + MIB_PER_ROW * rows) / 1024


def _cache(runner):
    """The wrapper's graph cache: the dict its clear_cache() clears (habana_frameworks hpu/graphs.py)."""
    fn = getattr(getattr(runner, "model", None), "clear_cache", None)
    for cell in getattr(fn, "__closure__", None) or ():
        if isinstance(cell.cell_contents, dict):
            return cell.cell_contents
    return None


def _one(v):
    """A forward input as the graph cache's hash sees it: scalars by value, tensors by shape, view and dtype."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if torch.is_tensor(v):
        return (tuple(v.shape), tuple(v.stride()), v.storage_offset(), str(v.dtype).replace("torch.", ""))
    if isinstance(v, (tuple, list)):
        return tuple(_one(x) for x in v)
    return type(v).__name__


def _describe(token_ids, position_ids, md) -> dict:
    names = getattr(md, "_fields", None) or list(vars(md))
    return {
        "token_ids": _one(token_ids),
        "position_ids": _one(position_ids),
        **{f: _one(getattr(md, f, None)) for f in names},
    }


def _setup() -> None:
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    tp = get_tensor_model_parallel_world_size()
    _st["scale"], _st["rank"] = tp, get_tensor_model_parallel_rank()
    if BUDGET == "auto":
        torch.hpu.synchronize()
        ms = torch.hpu.memory_stats()
        free = (ms.get("Limit", 0) - ms.get("InUse", 0)) / 2**30
        _st["budget"] = max(1.0, free - MARGIN)
        how = f"auto: {free:.2f} GiB free at the first decode minus a {MARGIN:g} GiB margin"
    else:
        _st["budget"] = float(BUDGET)
        how = "RIPPLE_GRAPH_BUDGET_GIB"
    if _st["rank"] == 0:
        logger.warning("ripple graph budget: %.2f GiB for decode HPU graphs (%s; TP %d)", _st["budget"], how, tp)


def _evict(cache, need: float, keep) -> None:
    """Drop least recently used decode graphs until `need` GiB more fits the budget."""
    from habana_frameworks.torch.hpu.graphs import CachedParams

    torch.hpu.synchronize()  # no replay of a graph being dropped is still in flight
    before = torch.hpu.memory_stats().get("InUse", 0)
    n, freed = 0, 0.0
    for key in sorted((k for k in _st["graphs"] if k != keep), key=lambda k: _st["graphs"][k]["last"]):
        if _st["used"] + need <= _st["budget"]:
            break
        g = _st["graphs"].pop(key)
        for h in g["hashes"]:
            cache.pop(h, None)
            CachedParams.cache_hits.pop(h, None)
            CachedParams.skip_replay_cnt.pop(h, None)
        _st["used"] -= g["gib"]
        freed += g["gib"]
        n += 1
    _st["evicted"] += n
    if _st["rank"] == 0:
        after = torch.hpu.memory_stats().get("InUse", 0)
        logger.warning(
            "ripple graph budget: dropped %d decode graphs (%.2f GiB predicted, in use %.2f -> %.2f GiB) "
            "for a new %s (+%.2f GiB); %d tracked, %.2f of %.2f GiB, %d dropped since start",
            n,
            freed,
            before / 2**30,
            after / 2**30,
            keep[:3],
            need,
            len(_st["graphs"]),
            _st["used"],
            _st["budget"],
            _st["evicted"],
        )


def install(runner_cls) -> bool:
    if _st["installed"] or BUDGET == "off":
        return False
    orig = runner_cls._execute_model_generic
    orig_use = runner_cls._use_graphs

    def _use_graphs(self, attn_metadata, batch_size):
        if _st["bypass"]:
            return False
        return orig_use(self, attn_metadata, batch_size)

    def _execute_model_generic(self, token_ids, position_ids, attn_metadata, *a, **k):
        if k.get("warmup_mode") or attn_metadata is None or attn_metadata.is_prompt:
            return orig(self, token_ids, position_ids, attn_metadata, *a, **k)
        cache = _cache(self)
        if cache is None:
            return orig(self, token_ids, position_ids, attn_metadata, *a, **k)
        if _st["budget"] is None:
            _setup()
        # (rows, blocks, window blocks, which swa_kv group slot mappings are None): a step that leaves a group's
        # entry None is hashed by the graph cache as a different graph from one that fills it
        bl, wl = attn_metadata.block_list, getattr(attn_metadata, "window_block_list", None)
        kg = getattr(attn_metadata, "kv_group_slot_mapping", None)
        key = (
            int(token_ids.shape[0]),
            0 if bl is None else int(bl.shape[0]),
            0 if wl is None else int(wl.shape[0]),
            None if kg is None else tuple(x is None for x in kg),
        )
        _st["step"] += 1
        g = _st["graphs"].get(key)
        if g is not None and g["hashes"] and g["hashes"][0] not in cache:  # cleared behind our back
            _st["used"] -= _st["graphs"].pop(key)["gib"]
            g = None
        need = graph_gib(*key[:3], tp=_st["scale"])
        if g is not None:
            g["last"] = _st["step"]
        elif self._use_graphs(attn_metadata, key[0]):
            if need > _st["budget"]:
                _st["bypass"] = True
                try:
                    return orig(self, token_ids, position_ids, attn_metadata, *a, **k)
                finally:
                    _st["bypass"] = False
            if _st["used"] + need > _st["budget"]:
                _evict(cache, need, key)
        n0 = len(cache)
        out = orig(self, token_ids, position_ids, attn_metadata, *a, **k)
        if len(cache) > n0:
            new = list(cache)[n0:]  # a dict keeps insertion order; the wrapper only adds, we only pop
            diag = _st["rank"] == 0 and _st["repeat_logged"] < 5
            d = _describe(token_ids, position_ids, attn_metadata) if diag else None
            if g is not None and d is not None:
                _st["repeat_logged"] += 1
                first = g.get("md") or {}
                logger.warning(
                    "ripple graph budget: shape %s captured another graph (%d already); metadata fields "
                    "that differ from its first capture (first, now): %s",
                    key[:3],
                    len(g["hashes"]),
                    {f: (first.get(f), v) for f, v in d.items() if first.get(f) != v} or "none",
                )
            g = _st["graphs"].setdefault(key, {"hashes": [], "gib": 0.0, "last": _st["step"], "md": d})
            g["hashes"] += new
            g["gib"] += need * len(new)
            _st["used"] += need * len(new)
            if _st["used"] > _st["budget"]:  # a capture the check above did not foresee (a known shape's 2nd graph)
                _evict(cache, 0.0, key)
        return out

    runner_cls._use_graphs = _use_graphs
    runner_cls._execute_model_generic = _execute_model_generic
    _st["installed"] = True
    logger.warning(
        "ripple_hpu: decode graph memory budget installed (RIPPLE_GRAPH_BUDGET_GIB=%s, margin %g GiB)", BUDGET, MARGIN
    )
    return True


__all__ = ["BUDGET", "MARGIN", "graph_gib", "install"]
