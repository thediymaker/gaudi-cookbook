"""Decode steps as HPU graphs of layer groups (the DeepSeek-V4.1 port's GroupStep pattern).

Eager lazy mode rebuilds the decode step's lazy graph in Python on every step: 72 layers of indexer + MLA + MoE,
~640 ms of host work per step at bring-up with the device mostly idle. Here a decode step
(S == 1) runs as ML4_LAYER_GROUP consecutive layers per HPU graph (htorch.hpu.wrap_in_hpu_graph). A graph is
captured on first use of each input shape (batch bucket x block-table width) and replayed after that; replay runs
no Python, so

  * every cross-layer value is an explicit tensor in a fixed order: the residual x and the top-k of the last
    full-indexer layer (shared-indexer layers read it; an empty tensor before the first full layer has run);
  * the cache views are fixed attributes (their tensors never move once vLLM binds them);
  * nothing host-side may depend on values. The indexer's "every row is short" shortcut (BatchCtx.host_max_pos)
    is derived from the block-table WIDTH instead of the batch's positions: the runner picks the smallest width
    that covers the batch (runner_patch._bt_width), so a table of <= 2048 keys means every position is <= 2047,
    and the graph cache already keys on that shape. Wider tables run the scored path; for rows at p <= 2047 it
    returns the same dense top-k (indexer.select), so the result is the same either way.

ML4_DECODE_HOIST=1 (default) gives the layers of one graph one BatchCtx with a memo: block-table addressing of the
step's positions, of the whole key range and of each index source's top-k, and the RoPE / log(p+1) lookups, are
computed once per graph instead of once per layer. Bit-identical (the same ops on the same inputs).

ML4_PREFILL_GRAPHS=1 (default) runs short prompt chunks through one HPU graph per layer: batch
1, S one of ML4_PREFILL_GRAPH_BUCKETS (default 64, 128, 256, 384, 512, 768: the runner's exp query buckets for prompts
of up to 768 tokens, with the 64-row bucket runner_patch adds below the block size), a block table of <= 2048 keys.
Each (S, table width) is captured on first use (~1 s). Eager, such a chunk is ~0.7-0.95 s of host dispatch, and every
decoding stream waits for it; replayed, it is its device work, mostly the MoE (~0.19 / 0.27 /
0.33 s at 30 / 50 / 64 tokens, ~0.44 s at S 128, ~0.62 s at S 512). The 896 and 1280 buckets also work (1.0-1.1 s
-> 0.75-0.89 s) but their graphs take 4.1 GiB more per card, and with them the 64K prompt beside 3 GiB held ran the
allocator into defragmentation and the engine died. Inside these graphs the MoE takes the mxk decode kernels at
S 64 (rows of padding skip them, moe.PAD_SKIP) and the stock MXFP4 fused op above, where eager takes v8p (v8p
flushes the lazy graph between its unpack and the fused op; the two give the same values eager). Longer chunks,
wider tables and batches of several prompts stay eager.

Other prefill chunks, DSpark taps and the forced-form tests (ML4_ATTN_FORM) stay eager.
"""

from __future__ import annotations

import logging
import os
import time

import torch
from torch import nn

from .indexer import TOPK
from .paged import BatchCtx

logger = logging.getLogger("vllm.mistral_large4_hpu")

LAYER_GRAPHS = os.environ.get("ML4_LAYER_GRAPHS", "0") == "1"
LAYER_GROUP = int(os.environ.get("ML4_LAYER_GROUP", "8"))
HOIST = os.environ.get("ML4_DECODE_HOIST", "1") == "1"
# ML4_DECODE_TIMER=N: rank 0 logs the decode step every N steps: host enqueue ms (no sync), and with
# ML4_DECODE_TIMER_SYNC=1 the full step ms (synchronize before and after; costs one sync per step).
TIMER_EVERY = int(os.environ.get("ML4_DECODE_TIMER", "0"))
TIMER_SYNC = os.environ.get("ML4_DECODE_TIMER_SYNC", "0") == "1"
PREFILL_GRAPHS = os.environ.get("ML4_PREFILL_GRAPHS", "1") == "1"
# Rows per request that still run the decode graphs (a DSpark verify step as [B, T] rows, spec.as_rows);
# longer rows are prompt chunks.
DECODE_ROWS_MAX = 8
PREFILL_BUCKETS = frozenset(
    int(v) for v in os.environ.get("ML4_PREFILL_GRAPH_BUCKETS", "64,128,256,384,512,768").split(",")
)
PREFILL_MAX_KEYS = int(os.environ.get("ML4_PREFILL_GRAPH_MAX_KEYS", "2048"))
# Prompt graphs: their own wrappers (own cache) over groups of ML4_PREFILL_GRAPH_GROUP layers (default 1: one graph per
# layer), captured with the wrapper's dry run (ML4_PREFILL_GRAPH_DRY_RUN=1): one trial replay at capture finds the
# graph's internal tensors that can be freed after each replay. Without the dry run (as the decode graphs run) every
# captured prompt shape keeps ~5 MiB per prompt row, and the 1280 bucket's capture ran out of device memory after the
# other six. One layer per graph is the closest to the fp32 oracle of the partitions tried (groups of 8 are a little
# further than eager; groups of 72 hung the device at capture: "No progress error");,
PREFILL_GROUP = int(os.environ.get("ML4_PREFILL_GRAPH_GROUP", "1"))
PREFILL_DRY_RUN = os.environ.get("ML4_PREFILL_GRAPH_DRY_RUN", "1") == "1"
# The decode graphs with the wrapper's dry run too (default; control key decode_graph_dry_run re-wraps them): a captured
# decode shape kept ~4.5 MiB per row without it, ~0.1-0.2 with it (407 -> 15 MiB per card for the 9 shapes of 1-24 rows
# at table 16); same tokens, same step time.
DECODE_DRY_RUN = os.environ.get("ML4_DECODE_GRAPH_DRY_RUN", "1") == "1"
# ML4_BOUNDARY_TRIM=1 (r5, default off): a plain decode step's front (runner_patch.batch_ctx_from_metadata's eager
# ops, spec.before_forward's slot copy, stack.embed_tokens with its all-reduce) runs inside the first group's graph
# (FrontStep); vllm_model._forward_trim. Same ops, same order; verify steps and prompt chunks are unchanged.
BOUNDARY_TRIM = os.environ.get("ML4_BOUNDARY_TRIM", "0") == "1"
from .control import ctl  # noqa: E402  (runtime A/B switches, control.py)


def prefill_eligible(ids: torch.Tensor, block_table: torch.Tensor, block_size: int) -> bool:
    """A prompt chunk the layer-group graphs take (ML4_PREFILL_GRAPHS / _BUCKETS; control keys prefill_graphs and
    prefill_graph_buckets override them)."""
    c = ctl()
    if not c.get("prefill_graphs", PREFILL_GRAPHS):
        return False
    return (
        ids.shape[0] == 1
        and ids.shape[1] in c.get("prefill_graph_buckets", PREFILL_BUCKETS)
        and block_table.shape[1] * block_size <= PREFILL_MAX_KEYS
    )


def host_max_from_width(block_table: torch.Tensor, block_size: int) -> int | None:
    """host_max_pos as a function of the table width only (see the module note)."""
    lk = block_table.shape[1] * block_size
    return lk - 1 if lk <= TOPK else None


class GroupStep(nn.Module):
    """Consecutive ML4Layers with a tensors-only interface: (x, positions, block_table, req_uid, block_epoch,
    topk_in) -> (x, topk_out). topk_out is the top-k of the last full-indexer layer at or before the group's end
    (topk_in itself when the group has none)."""

    def __init__(self, layers, caches: dict, rope, allreduce, block_size: int):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.caches, self.rope, self.allreduce, self.block_size = caches, rope, allreduce, block_size
        self.first, self.last = layers[0].role.layer_id, layers[-1].role.layer_id
        self.src_in = layers[0].role.index_source if not layers[0].role.has_indexer else None
        self.src_out = layers[-1].role.index_source

    def forward(self, x, positions, block_table, req_uid, block_epoch, topk_in):
        from .attention import Shared

        ctx = BatchCtx(
            positions=positions,
            block_table=block_table,
            req_uid=req_uid,
            block_size=self.block_size,
            block_epoch=block_epoch if block_epoch.numel() else None,
            memo={} if HOIST else None,
            host_max_pos=host_max_from_width(block_table, self.block_size),
        )
        shared = Shared()
        if self.src_in is not None:
            shared.topk[self.src_in] = topk_in
        for layer in self.layers:
            x = layer(x, ctx, self.caches, shared, self.rope, self.allreduce)
        return x, shared.topk[self.src_out]


class PromptStep(GroupStep):
    """GroupStep for the prompt graphs: x goes into `out`, a caller buffer, and only the top-k comes back. A graph's
    returned tensors are its static outputs, kept for as long as the graph: one [S, H] x per layer graph and prompt
    size, 4.25 GiB per card for the six sizes to 768. Two buffers per size, alternating, hold the
    same values (equal to the static-output graphs bit for bit, a quarter of the memory)."""

    def forward(self, x, out, positions, block_table, req_uid, block_epoch, topk_in):
        y, topk = super().forward(x, positions, block_table, req_uid, block_epoch, topk_in)
        out.copy_(y)
        return topk


class FrontStep(GroupStep):
    """The first decode group with the step front inside (ML4_BOUNDARY_TRIM): (ids, positions, block_table int32,
    req_uid, block_epoch, req_slot int32 or empty, topk_in) -> (x, topk_out, positions, block_table), the last two the
    BatchCtx tensors the other groups take. The front ops are runner_patch.batch_ctx_from_metadata's, then
    spec.before_forward's slot copy into the drafter's slot buffer (read by the tap hooks inside the graphs), then
    stack.embed_tokens, in that order, as the untrimmed step runs them eagerly."""

    def __init__(self, layers, caches: dict, rope, allreduce, block_size: int, stack, slot_buf):
        super().__init__(layers, caches, rope, allreduce, block_size)
        object.__setattr__(self, "_stack", stack)  # not a submodule: the embed is read, not owned
        object.__setattr__(self, "_slot_buf", slot_buf)  # spec.py's drafter.slot_buf, or None without DSpark

    def forward(self, ids, positions, block_table, req_uid, block_epoch, req_slot, topk_in):
        pos = positions.long()  # batch_ctx_from_metadata
        if pos.dim() == 1:
            pos = pos.view(req_uid.shape[0], -1)
        pos = torch.where(req_uid.view(-1, 1) >= 0, pos, torch.full_like(pos, -1))
        bt = block_table.long()
        slot = req_slot.long()
        if self._slot_buf is not None:  # spec.before_forward (req_slot present: _trim_ok)
            # as an index op on the base tensor: a slice copy_ here, read through a slice by the tap hooks in later
            # graphs, fails at capture ("Empty tensor optional ... Strided Params", variant W,
            # the first arm C start); index_copy_ gives the same buffer (xg.py variant I, exact over 6 steps)
            B = pos.shape[0]
            self._slot_buf.index_copy_(0, torch.arange(B, device=slot.device), slot.to(self._slot_buf.dtype))
        x = self._stack.embed_tokens(ids)  # DecodeGraphs._run
        x, topk = super().forward(x, pos, bt, req_uid, block_epoch, topk_in)
        return x, topk, pos, bt


class DecodeGraphs:
    """The model's decode groups, wrapped once per set of cache views."""

    def __init__(self, stack, views: dict, block_size: int, slot_buf=None):
        import habana_frameworks.torch as htorch
        from . import lean

        if lean.enabled():
            logger.warning("ml4: lean decode: %d layers prepared", lean.prepare(stack))
        layers = list(stack.layers)
        n = max(1, LAYER_GROUP)
        rope = stack.rope()
        self.groups = [
            GroupStep(layers[i : i + n], views, rope, stack.allreduce, block_size) for i in range(0, len(layers), n)
        ]
        self.dry_run = bool(ctl().get("decode_graph_dry_run", DECODE_DRY_RUN))
        self.graphs = [
            htorch.hpu.wrap_in_hpu_graph(g, disable_tensor_cache=True, dry_run=self.dry_run) for g in self.groups
        ]
        self.front = None
        if BOUNDARY_TRIM:
            self.front = htorch.hpu.wrap_in_hpu_graph(
                FrontStep(layers[:n], views, rope, stack.allreduce, block_size, stack, slot_buf),
                disable_tensor_cache=True,
                dry_run=self.dry_run,
            )
            logger.warning(
                "ml4: boundary trim on: plain decode steps run the step front inside graph 0 (slot copy %s)",
                "on" if slot_buf is not None else "off",
            )
        self._parts = (layers, views, rope, stack.allreduce, block_size)
        self.pgraphs, self.pbufs, self.pkey = [], {}, None
        self.stack = stack
        self.shapes: dict = {}
        self.n_steps = 0
        self.t_host = 0.0
        self.t_full = 0.0
        logger.warning(
            "ml4: decode HPU graphs armed: %d groups of %d layers (hoist %s, dry run %s); prompt graphs %s",
            len(self.groups),
            n,
            "on" if HOIST else "off",
            self.dry_run,
            f"groups of {PREFILL_GROUP} layers, dry run {PREFILL_DRY_RUN}" if PREFILL_GRAPHS else "off by default",
        )

    def _prompt_graphs(self) -> None:
        """Wrap the prompt groups on first use, and again when the control key prefill_graph_group asks for another
        group size or prefill_graph_epoch changes (tests only; the new set captures afresh)."""
        import habana_frameworks.torch as htorch

        c = ctl()
        want = (max(1, int(c.get("prefill_graph_group", PREFILL_GROUP))), c.get("prefill_graph_epoch"))
        if want == self.pkey:
            return
        layers, views, rope, allreduce, block_size = self._parts
        p = want[0]
        self.pgraphs, self.pbufs = None, {}
        self.pgraphs = [
            htorch.hpu.wrap_in_hpu_graph(
                PromptStep(layers[i : i + p], views, rope, allreduce, block_size),
                disable_tensor_cache=True,
                dry_run=PREFILL_DRY_RUN,
            )
            for i in range(0, len(layers), p)
        ]
        if self.pkey is not None:
            self.shapes = {k: v for k, v in self.shapes.items() if k[1] <= 8}
            logger.warning("ml4: prompt graphs rewrapped: groups of %d layers (epoch %s)", *want)
        self.pkey = want

    def __call__(self, ids: torch.Tensor, ctx: BatchCtx, rank0: bool) -> torch.Tensor:
        from . import moe

        if ids.shape[1] > 8:
            self._prompt_graphs()
        key = (ids.shape[0], ids.shape[1], ctx.block_table.shape[1])
        first = key not in self.shapes
        moe.GRAPH_PROMPT[0] = ids.shape[1] > 8  # MoE calls inside a prompt graph: no lazy-graph flush
        try:
            return self._run(ids, ctx, rank0, key, first)
        finally:
            moe.GRAPH_PROMPT[0] = False

    def run_trim(self, ids, positions, block_table, req_uid, block_epoch, req_slot, rank0: bool) -> torch.Tensor:
        """A plain decode step with the front inside graph 0 (ML4_BOUNDARY_TRIM; vllm_model._forward_trim)."""
        key = (ids.shape[0], ids.shape[1], block_table.shape[1])
        first = key not in self.shapes
        slot = req_slot if req_slot is not None else ids.new_zeros(0, dtype=torch.int32)
        return self._run(ids, None, rank0, key, first, front=(positions, block_table, req_uid, block_epoch, slot))

    def _run(self, ids, ctx, rank0, key, first, front=None):
        c = ctl()
        every, sync = int(c.get("timer_every", TIMER_EVERY)), bool(c.get("timer_sync", TIMER_SYNC))
        timed = rank0 and (first or every > 0)
        if timed and (first or sync):
            torch.hpu.synchronize()
        m0 = torch.hpu.memory_allocated() if first and rank0 else 0
        t0 = time.perf_counter()
        if front is not None:
            positions, block_table, req_uid, block_epoch, slot = front
            topk = ids.new_zeros(0, dtype=torch.long)
            epoch = block_epoch if block_epoch is not None else ids.new_zeros(0, dtype=torch.int32)
            x, topk, pos, bt = self.front(ids, positions, block_table, req_uid, epoch, slot, topk)
            for g in self.graphs[1:]:
                x, topk = g(x, pos, bt, req_uid, epoch, topk)
        else:
            x = self.stack.embed_tokens(ids)
            topk = ids.new_zeros(0, dtype=torch.long)
            epoch = ctx.block_epoch if ctx.block_epoch is not None else ids.new_zeros(0, dtype=torch.int32)
            if ids.shape[1] > 8:
                bufs = self.pbufs.get(x.shape)
                if bufs is None:
                    bufs = self.pbufs[x.shape] = (torch.empty_like(x), torch.empty_like(x))
                for i, g in enumerate(self.pgraphs):
                    topk = g(x, bufs[i % 2], ctx.positions, ctx.block_table, ctx.req_uid, epoch, topk)
                    x = bufs[i % 2]
            else:
                for g in self.graphs:
                    x, topk = g(x, ctx.positions, ctx.block_table, ctx.req_uid, epoch, topk)
        if timed:
            t1 = time.perf_counter()
            if first or sync:
                torch.hpu.synchronize()
            t2 = time.perf_counter()
            if first:
                self.shapes[key] = t2 - t0
                logger.warning(
                    "ml4: %s graphs captured for batch %d x %d tokens x table %d blocks: %.1f s, %+.0f MiB "
                    "live (%d shapes so far, %.1f s total)",
                    "prompt" if key[1] > 8 else "decode",
                    key[0],
                    key[1],
                    key[2],
                    t2 - t0,
                    (torch.hpu.memory_allocated() - m0) / 2**20,
                    len(self.shapes),
                    sum(self.shapes.values()),
                )
            else:
                self.n_steps += 1
                self.t_host += t1 - t0
                self.t_full += t2 - t0
                if self.n_steps >= every:
                    logger.warning(
                        "ml4: graph step (batch %d x %d, table %d): host enqueue %.2f ms%s over %d steps",
                        key[0],
                        key[1],
                        key[2],
                        1e3 * self.t_host / self.n_steps,
                        f", full {1e3 * self.t_full / self.n_steps:.2f} ms" if sync else "",
                        self.n_steps,
                    )
                    self.n_steps, self.t_host, self.t_full = 0, 0.0, 0.0
        else:
            self.shapes.setdefault(key, 0.0)
        return x


__all__ = [
    "LAYER_GRAPHS",
    "LAYER_GROUP",
    "BOUNDARY_TRIM",
    "FrontStep",
    "PREFILL_GRAPHS",
    "PREFILL_GROUP",
    "PREFILL_DRY_RUN",
    "DECODE_DRY_RUN",
    "GroupStep",
    "DecodeGraphs",
    "host_max_from_width",
    "prefill_eligible",
    "ctl",
]
