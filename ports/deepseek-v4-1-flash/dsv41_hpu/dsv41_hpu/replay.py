# SPDX-License-Identifier: Apache-2.0
"""Decoder replay in prefill: how DeepSeek serves V4.1.

DeepSeek's own engine, DeepInfer (https://github.com/deepseek-ai/DeepInfer, "decoder SWA bounded replay", keep
140 tokens), prefills V4.1 this way. Every decoder layer's global KV is the first ratio-1 layer's compressed cache
(layer 20's comp / idx_k), and that depends only on layer 20's INPUT. So a prompt token needs the decoder (layer
20's query side, indexer, attention and MoE, then layers 21-39) only for what is read later: the sampled last
position and the decoder window rows of the last 128 tokens. Layers 0-19 plus layer 20's compressor run on every
prompt token, the rest on the prompt's last KEEP tokens only, with each decoder window clipped to the replayed
segment: a query at i sees the window keys max(e - KEEP, i - 127) .. i, e = the prompt's end. It is an
approximation: in the reference a decoder window row depends on decoder states up to 20 chained windows (~2.5K
tokens) back. Decode is unchanged.

Here, per prompt row of a forward (one chunk, positions c .. e-1, n = e - c tokens):
  * layers 0-19 run on every token; layer 20's compressor writes comp / idx_k for every token;
  * the row's last min(n, KEEP) tokens are gathered into a compact [B, Sk] batch (Sk = min(S, KEEP)) that runs the
    rest of layer 20 and layers 21-39, so decoder window rows are written for those tokens only;
  * decoder window reads are floored at e - KEEP (Floored). EVERY chunk replays its own tail, not only the last
    one (~3% more decoder work at 4096-token chunks), so a chunk shorter than KEEP finds the rows between its floor
    and its start already written by the chunk before it, and no read below a floor is ever needed;
  * prefix-cache hits end at least KEEP tokens before the end of the request (install_hit_cap), so a replayed
    segment never reaches into shared blocks, whose decoder window rows were never written.
Decode at position p reads window rows >= p - 127 > e - KEEP: all written by the replay or by decode.

The returned hidden state keeps the batch's [B, S] layout. Only each row's last position (the sampled one) is real;
the other rows are zero. So a batch with a prompt_logprobs request runs the full stack (runner_patch sets
dsv41_replay = 0 for it), which also gives a same-server quality A/B.

DSV41_REPLAY_KEEP=140 turns it on (0, the default, runs the full stack). run(..., compact=False) runs every row
through every layer with the same window floors (DeepInfer's mask mode); tests/test_replay.py checks that the kept
rows match the compact run.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger("vllm.dsv41_hpu")

KEEP = int(os.environ.get("DSV41_REPLAY_KEEP", "0"))


def _paged():
    from . import paged

    return paged


def make_ctx(base, positions: torch.Tensor, floor: torch.Tensor):
    """The consumer layers' BatchCtx: base's block table, epochs and uids, `positions` [B, Sk], and the window floor
    [B, 1] that their Floored views read."""
    ctx = _paged().BatchCtx(
        positions=positions,
        block_table=base.block_table,
        req_uid=base.req_uid,
        block_size=base.block_size,
        block_epoch=base.block_epoch,
    )
    ctx.swa_floor = floor
    return ctx


def _floored_cls():
    """(floor mixin, read-only mixin), combined with each cache's own class."""

    class Floored:
        """A window cache whose rows below the row's floor (ctx.swa_floor) are neither wanted nor read."""

        def read(self, ctx, x, want, *args, **kwargs):
            fl = ctx.swa_floor.view(ctx.swa_floor.shape[0], *([1] * (x.dim() - 1)))
            return super().read(ctx, x, want & (x >= fl), *args, **kwargs)

    class ReadOnly:
        """Layer 20's entry caches in compact mode: its compressor already wrote every token's entry."""

        def write(self, *args, **kwargs) -> None:
            return None

    return Floored, ReadOnly


_COMBOS: dict = {}


def _rewrap(cache, mixin):
    key = (mixin, type(cache))
    if key not in _COMBOS:
        _COMBOS[key] = type(mixin.__name__ + type(cache).__name__, (mixin, type(cache)), {})
    cls = _COMBOS[key]
    c = cls.__new__(cls)
    c.__dict__.update(cache.__dict__)
    return c


class Replay:
    """Per-stack setup: the boundary layer and the consumer layers' cache views (rebuilt when the views change)."""

    def __init__(self, stack):
        roles = stack.roles
        self.b0 = next(r.layer_id for r in roles if r.ratio == 1)
        later = roles[self.b0 :]
        assert all(r.ratio == 1 and r.kv_source == self.b0 and r.index_source >= self.b0 for r in later), (
            "dsv41 replay: every consumer layer must read the boundary layer's ratio-1 cache"
        )
        assert roles[self.b0].is_kv_source and roles[self.b0].is_candidate_source, roles[self.b0]
        assert not any(lid >= self.b0 for lid in stack.cfg.engram_layer_ids), "dsv41 replay: engram past the boundary"
        self._key = None

    def views(self, caches: dict, compact: bool) -> dict:
        key = (tuple(id(v.data) for d in caches.values() for v in d.values()), compact)
        if key != self._key:
            Floored, ReadOnly = _floored_cls()
            out = dict(caches)
            for lid, d in caches.items():
                if lid < self.b0:
                    continue
                d = dict(d)
                d["swa"] = _rewrap(d["swa"], Floored)
                if lid == self.b0 and compact:
                    for kind in ("comp", "idx_k"):
                        d[kind] = _rewrap(d[kind], ReadOnly)
                out[lid] = d
            self._views, self._key = out, key
        return self._views


def _take(t: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """t [B, S, ...] -> [B, Sk, ...] rows src [B, Sk] along dim 1."""
    idx = src.view(*src.shape, *([1] * (t.dim() - 2))).expand(*src.shape, *t.shape[2:])
    return torch.gather(t, 1, idx)


def plan(positions: torch.Tensor, keep: int):
    """Rows are left-packed (c .. e-1, then -1 padding). Returns src [B, Sk] (the kept tokens' columns), start [B, 1]
    (first kept column), n [B, 1] (tokens in the row), positions [B, Sk] of the kept tokens (-1 padding) and the
    window floor [B, 1] = e - keep. All on the device: no sync."""
    B, S = positions.shape
    Sk = min(S, keep)
    n = (positions >= 0).sum(1, keepdim=True)
    k = n.clamp(max=Sk)
    start = n - k
    j = torch.arange(Sk, device=positions.device).view(1, Sk)
    src = (start + j).clamp(max=S - 1)
    ok = j < k
    pos_c = torch.where(ok, torch.gather(positions, 1, src), torch.full_like(src, -1))
    end = torch.gather(positions, 1, (n - 1).clamp(min=0)) + 1
    return src, start, n, pos_c, end - keep


def run(stack, rp: Replay, input_ids, ctx, caches, engram_rows, keep: int, compact: bool = True):
    """DSV41Stack.forward with the decoder replayed on each row's last `keep` tokens (compact=False: mask mode)."""
    from . import inflight
    from .csa import Shared
    from .mhc import hc_pre, identity_pre
    from .numerics import rms_norm

    bound = stack.mark_step is not None and inflight.bounded(input_ids.numel())

    def step():
        if stack.mark_step is not None:
            stack.mark_step()
            if bound:  # prefill run-ahead bound (inflight.py)
                inflight.after_layer()

    e = stack.embed(input_ids)
    h = e.unsqueeze(-2).repeat(*([1] * (e.dim() - 1)), stack.hc, 1)
    pre = identity_pre(h, stack.hc)
    shared = Shared()
    for layer in stack.layers[: rp.b0]:
        rows = engram_rows.get(layer.layer_id) if engram_rows is not None else None
        h, pre = layer(h, pre, ctx, caches, shared, rows)
        step()
    src, start, n, pos_c, floor = plan(ctx.positions, keep)
    views = rp.views(caches, compact)
    if compact:
        # the boundary layer's compressor reads its attention input, which every token has: all entries first
        first = stack.layers[rp.b0]
        x = rms_norm(hc_pre(h, pre), first.attn_norm, first.eps)
        first.attn._compress(x, ctx, caches[rp.b0])
        h, pre = _take(h, src), _take(pre, src)
        cctx = make_ctx(ctx, pos_c, floor)
    else:
        cctx = make_ctx(ctx, ctx.positions, floor)
    step()
    shared = Shared()
    for layer in stack.layers[rp.b0 :]:
        h, pre = layer(h, pre, cctx, views, shared, None)
        step()
    out = rms_norm(hc_pre(h, pre), stack.norm, stack.cfg.rms_norm_eps)
    if not compact:
        return out
    S = input_ids.shape[1]
    s = torch.arange(S, device=out.device).view(1, S)
    out = _take(out, (s - start).clamp(0, out.shape[1] - 1))
    return out.masked_fill(((s < start) | (s >= n)).unsqueeze(-1), 0)


def install_hit_cap(keep: int = KEEP) -> None:
    """Engine process: prefix-cache hits end at least `keep` tokens before the end of the request (vLLM stops them
    at num_tokens - 1), so the replayed segment is always computed by the request itself. Requests shorter than keep
    get no hit."""
    if keep <= 0:
        return
    from vllm.v1.core import kv_cache_manager as m

    cls = m.KVCacheManager
    if getattr(cls, "_dsv41_hit_cap", None) == keep:
        return
    orig = getattr(cls, "_dsv41_orig_get_computed_blocks", cls.get_computed_blocks)

    def get_computed_blocks(self, request):
        blocks, n, boundary = orig(self, request)
        cap = request.num_tokens - keep
        if n <= cap or n == 0:
            return blocks, n, boundary
        nb = len(blocks.blocks[0])
        bs = n // nb
        n_new = max(cap, 0) // bs * bs
        cut = m.KVCacheBlocks(tuple(list(g)[: n_new // bs] for g in blocks.blocks))
        return cut, n_new, boundary

    cls._dsv41_orig_get_computed_blocks = orig
    cls.get_computed_blocks = get_computed_blocks
    cls._dsv41_hit_cap = keep
    logger.warning(
        "dsv41: decoder replay on (keep %d): prefix-cache hits end >= %d tokens before the request's end", keep, keep
    )


__all__ = ["KEEP", "Replay", "plan", "run", "install_hit_cap"]
