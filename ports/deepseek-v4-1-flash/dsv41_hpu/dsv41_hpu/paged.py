# SPDX-License-Identifier: Apache-2.0
"""Per-forward addressing of DeepSeek-V4.1's paged caches, with ownership tags.

Every persistent per-token artifact (window KV, compressed KV, index K, the ratio-2 compressor's pending state)
lives in a vLLM-allocated paged cache and is addressed only through the request's own block table. A cache whose
rows are compressor entries (one per `ratio` tokens) keeps entry e in the block that holds its tokens:

    row(b, e) = block_table[b, e // (BLOCK / ratio)] * BLOCK + e % (BLOCK / ratio)

so block j of a request holds exactly what tokens j*BLOCK .. j*BLOCK+BLOCK-1 produced, in every cache. That is what
vLLM's prefix caching assumes when it hands one request another's full blocks: their contents depend only on the
shared prefix. (For ratio 2 the second half of each block's rows in those caches is unused.)

Each row carries a tag (owner, logical index), written with the row and checked on every read. With `block_epoch`
(serving) the owner is the block's allocation number, sent per request aligned with its block table: the runner
bumps a physical block's number whenever vLLM hands the block out fresh and keeps it when a request receives the
block as a prefix-cache hit. A row is valid only if it was written during the block's current allocation, at this
logical index: stale rows from a previous owner, freed or never-written rows and the null block are masked. Without
`block_epoch` (CPU tests) the owner is the request uid.

DSV41_TAG_CHECK=mask (default) masks such rows; count also counts them on the device (the isolation gate reads the
count, which must stay 0 in normal operation); off skips the check.

Physical block 0 is vLLM's null block and is never handed to a request, so rows 0 .. BLOCK-1 take the writes of
padding tokens (tag -1 never matches).

DSV41_DECODE_HOIST=1 (off by default): the layers of one decode graph share one BatchCtx with a memo, so per-step
work that is identical in every layer runs once per graph instead of once per layer: cache-row addressing for the
window write/read and the entry writes, RoPE table lookups, and whole reads that several layers repeat (the
compressed-entry keys every Reuse layer gathers with its index source's top-k, the index-K table the index sources
score). Same ops on the same inputs, so bit-identical; a write to a cache drops that cache's memoised reads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from .numerics import LEAN

TAG_MODE = os.environ.get("DSV41_TAG_CHECK", "mask")  # mask | count | off
HOIST = os.environ.get("DSV41_DECODE_HOIST", "0") == "1"
# count mode: per (device, cache kind, prefill|decode) [violating rows, rows read], updated in place on the device,
# so the counts keep accumulating inside HPU graph replays (which do not run Python) and the hot path never syncs.
# fp32: int64 is unreliable in lazy mode, and a violation count only has to be exactly zero.
_counters: dict = {}


def _counter(device, label: str) -> torch.Tensor:
    key = (str(device), label)
    c = _counters.get(key)
    if c is None:
        c = _counters[key] = torch.zeros(2, dtype=torch.float32, device=device)
    return c


def tag_violations(by_label: bool = False) -> dict:
    rows = reads = 0.0
    labels: dict = {}
    for (_, label), c in _counters.items():
        v = c.cpu().tolist()
        rows, reads = rows + v[0], reads + v[1]
        if v[0]:
            labels[label] = labels.get(label, 0) + int(v[0])
    out = {"rows": int(rows), "reads": int(reads)}
    if by_label:
        out["by_label"] = labels
    return out


def reset_tag_violations() -> None:
    with torch.inference_mode():  # the counters may have been created inside it
        for c in _counters.values():
            c.zero_()


@dataclass
class BatchCtx:
    """What one forward needs to address the caches. All [B, ...] with B the padded batch; padding requests have
    uid -1 and positions -1."""

    positions: torch.Tensor  # [B, S] long, -1 for padding tokens
    block_table: torch.Tensor  # [B, NB] long, physical block ids (-1 beyond the request)
    req_uid: torch.Tensor  # [B] int32, runner-assigned; -1 for padding
    block_size: int
    block_epoch: torch.Tensor | None = None  # [B, NB] int32 allocation number of each table entry (-1 past the end)
    memo: dict | None = None  # DSV41_DECODE_HOIST: per-forward results shared by layers

    def shared(self, key, make):
        """make() once per forward under DSV41_DECODE_HOIST (memo set), else every call."""
        if self.memo is None:
            return make()
        if key not in self.memo:
            self.memo[key] = make()
        return self.memo[key]

    @property
    def valid(self) -> torch.Tensor:
        return self.positions >= 0

    @property
    def max_len(self) -> int:
        return self.block_table.shape[1] * self.block_size

    def rows(self, x: torch.Tensor, per_block: int | None = None, key=None):
        """Physical rows for logical indices x [B, ...], `per_block` indices per block (block_size for per-token
        caches, block_size // ratio for compressor entries). Returns (rows, addressable, owner): not addressable
        (x < 0, past the table, or an unallocated block) maps to row 0; owner is the tag a valid row must carry.
        key names x for the DSV41_DECODE_HOIST memo (same key = same indices)."""
        per_block = per_block or self.block_size
        if key is not None and self.memo is not None:
            return self.shared(("rows", key, per_block), lambda: self._rows(x, per_block))
        return self._rows(x, per_block)

    def _rows(self, x: torch.Tensor, per_block: int):
        B = self.block_table.shape[0]
        flat = x.reshape(B, -1)
        blk_idx = torch.div(flat, per_block, rounding_mode="floor")
        inside = (flat >= 0) & (blk_idx < self.block_table.shape[1])
        blk = torch.gather(self.block_table, 1, blk_idx.clamp(0, self.block_table.shape[1] - 1))
        ok = inside & (blk >= 0)
        rows = torch.where(
            ok, blk * self.block_size + torch.remainder(flat, per_block), 0 if LEAN else torch.zeros_like(flat)
        )
        if self.block_epoch is not None:
            # same gather and index as the block table: block_epoch is aligned with it
            owner = torch.gather(self.block_epoch, 1, blk_idx.clamp(0, self.block_table.shape[1] - 1))
        else:
            owner = self.req_uid.to(torch.int32).view(B, 1).expand_as(flat)
        return rows.reshape(x.shape), ok.reshape(x.shape), owner.reshape(x.shape)


class TaggedCache:
    """A paged cache view [num_rows, width] plus its tag table [num_rows, 2] int32. `ratio` > 1: logical indices
    are compressor entries, `ratio` tokens each."""

    def __init__(self, data: torch.Tensor, tags: torch.Tensor, ratio: int = 1, label: str = "cache"):
        assert data.shape[0] == tags.shape[0], (data.shape, tags.shape)
        self.data = data
        self.tags = tags
        self.ratio = ratio
        self.label = label  # cache kind, for count mode's breakdown

    def _per_block(self, ctx: BatchCtx) -> int:
        assert ctx.block_size % self.ratio == 0
        return ctx.block_size // self.ratio

    def write(self, ctx: BatchCtx, x: torch.Tensor, values: torch.Tensor, keep: torch.Tensor, key=None) -> None:
        """Write values [B, S, W] at logical indices x [B, S] where keep [B, S]."""
        if ctx.memo is not None:  # memoised reads of this cache are stale now
            for k in [k for k in ctx.memo if k[0] == "read" and k[1] == id(self.data)]:
                del ctx.memo[k]
        rows, ok, owner = ctx.rows(x, self._per_block(ctx), key)
        keep = keep & ok
        if LEAN:
            rows = torch.where(keep, rows, 0).reshape(-1)
            tag = torch.stack([torch.where(keep, owner, -1), torch.where(keep, x.to(torch.int32), -1)], dim=-1)
        else:
            rows = torch.where(keep, rows, torch.zeros_like(rows)).reshape(-1)
            tag = torch.stack(
                [
                    torch.where(keep, owner, torch.full_like(owner, -1)),
                    torch.where(keep, x.to(torch.int32), torch.full_like(owner, -1)),
                ],
                dim=-1,
            )
        self.data.index_copy_(0, rows, values.reshape(-1, values.shape[-1]).to(self.data.dtype))
        self.tags.index_copy_(0, rows, tag.reshape(-1, 2))

    def read(
        self, ctx: BatchCtx, x: torch.Tensor, want: torch.Tensor, key=None, share: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather rows for logical indices x [B, ...] that the caller expects to exist (want). Returns
        (values [B, ..., W], valid [B, ...]) where valid = want and the row's tag is exactly (owner, x). key names
        (x, want) for the DSV41_DECODE_HOIST memo; share memoises the whole read per cache (layers reading the same
        cache at the same key get the same tensors)."""
        if share and key is not None and ctx.memo is not None:
            return ctx.shared(("read", id(self.data), key), lambda: self._read(ctx, x, want, key))
        return self._read(ctx, x, want, key)

    def _read(self, ctx: BatchCtx, x: torch.Tensor, want: torch.Tensor, key=None):
        rows, ok, owner = ctx.rows(x, self._per_block(ctx), key)
        flat = rows.reshape(-1)
        vals = self.data.index_select(0, flat).reshape(*x.shape, self.data.shape[-1])
        tag = self.tags.index_select(0, flat).reshape(*x.shape, 2)
        want = want & ok
        valid = want & (tag[..., 0] == owner) & (tag[..., 1] == x.to(torch.int32))
        if TAG_MODE == "count":
            phase = "decode" if ctx.positions.shape[1] == 1 else "prefill"
            _counter(vals.device, f"{self.label}/{phase}").add_(
                torch.stack([(want & ~valid).sum(), want.sum()]).float()
            )
        elif TAG_MODE == "off":
            valid = want
        return vals, valid
