"""Automatic prefix caching for Mistral Large 4.

vLLM's prefix cache (--enable-prefix-caching) hands a request the full blocks another request (or an earlier turn)
already computed. What that needs from this port, and where it lives:

  paged caches  every cache (63 latent in the KV codec, 44 index-K in U8, their scales inside the same row) is one
                row per token in vLLM's blocks, all under one block table (paged.cache_layout): a hit block holds
                every kind at once. Rows are tagged (block allocation number, position) (paged.py); the runner keeps
                a hit block's number and gives every other block it hands out a new one (runner_patch.py
                _update_states), so a hit reads as valid while rows left by a block's previous owner stay masked.
  same-step     a request whose hit covers blocks another prompt writes in the same step runs after it
                (runner_patch 4b, ascending context order).
  DSpark rings  per request slot, filled from the target's taps during the request's own forwards (dspark.py). A
                first draft at the prompt's end m reads positions m-2048 .. m-1; the part of them a hit covers was
                never written for this request, so it reads as masked (lower acceptance, never a wrong token: the
                verify sampler is exact), until the request's own tokens fill the window.

install_hit_cap (engine process, KVCacheManager.get_computed_blocks): with ML4_HIT_KEEP = k > 0, a hit of a request
that will draft ends at least k tokens before the request's end (k = 2048: the first drafts see a full window, as cold),
rounded down to a block. Requests that never draft (logprobs, penalties, seed, a grammar unless ML4_SPEC_GRAMMAR:
spec.draftable) and servers
without ML4_SPEC=dspark always keep vLLM's own cap (num_tokens - 1). Default 0: measured on 32K / 110K conversations
, keep 2048 cost 4.6-9 s more first-token time per turn for ~1-4 % better acceptance, so a
drafting request takes the whole hit too. The control file key hit_keep overrides ML4_HIT_KEEP (bring-up A/B; the
engine process reads the file at every lookup, a stat when it has not changed).

ML4_HIT_ALIGN=2048 (default 0: off; control key hit_align): every hit, drafting or not, also ends on a multiple of
2048, the prompt-chunk size (--max-num-batched-tokens), so a request alone on the server runs the same chunks as cold.
A request that never drafts then gives the cold run's output bit for bit; a drafting one also needs ML4_HIT_KEEP=2048
(otherwise its first drafts differ, so do its verify batches, which move greedy near-ties as any batch shape does).
A hit that ends elsewhere starts the request's chunks at other
positions, which moves greedy near-ties exactly as another prompt sharing the steps does without any cache
. Costs up to 2047 more prompt tokens per hit.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("vllm.mistral_large4_hpu")

KEEP = int(os.environ.get("ML4_HIT_KEEP", "0"))
ALIGN = int(os.environ.get("ML4_HIT_ALIGN", "0"))
_seen = {"capped": 0}


def keep_for(request) -> int:
    """Tokens before the request's end that its prefix-cache hit must leave to the request itself."""
    if os.environ.get("ML4_SPEC", "") != "dspark":
        return 0
    from .spec import draftable

    if not draftable(getattr(request, "sampling_params", None)):
        return 0
    from .control import get as ctl_get

    return max(0, ctl_get("hit_keep", KEEP))


def align_for() -> int:
    from .control import get as ctl_get

    return max(0, ctl_get("hit_align", ALIGN))


def cap_hit(n_hit: int, n_tokens: int, keep: int, block: int, align: int = 0) -> int:
    """The hit length after the cap: at most n_tokens - keep, rounded down to a block (vLLM's hits are whole
    blocks), then down to a multiple of align (a multiple of block) when align > 0."""
    if keep > 0 and n_hit > n_tokens - keep:
        n_hit = max(n_tokens - keep, 0) // block * block
    if align > 0:
        n_hit = n_hit // align * align
    return n_hit


def install_hit_cap() -> None:
    from vllm.v1.core import kv_cache_manager as m

    cls = m.KVCacheManager
    if getattr(cls, "_ml4_hit_cap", False):
        return
    orig = cls.get_computed_blocks

    def get_computed_blocks(self, request):
        blocks, n, boundary = orig(self, request)
        if n == 0:
            return blocks, n, boundary
        keep = keep_for(request)
        nb = len(blocks.blocks[0])
        bs = n // nb
        n_new = cap_hit(n, request.num_tokens, keep, bs, align_for())
        if n_new == n:
            return blocks, n, boundary
        if _seen["capped"] < 3:
            logger.warning(
                "ml4: prefix-cache hit %d of %d tokens capped to %d (DSpark keeps the last %d, align %d)",
                n,
                request.num_tokens,
                n_new,
                keep,
                align_for(),
            )
        _seen["capped"] += 1
        cut = m.KVCacheBlocks(tuple(list(g)[: n_new // bs] for g in blocks.blocks))
        return cut, n_new, boundary

    cls.get_computed_blocks = get_computed_blocks
    cls._ml4_hit_cap = True
    logger.warning(
        "ml4: prefix-cache hit cap installed (ML4_SPEC=%s, keep %d for drafting requests, align %d)",
        os.environ.get("ML4_SPEC", ""),
        KEEP,
        ALIGN,
    )


__all__ = ["KEEP", "ALIGN", "keep_for", "align_for", "cap_hit", "install_hit_cap"]
