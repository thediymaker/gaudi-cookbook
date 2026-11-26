"""Prompt-chunk time split (control key prefill_prof; a test key, never in serving).

With prefill_prof true, every eager prompt chunk (S > 8 rows, not inside a prompt graph) flushes the lazy graph and
synchronizes the device at each section boundary of every layer, and rank 0 prints one line per forward with the ms
spent in each section summed over the layers:
    q        the q projection, RoPE and the latent cache write (attention before the indexer)
    index    the indexer (44 layers); in prompt mask mode split further into
               ix_pre    the indexer's q / weights and the key read
               ix_rows   the scores and the k-th value search of this rank's rows
               index     the rest: the rows' all-gather and unpacking (index_split)
    attn     the attention core and wo; under attn_fsdpa split further into
               attn_kv   the keys and values through wkv_b
               attn_mask the allowed mask of each row block
               attn_sdpa FusedSDPA
               attn      the rest: the cache read and wo
    ar       the two all-reduces of every layer
    ffn      the MoE (or layer 0's dense FFN)
The syncs remove the host/device overlap of lazy mode, so the total is above the unprofiled chunk time; compare the
sections with each other, and the total with prefill_prof off.
"""

from __future__ import annotations

import time

import torch

from . import moe as _moe
from .control import get as ctl_get

DECODE_S = 8
_st = {"on": False, "t": 0.0, "acc": {}}


def begin(x: torch.Tensor) -> None:
    """At the start of a forward: arm the timer for a profiled prompt chunk."""
    _st["on"] = (
        x.shape[1] > DECODE_S
        and x.device.type == "hpu"
        and not _moe.GRAPH_PROMPT[0]
        and bool(ctl_get("prefill_prof", False))
    )
    if _st["on"]:
        _st["acc"] = {}
        _st["t0"] = _st["t"] = _now()


def lap(key: str) -> None:
    """Charge the time since the last lap to key."""
    if _st["on"]:
        t = _now()
        _st["acc"][key] = _st["acc"].get(key, 0.0) + t - _st["t"]
        _st["t"] = t


def end(rows: int, keys: int, rank: int) -> None:
    """At the end of a forward: rank 0 prints the split."""
    if not _st["on"]:
        return
    lap("rest")
    _st["on"] = False
    if rank == 0:
        acc = _st["acc"]
        parts = ", ".join(f"{k} {1e3 * v:.0f}" for k, v in acc.items())
        print(f"ml4: prof rows {rows} keys {keys}: {1e3 * (_st['t'] - _st['t0']):.0f} ms ({parts})", flush=True)


def _now() -> float:
    from .indexer import _mark_step

    _mark_step()
    torch.hpu.synchronize()
    return time.perf_counter()
