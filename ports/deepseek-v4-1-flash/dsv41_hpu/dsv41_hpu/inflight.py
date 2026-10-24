# SPDX-License-Identifier: Apache-2.0
"""Prefill run-ahead bound: the host stops launching prefill layers while device memory is nearly full of work it
has launched and the device has not run yet.

In lazy mode a prefill forward is one graph per layer (core.DSV41Stack and replay.run mark a step after each layer).
The host records and launches a layer in roughly the time the device needs to run it, and nothing in a prompt chunk
that does not finish its prompt waits for the device (no logits, no sampled token, no .cpu()): vllm-gaudi returns to
the scheduler and the host starts the next chunk. So the host drifts ahead of the device by whole chunks, and every
launched but unrun graph keeps its inputs, outputs and intermediates allocated. On a TP 8 replica with this port's
serve arguments and a 128K prompt, from the third chunk on the device had never finished the previous chunk when the
host started the next, and HBM in use reached 94.48 of the 94.62 GiB pool by the 13th chunk. In that state the
bridge can fail a graph launch with "ValidateSyncInputTensors tensor_data is empty".

DSV41_PREFILL_HEADROOM_GIB=H (default 6; 0 = off): once the allocator's in-use bytes come within 2 x H GiB of its
limit, an HPU event is recorded after each prefill layer's launch, and before launching more the host waits for the
oldest unfinished layer while in use is within H GiB of the limit. Below the first threshold nothing is recorded
(an event costs the host ~2-3 ms, ~8% of a 128K prefill if recorded after every layer). The queue is kept across
forwards, so a new chunk cannot run ahead of an unfinished one either. Same ops in the same order: outputs are
unchanged. Measured with a 128K prompt and 2304 KV blocks: peak in use at most 88.9 GiB, 0-2% more prefill time.
Decode (HPU graph replay, S == 1) and forwards under MIN_T tokens are untouched.
"""

from __future__ import annotations

import collections
import os

HEADROOM = int(float(os.environ.get("DSV41_PREFILL_HEADROOM_GIB", "6")) * (1 << 30))
MIN_T = 256
_events: collections.deque = collections.deque()
_state = {"limit": None, "waits": 0}


def bounded(tokens: int) -> bool:
    """Whether a forward of `tokens` tokens (B * S) is bounded."""
    return HEADROOM > 0 and tokens >= MIN_T


def after_layer() -> None:
    """Call right after a layer's mark_step in a bounded forward."""
    import habana_frameworks.torch as htorch

    hpu = htorch.hpu
    if _state["limit"] is None:
        _state["limit"] = int(hpu.memory_stats()["Limit"])
    cap = _state["limit"] - HEADROOM
    if hpu.memory_allocated() <= cap - HEADROOM:
        # far from the cap: record nothing
        _events.clear()
        return
    ev = hpu.Event()
    ev.record()
    _events.append(ev)
    while len(_events) > 1 and _events[0].query():  # finished: forget them
        _events.popleft()
    # oldest first; with only this layer's event left that is a full drain of the launched work
    while _events and hpu.memory_allocated() > cap:
        _events.popleft().synchronize()
        _state["waits"] += 1


__all__ = ["HEADROOM", "MIN_T", "bounded", "after_layer"]
