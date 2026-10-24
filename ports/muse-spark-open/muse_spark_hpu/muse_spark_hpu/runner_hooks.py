# SPDX-License-Identifier: Apache-2.0
"""Runner hooks installed from the model constructor (worker processes only).

1. FusedSDPA workspace. A prompt attention call with an explicit bias (every prefill chunk with cached
   context: the global layers' bias is [1, 1, q, ctx + q]) keeps a device workspace of about 60 B per
   (query row, key) for the life of the process; later calls reuse and grow it. At 4096-token chunks that
   reached 15.5 GiB per card around 64K of context with 13.1 GiB free, and the logits came back NaN. Query
   rows are independent, so vllm-gaudi's own query tiling computes the same attention; this makes it tile in
   FSDPA_Q_TILE rows whenever the workspace would exceed FSDPA_WS_GIB (stock tiles only past a 2**31-byte bias
   plane). 256-row tiles keep the workspace near zero for about +3.5% attention time. long_context.py takes
   over above 64K keys with MUSE_SPARK_LONG_ATTN=1.

2. MUSE_SPARK_MEMLOG=1 (optional): one "muse_spark mem:" log line from TP rank 0 with the device allocator's
   peak and current bytes after a step that raises the peak by >= 256 MiB or moves the current figure by
   >= 512 MiB, and every 200th step. An HBM out-of-memory on this model can hang all eight cards, so it is
   worth watching how close real traffic gets to the limit. No measurable cost.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("vllm.muse_spark_hpu")

FSDPA_WS_GIB = 2.0
FSDPA_Q_TILE = 256
# Device bytes FusedSDPA keeps per (query row, key) of a biased prompt call, measured on Gaudi 2 (16 q / 2 kv
# heads, head 64, fp32 softmax): 2.79 / 8.27 / 15.54 / 30.93 GiB after one call at 4096 queries and
# 12160 / 36096 / 67840 / 135040 keys, about 60 B.
FSDPA_BYTES_PER_QK = 64
MEMLOG = os.environ.get("MUSE_SPARK_MEMLOG", "0") == "1"
_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    from vllm_gaudi.extension import ops as hpu_ops
    from vllm_gaudi.v1.worker import hpu_model_runner as hmr

    stock_tiles = hpu_ops._fsdpa_num_q_tiles

    def _fsdpa_num_q_tiles(attn_bias):
        n = stock_tiles(attn_bias)
        if attn_bias is None:
            return n
        q_len, kv_len = attn_bias.size(-2), attn_bias.size(-1)
        if q_len <= FSDPA_Q_TILE or FSDPA_BYTES_PER_QK * q_len * kv_len <= FSDPA_WS_GIB * 2**30:
            return n
        return max(n, -(-q_len // FSDPA_Q_TILE))

    hpu_ops._fsdpa_num_q_tiles = _fsdpa_num_q_tiles

    from . import long_context

    long_context.install()
    if MEMLOG:
        _install_memlog(hmr.HPUModelRunner)
    _installed = True
    logger.warning(
        "muse_spark_hpu: runner hooks installed (FusedSDPA query tiles of %d rows above a %g GiB workspace%s)",
        FSDPA_Q_TILE,
        FSDPA_WS_GIB,
        ", memory log" if MEMLOG else "",
    )


def _install_memlog(runner_cls) -> None:
    orig_sample = runner_cls.sample_tokens
    mem = {"peak": 0, "now": 0, "steps": 0}

    def sample_tokens(self, *args, **kwargs):
        out = orig_sample(self, *args, **kwargs)
        from vllm.distributed import get_tensor_model_parallel_rank

        if get_tensor_model_parallel_rank() == 0:
            import habana_frameworks.torch as htorch

            mem["steps"] += 1
            st = htorch.hpu.memory_stats()
            peak, inuse, limit = st.get("MaxInUse", 0), st.get("InUse", 0), st.get("Limit", 0)
            if peak >= mem["peak"] + (256 << 20) or abs(inuse - mem["now"]) >= (512 << 20) or mem["steps"] % 200 == 0:
                mem["peak"], mem["now"] = max(peak, mem["peak"]), inuse
                logger.warning(
                    "muse_spark mem: step %d peak in use %.2f GiB, now %.2f GiB, limit %.2f GiB, "
                    "headroom at peak %.2f GiB",
                    mem["steps"],
                    peak / 2**30,
                    inuse / 2**30,
                    limit / 2**30,
                    (limit - peak) / 2**30,
                )
        return out

    runner_cls.sample_tokens = sample_tokens
