# SPDX-License-Identifier: Apache-2.0
"""Hooks on the vllm-gaudi v0.30.0 runner and ops for Ripple, installed from the model constructor (so only in
worker processes). Each has a switch whose off value restores vllm-gaudi's behaviour.

  FusedSDPA query tiles   RIPPLE_FSDPA_WS_GIB (default 2; 0 = off), RIPPLE_FSDPA_Q_TILE (default 256)
  fast softmax            RIPPLE_FAST_SOFTMAX_MIN_KV (default 0 = off; the recipe sets 16384)
  decode graph budget     RIPPLE_GRAPH_BUDGET_GIB (default auto; off = none), graph_budget.py

vllm-gaudi's FusedSDPA slicer (VLLM_HPU_FSDPA_SLICE_ENABLED, on by default with VLLM_BUCKETING_STRATEGY=pad) is
refused: its online-softmax merge returns NaN when a key slice is all bucket padding (Ripple prompts of 8192+
tokens gave NaN prompt logprobs), and with that fixed it was faster only for full 4096-token chunks below 128K of
context and slower for short turns on a long cached context.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("vllm.ripple_hpu")

# A biased FusedSDPA prompt call (every prefill chunk with cached context: the full layers' bias is
# [1, 1, q, context + q]) keeps a device workspace of about 60 B per (query row, key) for as long as the process
# lives, reused and grown by later calls: 8 to 31 GiB per card from 32K to 128K of context at 4096-token chunks.
# Query rows are independent (each tile sees all keys and its own bias rows), so vllm-gaudi's own q-tiling path
# computes the same attention; 256-row tiles keep the workspace near zero for about 3.5% more attention time.
FSDPA_WS_GIB = float(os.environ.get("RIPPLE_FSDPA_WS_GIB", "2"))
FSDPA_Q_TILE = int(os.environ.get("RIPPLE_FSDPA_Q_TILE", "256"))
FSDPA_BYTES_PER_QK = 64
# Fast softmax for the prompt attention of a chunk against at least this many keys, in the full-attention layers
# only. A long prompt's time goes almost all to the fp32 softmax of those layers (a 4096-row chunk against 258K
# keys: 329 ms per layer with fp32 softmax, 104 ms with the fast one, both within 0.0035 of an exact fp32
# attention). Sliding layers, short prompts and decode keep VLLM_FP32_SOFTMAX; fast softmax everywhere moved the
# full-depth oracle comparison past its limit.
FAST_SOFTMAX_MIN_KV = int(os.environ.get("RIPPLE_FAST_SOFTMAX_MIN_KV", "0"))

_state = {"installed": False}


def fast_softmax_wanted(softmax_mode, full_attention: bool, window_size, attn_mask, q_len: int, kv_len: int) -> bool:
    """Whether a FusedSDPA call switches from fp32 to fast softmax (pure, for the tests)."""
    return (
        FAST_SOFTMAX_MIN_KV > 0
        and softmax_mode == "fp32"
        and full_attention
        and window_size is None
        and attn_mask is not None
        and kv_len >= FAST_SOFTMAX_MIN_KV
        and q_len != kv_len
    )


def num_q_tiles(n: int, q_len: int, kv_len: int) -> int:
    """Query tiles for a biased prompt call, given vllm-gaudi's own count n (pure, for the tests)."""
    if FSDPA_WS_GIB <= 0 or q_len <= FSDPA_Q_TILE or FSDPA_BYTES_PER_QK * q_len * kv_len <= FSDPA_WS_GIB * 2**30:
        return n
    return max(n, -(-q_len // FSDPA_Q_TILE))


def install() -> None:
    if _state["installed"]:
        return
    from vllm_gaudi.extension.runtime import get_config
    from vllm_gaudi.v1.worker import hpu_model_runner as hmr

    if getattr(get_config(), "enable_fsdpa_slicing", False):
        raise RuntimeError(
            "ripple_hpu: vllm-gaudi's FusedSDPA slicing is on; set VLLM_HPU_FSDPA_SLICE_ENABLED=false "
            "(its merge returns NaN on padded key slices)"
        )

    if FSDPA_WS_GIB > 0:
        from vllm_gaudi.extension import ops as hpu_ops

        orig_tiles = hpu_ops._fsdpa_num_q_tiles

        def _fsdpa_num_q_tiles(attn_bias):
            n = orig_tiles(attn_bias)
            if attn_bias is None:
                return n
            return num_q_tiles(n, attn_bias.size(-2), attn_bias.size(-1))

        hpu_ops._fsdpa_num_q_tiles = _fsdpa_num_q_tiles

    if FAST_SOFTMAX_MIN_KV > 0:
        from vllm_gaudi.extension.utils import ModuleFusedSDPA

        orig_fwd = ModuleFusedSDPA.forward

        def forward(
            self,
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            softmax_mode,
            recompute_mode,
            valid_sequence_lengths,
            padding_side="left",
            window_size=None,
            sinks=None,
        ):
            # _ripple_full_attention: set per layer by ripple_hpu.model (False on sliding layers)
            if fast_softmax_wanted(
                softmax_mode,
                getattr(self, "_ripple_full_attention", False),
                window_size,
                attn_mask,
                query.shape[-2],
                key.shape[-2],
            ):
                softmax_mode = "fast"
            return orig_fwd(
                self,
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                softmax_mode,
                recompute_mode,
                valid_sequence_lengths,
                padding_side,
                window_size,
                sinks,
            )

        ModuleFusedSDPA.forward = forward

    from . import graph_budget

    graph_budget.install(hmr.HPUModelRunner)
    _state["installed"] = True
    logger.warning(
        "ripple_hpu: runner hooks installed (FusedSDPA q tiles %s, fast softmax for full-layer prompt attention %s)",
        f"of {FSDPA_Q_TILE} rows above a {FSDPA_WS_GIB:g} GiB workspace" if FSDPA_WS_GIB > 0 else "off",
        f"from {FAST_SOFTMAX_MIN_KV} keys" if FAST_SOFTMAX_MIN_KV > 0 else "off",
    )


__all__ = ["install", "fast_softmax_wanted", "num_q_tiles", "FSDPA_WS_GIB", "FSDPA_Q_TILE", "FAST_SOFTMAX_MIN_KV"]
