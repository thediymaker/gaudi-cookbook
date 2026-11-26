"""Runtime control for same-server A/B on a test server (not a serving feature).

A JSON file (ML4_CTL, default /tmp/ml4_ctl.json) re-read whenever its mtime changes. Keys:
  decode_eager                    true: decode steps run the eager stack instead of the HPU graphs (graphs.py)
  timer_every, timer_sync         override ML4_DECODE_TIMER / ML4_DECODE_TIMER_SYNC
  mem_epoch                       a new value restarts the allocator peak that ML4_MEM_LOG reports
  mem_hold_gib                    every rank keeps this many GiB allocated (headroom test: a 64K prompt must still run)
  exec                            with ML4_CTL_EXEC=1: a Python file every rank runs once per change (vllm_model.py)
  index_select_bytes, index_step_bytes, attn_step_bytes, prefill_mark
                                  override the prompt-chunk tiling env (indexer.py, attention.py, stack.py)
  prompt_mask, prompt_absorb_keys, index_bf16_dot
                                  override ML4_PROMPT_MASK / ML4_PROMPT_ABSORB_KEYS / ML4_INDEX_BF16_DOT (indexer.py)
  prefill_graphs, prefill_graph_buckets, prefill_graph_group
                                  override ML4_PREFILL_GRAPHS / _BUCKETS (a list) / _GROUP (graphs.py; a new group size
                                  re-wraps the prompt graphs)
  prefill_graph_epoch             a new value re-wraps the prompt graphs (a fresh capture)
  decode_graph_dry_run            overrides ML4_DECODE_GRAPH_DRY_RUN (re-wraps all graphs; a re-wrap on a serving
                                  engine hung its next collective)
  prefill_mxk_max, moe_pad_skip   override ML4_PREFILL_MXK_MAX / ML4_MOE_PAD_SKIP (moe.py; read at capture: pad skip
                                  re-wraps all graphs, a new mxk max needs a new prefill_graph_epoch)
  dspark_k, dspark_cost, dspark_adapt, dspark_ctx_g, dspark_ctx
                                  override the DSpark draft policy env (spec.py; the last two: the long-context gate)
  verify_rows                     overrides ML4_VERIFY_ROWS (spec.py: verify steps as [B, T] rows or flat; no
                                  re-capture, each layout has its own graph shapes)
  index_decode_select             overrides ML4_INDEX_DECODE_SELECT (indexer.py; re-captures the decode graphs, and a
                                  re-capture on a running engine has hung its collectives; set it before
                                  the first decode step)
  attn_fsdpa, attn_fsdpa_rows      override ML4_ATTN_FSDPA / _ROWS (attention.py: prompt attention through FusedSDPA)
  attn_fsdpa_dtype                test: "f32kv" / "f32" run FusedSDPA's k / v or the whole kernel in fp32 (no better)
  bt_prompt_half                  overrides ML4_BT_PROMPT_HALF (runner_patch.py: 3/4-power-of-two prompt widths)
  index_split                     overrides ML4_INDEX_SPLIT (indexer.py: the TP ranks share a prompt chunk's mask rows)
  attn_soft_lean                  overrides ML4_ATTN_SOFT_LEAN (attention.py: fp32 prompt softmax, two fewer passes)
  prefill_prof                    true: every eager prompt chunk syncs at each layer section and rank 0 prints the
                                  split (prof.py; test only, it removes the host/device overlap)
Every rank reads it at its next forward, so change it only while the server is idle: all ranks then switch at the
same step.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger("vllm.mistral_large4_hpu")

CTL_PATH = os.environ.get("ML4_CTL", "/tmp/ml4_ctl.json")
_ctl = {"mtime": None, "v": {}}


def ctl() -> dict:
    try:
        m = os.stat(CTL_PATH).st_mtime
    except OSError:
        m = None
    if m != _ctl["mtime"]:
        _ctl["mtime"] = m
        try:
            _ctl["v"] = json.load(open(CTL_PATH)) if m is not None else {}
        except (OSError, ValueError):
            _ctl["v"] = {}
        logger.warning("ml4: control file %s: %s", CTL_PATH, _ctl["v"])
    return _ctl["v"]


def get(key: str, default):
    v = ctl().get(key)
    return default if v is None else type(default)(v)


__all__ = ["ctl", "get", "CTL_PATH"]
