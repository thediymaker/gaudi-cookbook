# SPDX-License-Identifier: Apache-2.0
"""Routed-expert layout for GLM-5.x at TP 8 with expert parallelism (``patches/moe_layout.py`` wires it in).

With ``--enable-expert-parallel`` vLLM puts 32 whole experts on each of the 8 ranks (EP 8, contiguous: rank k holds
experts 32k..32k+31). Every decode MoE layer then waits at its all-reduce for the rank with the most local routes:
16 routes at one stream (verify token and MTP draft, top-8 each) into 8 ranks give the busiest rank about 4.5
routes against a mean of 2. The imbalance is statistical, not a hot rank, so the fix is a wider layout:

``GLM_MOE_TP=2`` (unset or 1: EP 8)
    The 8 TP ranks form 4 expert groups of 2 ranks ("EP 4 x TP 2"). Group g holds experts [64g, 64g+64); inside a
    group, rank r holds rows [r I/2, (r+1) I/2) of each expert's gate and up projections and the same columns of
    the down projection. Every rank still holds 32 experts' worth of weights. vLLM derives the intermediate size per
    partition and the local expert range from the parallel config; the MoE output is all-reduced over the 8 ranks
    as before. Outputs differ from EP 8 by bf16 rounding (each route's down projection is summed over two partial
    slices in the all-reduce instead of in one fp32 accumulator).

``GLM_MOE_PLACEMENT=<file.json>`` (unset: contiguous groups)
    Which experts each group holds, per layer: ``{"n_experts": 256, "n_ranks": 4, "layers": {"<layer>": [expert of
    slot 0, ..., expert of slot 255]}}``; slot s lives in group s // 64. The router's top-k ids are mapped to slot ids
    with one gather from a per-layer table on the device, so routing itself is unchanged and every route still
    reaches the same expert weights. ``serving/glm53-ep4-placement.json`` is fitted to recorded GLM-5.3 routing (it
    splits experts that the MTP verify token and its draft tend to share); GLM-5.2 runs without one.
"""

import ctypes
import json
import os
import re
import time

import torch

TP = int(os.environ.get("GLM_MOE_TP", "1") or 1)
PLACEMENT = os.environ.get("GLM_MOE_PLACEMENT", "")
_cache = {}


def group_layout(tp_size, tp_rank, dp_size, pcp_size):
    """(moe tp_size, moe tp_rank, ep_size, ep_rank) for GLM_MOE_TP, or None for vLLM's own EP layout."""
    t = TP
    if t > 1 and dp_size == 1 and pcp_size == 1 and tp_size % t == 0 and tp_size > t:
        return t, tp_rank % t, tp_size // t, tp_rank // t
    return None


def placement():
    """{layer index: experts by slot} from GLM_MOE_PLACEMENT, read and validated once per process."""
    if "layers" not in _cache:
        layers = {}
        if PLACEMENT:
            with open(PLACEMENT) as f:
                d = json.load(f)
            ne = int(d["n_experts"])
            for k, v in d["layers"].items():
                v = [int(x) for x in v]
                if sorted(v) != list(range(ne)):
                    raise ValueError(f"{PLACEMENT}: layer {k} is not a permutation of 0..{ne - 1}")
                layers[int(k)] = v
            print(f"[glm-moe] placement {PLACEMENT}: {len(layers)} layers, {ne} experts, {d['n_ranks']} groups")
        _cache["layers"] = layers
    return _cache["layers"]


def experts_by_slot(layer_name):
    if not PLACEMENT:
        return None
    m = re.search(r"layers\.(\d+)\.", layer_name or "")
    return placement().get(int(m.group(1))) if m else None


def load_fused(layer, param, loaded_weight):
    """Fused-MXFP4 loader hook for a checkpoint tensor [E, d1, d2] that holds every expert. Copies this rank's slice
    into param and returns True, or returns False to let the stock branch do it (EP 8 without a placement)."""
    n_local = param.shape[0]
    pc = layer.moe_config.moe_parallel_config
    ebs = experts_by_slot(layer.layer_name)
    num_e = loaded_weight.shape[0]
    ep = layer.moe_config.ep_rank
    ids = list(range(num_e))
    if num_e > n_local:
        lo, hi = ep * n_local, (ep + 1) * n_local
        ids = ebs[lo:hi] if ebs is not None else list(range(lo, hi))
    if pc.tp_size > 1 and pc.use_ep and loaded_weight.dim() == 3:
        t, r = pc.tp_size, pc.tp_rank
        _, d1, d2 = loaded_weight.shape
        p1, p2 = param.shape[1], param.shape[2]
        # deepseek_v2 calls the loader without a weight name: w13 / w13 scale are t times wider in dim 1 ([gate; up]
        # rows), w2 / w2 scale in dim 2 (packed columns / scale groups)
        w13 = d1 == t * p1 and p1 % 2 == 0
        if w13 or (d1 == p1 and d2 == t * p2):
            _load_tp_slice(param.data, loaded_weight, ids, w13, r, t, ep * t + r, pc.ep_size * t)
            return True
    if ebs is not None and num_e > n_local:
        idx = torch.tensor(ids, dtype=torch.long, device=loaded_weight.device)
        sel = loaded_weight.index_select(0, idx)
        param.data[:n_local, : sel.shape[1], : sel.shape[2]].copy_(sel)
        return True
    return False


def _cached(addr, nbytes, stall_s):
    """Wait until host bytes [addr, addr + nbytes) of the safetensors mmap are in the page cache. False when their
    missing-page count stops falling for stall_s (stall_s <= 0: look once) or mincore cannot tell."""
    libc = ctypes.CDLL(None, use_errno=True)
    pg = os.sysconf("SC_PAGE_SIZE")
    a0 = addr - addr % pg
    npg = (addr + nbytes - a0 + pg - 1) // pg
    vec = (ctypes.c_ubyte * npg)()
    last, since = -1, time.monotonic()
    while True:
        if libc.mincore(ctypes.c_void_p(a0), ctypes.c_size_t(npg * pg), vec) != 0:
            return False
        miss = bytes(vec).count(0)
        if miss == 0:
            return True
        if stall_s <= 0:
            return False
        if miss != last:
            last, since = miss, time.monotonic()
        elif time.monotonic() - since > stall_s:
            return False
        time.sleep(0.002)


def _load_tp_slice(p, lw, ids, w13, r, t, k, num_ranks):
    """Copy experts `ids` (in that order, p's dim 0) of checkpoint tensor lw [E, d1, d2] into p: for w13 the gate
    rows block r of t and the up rows block r, for w2 the column block r of t.

    The obvious host gather (index_select of whole experts, then slicing) page-faults the safetensors mapping
    expert by expert, scattered over the file, and loaded 1.4 to 1.9x slower than the stock layout from a cold page
    cache. Instead:
      1. rank k first copies its stock EP 8 share (experts [k E/K, (k+1) E/K)) to a device scratch and drops it, so
         the node reads each tensor once, one sequential stream per rank, as the stock loader does;
      2. it then copies each contiguous run it needs once that run is in the page cache (another rank's share read
         brought it in; a run that stalls 5 s ends the waiting and is read directly). w2's column blocks are taken
         in one single-threaded host gather.
    Step 1 is skipped when the tensor is already cached. The bytes that land in p are the same either way.
    """
    num_e, d1, d2 = lw.shape
    h, w = d1 // 2 // t, d2 // t
    row = d2 * lw.element_size()
    share = (
        p.device.type != "cpu"
        and num_ranks > 1
        and num_e % num_ranks == 0
        and not _cached(lw.data_ptr(), num_e * d1 * row, 0.0)
    )
    ke = num_e // num_ranks if share else num_e
    if share:
        torch.empty((ke, d1, d2), dtype=lw.dtype, device=p.device).copy_(lw.narrow(0, k * ke, ke))
        getattr(torch, p.device.type).synchronize()
    wait = share
    for i in sorted(range(len(ids)), key=lambda i: (ids[i] % ke, ids[i])):
        e = ids[i]
        a = lw.data_ptr() + e * d1 * row
        if w13:
            for j, r0 in enumerate((r * h, d1 // 2 + r * h)):
                wait = wait and _cached(a + r0 * row, h * row, 5.0)
                p[i, j * h : (j + 1) * h].copy_(lw[e, r0 : r0 + h])
        else:
            wait = wait and _cached(a, d1 * row, 5.0)
    if not w13:
        nt = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            p[:, :, :w].copy_(lw.narrow(2, r * w, w).index_select(0, torch.tensor(ids, dtype=torch.long)))
        finally:
            torch.set_num_threads(nt)


def map_ids(layer, topk_ids):
    """Router hook: global expert ids -> slot ids (no-op without a placement for this layer)."""
    tab = getattr(layer, "_glm_slot_of", None)
    if tab is None:
        return topk_ids
    return torch.index_select(tab, 0, topk_ids.reshape(-1)).view(topk_ids.shape)


def install(method_cls):
    """Build each layer's slot table after its weights are loaded (end of vllm_gaudi/ops/hpu_mxfp4.py)."""
    if not PLACEMENT or getattr(method_cls, "_glm_moe_installed", False):
        return
    method_cls._glm_moe_installed = True
    orig = method_cls.process_weights_after_loading

    def process_weights_after_loading(self, layer):
        orig(self, layer)
        ebs = experts_by_slot(getattr(layer, "layer_name", ""))
        if ebs is None:
            return
        if layer.global_num_experts != len(ebs):
            raise ValueError(f"{PLACEMENT}: {len(ebs)} experts, the layer has {layer.global_num_experts}")
        slot_of = [0] * len(ebs)
        for s, e in enumerate(ebs):
            slot_of[e] = s
        layer._glm_slot_of = torch.tensor(slot_of, dtype=torch.int32, device=layer.w13_weight.device)

    method_cls.process_weights_after_loading = process_weights_after_loading
