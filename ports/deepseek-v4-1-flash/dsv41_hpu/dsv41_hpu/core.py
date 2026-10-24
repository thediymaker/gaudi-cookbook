# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 decoder stack, independent of vLLM.

Order per layer l (reference Transformer.forward / Block.forward):
    h = engram_l(h)                     (layers 1 and 14 only; before the block)
    attn:  (pre_a, post_a, comb_a) = hc_mixes(h);  x = hc_pre(h, pre_prev)
           h = hc_post(attn(attn_norm(x)), h, post_a, comb_a)
    ffn:   (pre_f, post_f, comb_f) = hc_mixes(h);  x = hc_pre(h, pre_a)
           h = hc_post(moe(ffn_norm(x)), h, post_f, comb_f);  pre_prev = pre_f
then logits = head(norm(hc_pre(h, pre_prev))), with pre_prev one-hot before layer 0.

Parameters are created here; `linear` builds the projection modules and `make_moe`
the FFN, so tests plug in plain modules and serving plugs in vLLM's parallel ones.
"""

from __future__ import annotations

import torch
from torch import nn

from . import inflight
from .csa import DSV41Attention, Shared, csa_roles
from .engram import engram_gate, lookup_rows
from .mhc import hc_mixes, hc_post, hc_pre, identity_pre
from .numerics import rms_norm
from .paged import HOIST, BatchCtx


class DSV41Engram(nn.Module):
    """Tables for this rank's hash columns [col0, col1), concatenated; `gather` joins the
    per-rank embeddings back into all columns (identity at TP1)."""

    def __init__(self, cfg, layer_id, bucket_sizes, col0, col1, linear, gather=None):
        super().__init__()
        self.layer_id, self.hc, self.eps = layer_id, cfg.hc_mult, cfg.rms_norm_eps
        self.col0, self.col1 = col0, col1
        hd = cfg.engram_head_dim
        local = bucket_sizes[col0:col1]
        offs = torch.tensor([0] + list(torch.tensor(local).cumsum(0)[:-1].tolist()), dtype=torch.long)
        self.register_buffer("col_offsets", offs, persistent=False)
        n = int(sum(local))
        self.table = nn.Parameter(torch.empty(n, hd, dtype=torch.uint8), requires_grad=False)
        self.table_scale = nn.Parameter(torch.empty(n, hd // 32, dtype=torch.uint8), requires_grad=False)
        n_cols = len(bucket_sizes)
        self.wkv = linear("engram.wkv", n_cols * hd, cfg.hidden_size * (cfg.hc_mult + 1), "replicated", None)
        self.q_weight = nn.Parameter(torch.ones(cfg.hc_mult, cfg.hidden_size), requires_grad=False)
        self.k_weight = nn.Parameter(torch.ones(cfg.hc_mult, cfg.hidden_size), requires_grad=False)
        self.gather = gather
        self.out_dtype = torch.bfloat16  # the reference rounds looked-up rows to bf16

    def forward(self, h: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """h [B, S, hc, D]; rows [B, S, n_cols] rows inside each column's bucket."""
        local = rows[..., self.col0 : self.col1] + self.col_offsets
        emb = lookup_rows(self.table, self.table_scale, local, out_dtype=self.out_dtype)  # [B, S, ncl, hd]
        if self.gather is not None:
            emb = self.gather(emb)  # -> all columns
        out = self.wkv(emb.flatten(-2))
        kv = out[0] if isinstance(out, tuple) else out
        return engram_gate(h, kv, self.q_weight, self.k_weight, self.hc, self.eps)


class DSV41Block(nn.Module):
    def __init__(self, cfg, role, linear, make_moe, tp_size=1, engram=None, max_pos=1 << 20):
        super().__init__()
        self.layer_id, self.hc = role.layer_id, cfg.hc_mult
        self.iters, self.hc_eps, self.eps = cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.rms_norm_eps
        D = cfg.hidden_size
        mix, hcd = (2 + self.hc) * self.hc, self.hc * D
        self.attn = DSV41Attention(cfg, role, linear, tp_size, max_pos)
        self.ffn = make_moe(role.layer_id)
        self.engram = engram
        self.attn_norm = nn.Parameter(torch.ones(D), requires_grad=False)
        self.ffn_norm = nn.Parameter(torch.ones(D), requires_grad=False)
        for s in ("attn", "ffn"):
            setattr(self, f"hc_{s}_fn", nn.Parameter(torch.empty(mix, hcd, dtype=torch.float32), requires_grad=False))
            setattr(self, f"hc_{s}_base", nn.Parameter(torch.empty(mix, dtype=torch.float32), requires_grad=False))
            setattr(self, f"hc_{s}_scale", nn.Parameter(torch.empty(3, dtype=torch.float32), requires_grad=False))

    def _mix(self, h, s):
        return hc_mixes(
            h,
            getattr(self, f"hc_{s}_fn"),
            getattr(self, f"hc_{s}_scale"),
            getattr(self, f"hc_{s}_base"),
            self.hc,
            self.iters,
            self.hc_eps,
            self.eps,
        )

    def forward(self, h, pre_prev, ctx: BatchCtx, caches, shared: Shared, engram_rows=None):
        if self.engram is not None:
            h = self.engram(h, engram_rows)
        pre_a, post_a, comb_a = self._mix(h, "attn")
        x = rms_norm(hc_pre(h, pre_prev), self.attn_norm, self.eps)
        a = self.attn(x, ctx, caches, shared)
        h = hc_post(a, h, post_a, comb_a)
        pre_f, post_f, comb_f = self._mix(h, "ffn")
        x = rms_norm(hc_pre(h, pre_a), self.ffn_norm, self.eps)
        f = self.ffn(x)
        h = hc_post(f, h, post_f, comb_f)
        return h, pre_f


class BlockStep(nn.Module):
    """One block with a TENSORS-ONLY interface, for per-layer HPU graphs.

    A whole-model HPU graph does not compile in reasonable time (one ~40-layer lazy
    graph; Synapse compile time grows faster than linearly), so decode graphs are per
    layer. HPU graph replay does not run Python, so nothing may flow through side
    channels: the cross-layer tensors (the index source's top-k, layer 20's candidate
    blocks) are explicit inputs and outputs, and the cache views are fixed attributes
    (their tensors never move once bound). Parameters are the wrapped block's own."""

    def __init__(self, block: DSV41Block, caches: dict, block_size: int):
        super().__init__()
        self.block = block
        self.caches = caches
        self.block_size = block_size

    def forward(self, h, pre, positions, block_table, req_uid, block_epoch, engram_rows, topk_src, cand_in, ctx=None):
        role = self.block.attn.role
        if ctx is None:  # GroupStep passes one shared ctx under DSV41_DECODE_HOIST
            ctx = BatchCtx(
                positions=positions,
                block_table=block_table,
                req_uid=req_uid,
                block_size=self.block_size,
                block_epoch=block_epoch,
            )
        shared = Shared()
        if role.ratio and not role.is_index_source:
            shared.topk[role.index_source] = topk_src
        if role.uses_candidates:
            shared.cand_blocks = cand_in
        rows = engram_rows if self.block.engram is not None else None
        h, pre = self.block(h, pre, ctx, self.caches, shared, rows)
        empty = h.new_zeros(0)
        topk_out = shared.topk[role.layer_id] if role.is_index_source else empty
        cand_out = shared.cand_blocks if role.is_candidate_source else empty
        return h, pre, topk_out, cand_out


class GroupStep(nn.Module):
    """Consecutive layers in ONE HPU graph (fewer graph launches per decode step).

    Every cross-layer tensor travels explicitly, in a fixed order, so graph replay sees
    the same flat tensor interface each step: the engram rows of the engram layers, the
    top-k of every index source (an empty tensor until its source has run) and the
    candidate blocks. Inside, the layers run their plain BlockStep forward."""

    def __init__(self, steps, src_ids, engram_ids):
        super().__init__()
        self.steps = nn.ModuleList(steps)
        self.src_ids, self.engram_ids = list(src_ids), list(engram_ids)

    def forward(self, h, pre, positions, block_table, req_uid, block_epoch, *flat):
        ne, ns = len(self.engram_ids), len(self.src_ids)
        rows = dict(zip(self.engram_ids, flat[:ne], strict=False))
        topk = dict(zip(self.src_ids, flat[ne : ne + ns], strict=False))
        cand = flat[ne + ns]
        none = h.new_zeros(0)
        ctx = None
        if HOIST:  # one ctx + memo for every layer of this graph (paged.HOIST)
            ctx = BatchCtx(
                positions=positions,
                block_table=block_table,
                req_uid=req_uid,
                block_size=self.steps[0].block_size,
                block_epoch=block_epoch,
                memo={},
            )
        for st in self.steps:
            role = st.block.attn.role
            src = topk[role.index_source] if role.ratio and not role.is_index_source else none
            h, pre, tk, cb = BlockStep.forward(
                st,
                h,
                pre,
                positions,
                block_table,
                req_uid,
                block_epoch,
                rows.get(role.layer_id, none),
                src,
                cand if role.uses_candidates else none,
                ctx,
            )
            if role.is_index_source:
                topk[role.layer_id] = tk
            if role.is_candidate_source:
                cand = cb
        return (h, pre, *[topk[i] for i in self.src_ids], cand)


def run_groups(groups, h, pre, ctx: BatchCtx, engram_rows: dict):
    """Drive GroupSteps, carrying the flat cross-layer state between them."""
    g0 = groups[0]
    none = h.new_zeros(0)
    topks = [none] * len(g0.src_ids)
    cand = none
    rows = [engram_rows[lid] for lid in g0.engram_ids]
    for g in groups:
        out = g(h, pre, ctx.positions, ctx.block_table, ctx.req_uid, ctx.block_epoch, *rows, *topks, cand)
        h, pre, topks, cand = out[0], out[1], list(out[2:-1]), out[-1]
    return h, pre


def run_steps(steps, h, pre, ctx: BatchCtx, engram_rows: dict | None):
    """Drive BlockSteps in order, carrying each index source's top-k and the candidate
    blocks to the layers that consume them."""
    topk, cand = {}, h.new_zeros(0)
    none = h.new_zeros(0)
    for step in steps:
        role = step.block.attn.role  # wrap_in_hpu_graph replaces forward in place, attributes stay
        src = topk.get(role.index_source, none) if role.ratio and not role.is_index_source else none
        rows = engram_rows.get(role.layer_id, none) if engram_rows is not None else none
        h, pre, tk, cb = step(
            h,
            pre,
            ctx.positions,
            ctx.block_table,
            ctx.req_uid,
            ctx.block_epoch,
            rows,
            src,
            cand if role.uses_candidates else none,
        )
        if role.is_index_source:
            topk[role.layer_id] = tk
        if role.is_candidate_source:
            cand = cb
    return h, pre


class DSV41Stack(nn.Module):
    """Embedding -> 40 blocks -> collapse -> final norm. The head is left to the caller
    (vLLM uses a vocab-parallel head + LogitsProcessor)."""

    def __init__(self, cfg, linear, make_moe, make_embed, tp_size=1, engram_factory=None, max_pos=1 << 20):
        super().__init__()
        self.cfg, self.hc = cfg, cfg.hc_mult
        self.roles = csa_roles(cfg)
        self.embed = make_embed()
        self.layers = nn.ModuleList(
            [
                DSV41Block(
                    cfg,
                    r,
                    linear_for(linear, r.layer_id),
                    make_moe,
                    tp_size,
                    engram_factory(r.layer_id) if engram_factory and r.layer_id in cfg.engram_layer_ids else None,
                    max_pos,
                )
                for r in self.roles
            ]
        )
        self.norm = nn.Parameter(torch.ones(cfg.hidden_size), requires_grad=False)
        # Set to htorch.core.mark_step in HPU lazy mode: one graph per layer instead of one
        # for the whole forward, which Synapse compiles in seconds instead of stalling.
        self.mark_step = None

    def forward(self, input_ids, ctx: BatchCtx, caches, engram_rows=None):
        e = self.embed(input_ids)
        h = e.unsqueeze(-2).repeat(*([1] * (e.dim() - 1)), self.hc, 1)
        pre = identity_pre(h, self.hc)
        shared = Shared()
        bound = self.mark_step is not None and inflight.bounded(input_ids.numel())
        for layer in self.layers:
            rows = engram_rows.get(layer.layer_id) if engram_rows is not None else None
            h, pre = layer(h, pre, ctx, caches, shared, rows)
            if self.mark_step is not None:
                self.mark_step()
                if bound:  # prefill run-ahead bound (inflight.py, DSV41_PREFILL_HEADROOM_GIB)
                    inflight.after_layer()
        return rms_norm(hc_pre(h, pre), self.norm, self.cfg.rms_norm_eps)


def linear_for(linear, layer_id):
    """Bind the layer id into the linear factory so it can build per-layer prefixes."""
    return lambda name, i, o, kind, dtype: linear(f"layers.{layer_id}.attn.{name}", i, o, kind, dtype)
