"""DSpark: Mistral Large 4's own block drafter (checkpoint `mtp_dspark.*`) on HPU at TP8.

Math: ref/dspark.py (the oracle, gated against the PR's own drafter code).
In short, at decode point m (x_m the token just sampled, the target has processed
positions 0 .. m-1):
  ctx_t   = target_norm(target_projection([tap_1 | tap_18 | tap_36 | tap_54 | tap_71]_t))   every target token
  ring_j  = [kv_a_norm_j(c) | rope(k_pe)] of wkv_a_j(ctx_t)        draft layer j's row of position t (no input norm)
  block   = [E[x_m], mask_0, mask_1] at positions m, m+1, m+2
  3 dense MLA layers (64 heads, no SSMax, plain YaRN, non-causal over ring rows m-2048 .. m-1 + the block's own
  3 rows), each with a dense SiLU MLP 24576; final norm; base_i = 30 tanh(lm_head(hd_i) / 30)
  RNN Markov head: state 0, prev x_m; per position i: joint = joint_proj([state, W1[prev], hd_i]) + b,
  state = sig(g) state + (1 - sig(g)) tanh(c), d_i = argmax(base_i + W2 tanh(o)), prev = d_i  (greedy)
  -> drafts for positions m+1, m+2, m+3.

TP8 layout (per rank r; D = 6144, Dl = D / 8, V = 204800):
  target_projection  row-parallel over each tap's slice: rank r keeps columns [k D + r Dl, k D + (r+1) Dl) of every
                     tap k ([D, 5 Dl] = 45 MiB), projects its slices of the 5 taps, one all-reduce, then target_norm
                     (replicated). The plan's 0.35 GiB replicated copy costs 0.31 GiB/card more and buys nothing:
                     every rank has the whole residual stream, so the slice is free, and the all-reduce is
                     [rows, D] once per forward.
  wq / wkv_b         by head (8 of 64 per rank); wkv_a + kv_a_norm replicated (every rank holds the same ring rows,
                     as the target's latent); wo row-parallel, all-reduce
  w1 / w3 / w2       column / column / row (3072 of 24576 per rank), all-reduce
  norms, mask rows   replicated
  Markov W1, W2      vocab-parallel like the target's embedding and lm_head (25,600 rows per rank). The greedy chain
                     needs the winner's W1 row next: each rank puts (max, global index, W1[its argmax]) in its own
                     column of a zero [B, tp, 2 + 512] fp32 and one all-reduce gives every rank every candidate, so
                     the argmax and the next position's embedding cost one collective (vllm-gaudi's all_gather marks
                     a step inside the call, which an HPU graph must not contain: DSV 20).
  joint_proj         replicated (fp32 copy, 42 MiB): the RNN cell runs on every rank on the same values
  lm_head, embedding the target's own vocab-parallel tables (fp32 lm_head slice), bound after load
Collectives per draft call: 1 (anchor embedding + W1 row) + 3 layers x 2 + 3 (argmax) = 10 small all-reduces.

Rings, not paged caches (DSV 20/22): a draft at m reads only positions m-2048 .. m-1. Each draft layer keeps one
ring per request slot, [slots x R + 1, 576] bf16 with R = 2048 + 64 rows per slot: row slot * R + pos % R, tagged
(request uid, position) and checked on every read (another request's or a never-written row reads as invalid and is
masked: lower acceptance, never a wrong output token). R >= 2048 + the longest step's rows past the read window
(verify writes up to m + 3), so a step's writes never evict a position the next draft reads. A forward writes only
each row's last 2048 positions (a prompt chunk's earlier ones are never read). Rollback is free, as in DSV: rows are
keyed by position, a draft reads only positions below its anchor, and junk rows of rejected drafts are rewritten
before any draft reads them.

Target taps: `TapBuffer` keeps each rank's slices of the five taps in one persistent [rows, 5 Dl] bf16 buffer,
written in place when the target's stack passes layers 1, 18, 36, 54 and the final pre-norm stream (in place, so
the writes replay inside the target's decode-group HPU graphs; the taps span every group graph, unlike DSV's three
adjacent layers), and `write_context` projects them and writes the rings at the end of the forward.

ML4_DSPARK_FAULT (tests only): no_carry | shared_mask | causal (the oracle's controls, on the HPU path).
"""

from __future__ import annotations

import logging
import os
import time

import torch
import torch.nn.functional as F
from torch import nn

from .paged import TAG_MODE, _counter
from .rope import RopeCache, rope_tail, rotate

logger = logging.getLogger("vllm.mistral_large4_hpu")

PREFIX = "mtp_dspark."
WINDOW = 2048  # draft layers' sliding window (params mtp.ragged_attention)
RING_R = WINDOW + 64  # ring rows per request slot: >= WINDOW + the longest step's rows past the window
LATENT_W = 576  # [c 512 | k_pe 64]
FAULT = os.environ.get("ML4_DSPARK_FAULT", "")
NEG_FLOOR = -1e30


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """vllm ir rms_norm: fp32 statistics, rounded to x.dtype, times the weight (one more rounding)."""
    xf = x.float()
    y = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(x.dtype)
    return (y.float() * w.float()).to(x.dtype)


def _add_rms(x: torch.Tensor, res: torch.Tensor, w: torch.Tensor, eps: float):
    """vllm ir fused_add_rms_norm: s = x + res in fp32; residual = s rounded; the norm reads s itself."""
    s = x.float() + res.float()
    y = (s * torch.rsqrt(s.square().mean(-1, keepdim=True) + eps)).to(x.dtype)
    return (y.float() * w.float()).to(x.dtype), s.to(x.dtype)


# ------------------------------------------------------------------------------------------------ rings
class DraftRing:
    """One draft layer's ring: data [slots * R + 1, 576] bf16, tags [slots * R + 1, 2] int32 (uid, position); the
    last row takes the writes of rows that keep nothing. Allocated once, never moved (HPU graph inputs)."""

    def __init__(self, slots: int, R: int, device, dtype=torch.bfloat16, width: int = LATENT_W):
        self.slots, self.R = slots, R
        n = slots * R + 1
        self.data = torch.zeros(n, width, dtype=dtype, device=device)
        self.tags = torch.full((n, 2), -1, dtype=torch.int32, device=device)
        self.dump = n - 1

    def _rows(self, slot: torch.Tensor, x: torch.Tensor):
        """slot [B], positions x [B, S] -> rows [B, S], addressable [B, S]."""
        ok = (x >= 0) & (slot.view(-1, 1) >= 0) & (slot.view(-1, 1) < self.slots)
        rows = slot.view(-1, 1).clamp(min=0) * self.R + torch.remainder(x.clamp(min=0), self.R)
        return torch.where(ok, rows, self.dump), ok

    def write(self, slot, uid, x, values, keep) -> None:
        rows, ok = self._rows(slot, x)
        keep = keep & ok
        rows = torch.where(keep, rows, self.dump).reshape(-1)
        u = uid.to(torch.int32).view(-1, 1).expand_as(x)
        tag = torch.stack([torch.where(keep, u, -1), torch.where(keep, x.to(torch.int32), -1)], -1)
        self.data.index_copy_(0, rows, values.reshape(-1, values.shape[-1]).to(self.data.dtype))
        self.tags.index_copy_(0, rows, tag.reshape(-1, 2))

    def read(self, slot, uid, x):
        """-> (values [B, S, 576], valid [B, S]): valid iff addressable and the tag is exactly (uid, x)."""
        rows, ok = self._rows(slot, x)
        flat = rows.reshape(-1)
        vals = self.data.index_select(0, flat).view(*x.shape, self.data.shape[-1])
        tag = self.tags.index_select(0, flat).view(*x.shape, 2)
        u = uid.to(torch.int32).view(-1, 1)
        return vals, ok & (tag[..., 0] == u) & (tag[..., 1] == x.to(torch.int32))

    def nbytes(self) -> int:
        return self.data.numel() * self.data.element_size() + self.tags.numel() * 4


class TapBuffer:
    """This rank's slices of the five taps for one forward: n_taps persistent [max_rows, Dl] bf16 base tensors.
    tap(k, x) copies columns [r Dl, (r+1) Dl) of the stream x [B, S, D] into tap k (rows = B * S, in x's order);
    rows(N) is [N, n_taps * Dl] in tap order, the target_projection's column order.
    Written and read with index_copy_ / index_select on the base tensors, never through slices: in serving the taps
    are written by the decode group graphs 0, 2, 4, 6 and 8 and read by graph 8, and a buffer written through a
    slice (buf[:N].copy_) in one HPU graph and read in another fails at capture ("ValidateSyncInputTensors
    tensor_data is empty" in the server; "Empty tensor optional ... Strided Params" on one card, t5i/xgraph_test.py),
    while index_copy_ / index_select, the paged caches' pattern, replay exactly."""

    def __init__(self, max_rows: int, n_taps: int, d_local: int, tp_rank: int, device):
        self.bufs = [torch.zeros(max_rows, d_local, dtype=torch.bfloat16, device=device) for _ in range(n_taps)]
        self.dl, self.r0, self.n, self.max_rows = d_local, tp_rank * d_local, n_taps, max_rows

    def tap(self, k: int, x: torch.Tensor) -> None:
        N = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        assert N <= self.max_rows, (N, self.max_rows)
        idx = torch.arange(N, device=x.device)
        self.bufs[k].index_copy_(0, idx, x.reshape(N, -1)[:, self.r0 : self.r0 + self.dl])

    def rows(self, N: int) -> torch.Tensor:
        idx = torch.arange(N, device=self.bufs[0].device)
        return torch.cat([b.index_select(0, idx) for b in self.bufs], -1)

    def nbytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in self.bufs)


# ------------------------------------------------------------------------------------------------ drafter
class ML4DSpark(nn.Module):
    """The drafter at TP rank r of tp. `all_reduce(x)` sums over the TP group (identity at tp 1; tests inject
    their own). Params hold this rank's shards; load() fills them from checkpoint tensors."""

    def __init__(self, spec, tp_rank: int = 0, tp_size: int = 1, max_pos: int = 1 << 18, all_reduce=None, device="cpu"):
        super().__init__()
        s = spec
        self.spec, self.tp_rank, self.tp_size = s, tp_rank, tp_size
        assert s.n_heads % tp_size == 0 and s.ffn % tp_size == 0 and s.vocab % tp_size == 0 and s.dim % tp_size == 0
        assert s.head_type == "rnn" and s.draft_input == "position_specific" and s.window <= WINDOW
        D, R = s.dim, s.markov_rank
        self.Hl, self.Il, self.Dl = s.n_heads // tp_size, s.ffn // tp_size, s.dim // tp_size
        self.Vl, self.v0 = s.vocab // tp_size, tp_rank * (s.vocab // tp_size)
        self.P, self.Rd, self.C, self.Vd = s.qk_nope, s.qk_rope, s.kv_rank, s.v_dim
        self.nt = len(s.target_layers)
        self.all_reduce = all_reduce or self._default_all_reduce
        bf, f32 = torch.bfloat16, torch.float32

        def P_(*shape, dtype=bf):
            return nn.Parameter(torch.zeros(*shape, dtype=dtype, device=device), requires_grad=False)

        self.tproj = P_(D, self.nt * self.Dl)
        self.target_norm, self.final_norm = P_(D), P_(D)
        self.mask = P_(s.depth - 1, D)
        self.layers = nn.ModuleList()
        for _ in range(s.n_layers):
            L = nn.Module()
            L.wq = P_(self.Hl * (self.P + self.Rd), D)
            L.wkv_a = P_(self.C + self.Rd, D)
            L.kv_a_norm = P_(self.C)
            L.wkv_b = P_(self.Hl * (self.P + self.Vd), self.C)
            L.wo = P_(D, self.Hl * self.Vd)
            L.attn_norm, L.ffn_norm = P_(D), P_(D)
            L.w13 = P_(2 * self.Il, D)
            L.w2 = P_(D, self.Il)
            self.layers.append(L)
        self.mk_w1 = P_(self.Vl, R)
        self.mk_w2 = P_(self.Vl, R, dtype=f32)
        self.joint_w = P_(3 * R, D + 2 * R, dtype=f32)
        self.joint_b = P_(3 * R, dtype=f32)
        self.head = None  # the target's lm_head slice [Vl, D] fp32
        self.embed_w = None  # the target's embedding slice [Vl, D] bf16
        self.rope = RopeCache(max_pos, alpha=None)  # plain YaRN: the draft never uses LeRoPE alpha
        self.rings: list = []
        self.taps: TapBuffer | None = None
        self.slot_buf: torch.Tensor | None = None  # serving: this forward's ring slot per row (spec.before_forward)
        self.capture = True  # serving: taps + ring writes on (spec control key "spec")
        self._graphs: dict = {}
        self.trace: dict | None = None  # tests (eager only): per-layer streams, hd, logits

    def _default_all_reduce(self, x):
        if self.tp_size == 1:
            return x
        from vllm.distributed import tensor_model_parallel_all_reduce

        return tensor_model_parallel_all_reduce(x)

    # ------------------------------------------------------------------ load
    def shard_plan(self) -> dict:
        """checkpoint name -> (param, [(param rows/cols slice, checkpoint slice)] as (dim, start, stop) on both
        sides). The loader gate checks every entry against the header for every rank."""
        s, r = self.spec, self.tp_rank
        D, Dl = s.dim, self.Dl
        plan = {}

        def whole(name, param):
            plan[PREFIX + name] = (param, [(None, None)])

        plan[PREFIX + "backbone.target_projection.weight"] = (
            self.tproj,
            [((1, k * Dl, (k + 1) * Dl), (1, k * D + r * Dl, k * D + (r + 1) * Dl)) for k in range(self.nt)],
        )
        whole("backbone.target_norm.weight", self.target_norm)
        whole("backbone.final_norm.weight", self.final_norm)
        whole("backbone.mask_embedding", self.mask)
        vr = (0, self.v0, self.v0 + self.Vl)
        plan[PREFIX + "autoregressive_head.embedding.weight"] = (self.mk_w1, [(None, vr)])
        plan[PREFIX + "autoregressive_head.output.weight"] = (self.mk_w2, [(None, vr)])
        whole("autoregressive_head.joint_proj.weight", self.joint_w)
        whole("autoregressive_head.joint_proj.bias", self.joint_b)
        hq, hkv, hv = self.Hl * (self.P + self.Rd), self.Hl * (self.P + self.Vd), self.Hl * self.Vd
        for j, L in enumerate(self.layers):
            p = f"backbone.layers.{j}."
            plan[PREFIX + p + "attention.wq.weight"] = (L.wq, [(None, (0, r * hq, (r + 1) * hq))])
            whole(p + "attention.wkv_a_with_mqa.weight", L.wkv_a)
            whole(p + "attention.kv_a_norm.weight", L.kv_a_norm)
            plan[PREFIX + p + "attention.wkv_b.weight"] = (L.wkv_b, [(None, (0, r * hkv, (r + 1) * hkv))])
            plan[PREFIX + p + "attention.wo.weight"] = (L.wo, [(None, (1, r * hv, (r + 1) * hv))])
            whole(p + "attention_norm.weight", L.attn_norm)
            whole(p + "ffn_norm.weight", L.ffn_norm)
            il = self.Il
            plan[PREFIX + p + "feed_forward.w1.weight"] = (L.w13, [((0, 0, il), (0, r * il, (r + 1) * il))])
            plan[PREFIX + p + "feed_forward.w3.weight"] = (L.w13, [((0, il, 2 * il), (0, r * il, (r + 1) * il))])
            plan[PREFIX + p + "feed_forward.w2.weight"] = (L.w2, [(None, (1, r * il, (r + 1) * il))])
        return plan

    @torch.no_grad()
    def load(self, get) -> set:
        """get(name) -> the full checkpoint tensor (CPU). Fills every shard; returns the names loaded."""
        done = set()
        for name, (param, parts) in self.shard_plan().items():
            t = get(name)
            for dst, src in parts:
                v = t if src is None else t.narrow(src[0], src[1], src[2] - src[1])
                d = param.data if dst is None else param.data.narrow(dst[0], dst[1], dst[2] - dst[1])
                assert tuple(d.shape) == tuple(v.shape), (name, tuple(d.shape), tuple(v.shape))
                d.copy_(v.to(d.dtype))
            done.add(name)
        return done

    def bind_target(self, lm_head_slice: torch.Tensor, embed_slice: torch.Tensor) -> None:
        """The target's vocab-parallel lm_head (fp32) and embedding (bf16) slices of this rank."""
        assert lm_head_slice.shape[0] == self.Vl and embed_slice.shape[0] == self.Vl
        self.head, self.embed_w = lm_head_slice, embed_slice

    def make_buffers(self, slots: int, max_rows: int, device) -> None:
        """Rings for `slots` request slots and the tap buffer for `max_rows` rows per forward."""
        self.rings = [DraftRing(slots, RING_R, device) for _ in range(self.spec.n_layers)]
        self.taps = TapBuffer(max_rows, self.nt, self.Dl, self.tp_rank, device)
        self.rope.materialize(self.mk_w1.device)  # the params' device as positions report it (hpu:0)

    # ------------------------------------------------------------------ context side (every target forward)
    def latent_rows(self, L, x, pos):
        """[kv_a_norm(c) | rope(k_pe)] [.., 576] of layer L's wkv_a over x [.., D] at positions pos [..]."""
        kv = F.linear(x, L.wkv_a)
        c = _rms(kv[..., : self.C], L.kv_a_norm, self.spec.eps)
        cos, sin, _ = self.rope.at(pos)
        return torch.cat([c, rotate(kv[..., self.C :], cos, sin)], -1)

    def project_taps(self, N: int) -> torch.Tensor:
        """ctx [N, D] from this forward's tap buffer: this rank's partial projection, all-reduced, target_norm."""
        part = F.linear(self.taps.rows(N), self.tproj)
        return _rms(self.all_reduce(part), self.target_norm, self.spec.eps)

    def write_context(self, pos: torch.Tensor, slot: torch.Tensor, uid: torch.Tensor) -> None:
        """After the target's forward over positions pos [B, S] (-1 padding) of requests in ring slots slot [B]
        (uid [B]): project the taps and write every layer's ring rows for each row's last WINDOW positions."""
        B, S = pos.shape
        ctx = self.project_taps(B * S).view(B, S, -1)
        last = pos.amax(-1, keepdim=True)
        keep = (pos >= 0) & (pos > last - WINDOW)
        for L, ring in zip(self.layers, self.rings):
            ring.write(slot, uid, pos, self.latent_rows(L, ctx, pos), keep)

    # ------------------------------------------------------------------ the draft
    def _embed_anchor(self, tok: torch.Tensor):
        """E[tok] [B, D] bf16 and W1[tok] [B, R] fp32 from the vocab-parallel slices, one all-reduce."""
        loc = tok - self.v0
        mine = (loc >= 0) & (loc < self.Vl)
        li = loc.clamp(0, self.Vl - 1)
        both = torch.cat([self.embed_w[li], self.mk_w1[li]], -1)
        both = torch.where(mine.view(-1, 1), both, torch.zeros_like(both))
        both = self.all_reduce(both)
        return both[:, : self.spec.dim], both[:, self.spec.dim :].float()

    def _argmax_carry(self, scores: torch.Tensor):
        """Vocab-parallel argmax of scores [B, Vl] -> (global token [B], its W1 row [B, R] fp32). Ties go to the
        lowest rank, i.e. the lowest index, as a full-vocab argmax."""
        m, i = scores.max(-1)
        row = self.mk_w1[i].float()
        if self.tp_size == 1:
            return i, row
        B = scores.shape[0]
        pay = torch.cat([m.float().view(B, 1), (i + self.v0).float().view(B, 1), row], -1).view(B, 1, -1)
        mine = (torch.arange(self.tp_size, device=scores.device) == self.tp_rank).view(1, -1, 1)
        allp = self.all_reduce(torch.where(mine, pay, torch.zeros_like(pay)))  # [B, tp, 2 + R]
        best = allp[..., 0].argmax(-1).view(B, 1, 1).expand(B, 1, allp.shape[-1])
        sel = allp.gather(1, best).view(B, -1)
        return sel[:, 1].long(), sel[:, 2:]

    def attention(self, L, a, qpos, ring_vals, ring_valid):
        """a [B, S, D] normed, qpos [B, S]; ring_vals [B, K, 576] / ring_valid [B, K] -> this rank's partial
        [B, S, D] (before the all-reduce). Absorbed form in fp32 (S = 3 queries over K + 3 latent rows)."""
        B, S, _ = a.shape
        cos, sin, _ = self.rope.at(qpos)
        q = F.linear(a, L.wq).view(B, S, self.Hl, self.P + self.Rd)
        q = rope_tail(q, cos.unsqueeze(-2), sin.unsqueeze(-2), self.Rd)
        own = self.latent_rows(L, a, qpos)
        rows = torch.cat([ring_vals, own], 1)
        valid = torch.cat([ring_valid, torch.ones(B, S, dtype=torch.bool, device=a.device)], 1)
        c, kpe = rows[..., : self.C].float(), rows[..., self.C :].float()
        w = L.wkv_b.view(self.Hl, self.P + self.Vd, self.C).float()
        W_UK, W_UV = w[:, : self.P], w[:, self.P :]
        ql = torch.einsum("bshp,hpc->bshc", q[..., : self.P].float(), W_UK)
        lg = torch.einsum("bshc,bkc->bshk", ql, c) + torch.einsum("bshr,bkr->bshk", q[..., self.P :].float(), kpe)
        lg = lg * ((self.P + self.Rd) ** -0.5)
        mask = valid.view(B, 1, 1, -1)
        if FAULT == "causal":
            K = ring_vals.shape[1]
            tri = torch.ones(S, S, dtype=torch.bool, device=a.device).tril()
            mask = torch.cat(
                [valid[:, :K].view(B, 1, 1, K).expand(B, S, 1, K), tri.view(1, S, 1, S).expand(B, S, 1, S)], -1
            )
        lg = lg.masked_fill(~mask, float("-inf"))
        mx = lg.amax(-1, keepdim=True).clamp_min(NEG_FLOOR)
        e = torch.exp(lg - mx)
        ol = torch.einsum("bshk,bkc->bshc", e, c) / e.sum(-1, keepdim=True).clamp_min(1e-30)
        o = torch.einsum("bshc,hvc->bshv", ol, W_UV).to(a.dtype)
        return F.linear(o.flatten(2), L.wo)

    def backbone(self, x0, qpos, ring_vals, ring_valid):
        """x0 [B, S, D] block embeddings -> hd [B, S, D] (final-normed)."""
        eps = self.spec.eps
        x, res = x0, None
        for L, rv in zip(self.layers, ring_vals):
            if res is None:
                a, res = _rms(x, L.attn_norm, eps), x
            else:
                a, res = _add_rms(x, res, L.attn_norm, eps)
            o = self.all_reduce(self.attention(L, a, qpos, rv, ring_valid))
            f, res = _add_rms(o, res, L.ffn_norm, eps)
            gu = F.linear(f, L.w13)
            h = (F.silu(gu[..., : self.Il].float()).to(gu.dtype).float() * gu[..., self.Il :].float()).to(gu.dtype)
            x = self.all_reduce(F.linear(h, L.w2))
            if self.trace is not None:
                self.trace[f"layer{len([k for k in self.trace if k.startswith('layer')])}"] = x.float() + res.float()
        hd, _ = _add_rms(x, res, self.final_norm, eps)
        if self.trace is not None:
            self.trace["hd"] = hd
        return hd

    def draft(self, tok, m, ring_vals, ring_valid):
        """tok [B] anchors at positions m [B]; ring_vals[j] [B, K, 576] rows of positions m-K .. m-1 (valid
        [B, K]) -> greedy drafts [B, depth] for positions m+1 .. m+depth."""
        s = self.spec
        B = tok.shape[0]
        e, row = self._embed_anchor(tok)
        masks = self.mask if FAULT != "shared_mask" else self.mask[:1].expand(s.depth - 1, -1)
        x0 = torch.cat([e.view(B, 1, -1), masks.reshape(1, s.depth - 1, -1).expand(B, -1, -1).to(e.dtype)], 1)
        qpos = m.view(B, 1) + torch.arange(s.depth, device=tok.device).view(1, -1)
        hd = self.backbone(x0, qpos, ring_vals, ring_valid)
        base = F.linear(hd.float(), self.head)
        if s.softcap:
            base = torch.tanh(base / s.softcap) * s.softcap
        R = s.markov_rank
        state = torch.zeros(B, R, dtype=torch.float32, device=tok.device)
        out = []
        for i in range(s.depth):
            if FAULT == "no_carry":
                state = torch.zeros_like(state)
            j = F.linear(torch.cat([state, row, hd[:, i].float()], -1), self.joint_w, self.joint_b)
            gate = torch.sigmoid(j[:, :R])
            state = gate * state + (1.0 - gate) * torch.tanh(j[:, R : 2 * R])
            sc = base[:, i] + F.linear(torch.tanh(j[:, 2 * R :]), self.mk_w2)
            if self.trace is not None:
                self.trace.setdefault("logits", []).append(sc)
            t, row = self._argmax_carry(sc)
            out.append(t)
        return torch.stack(out, 1)

    def read_rings(self, slot, uid, m):
        """Ring rows of positions m-WINDOW .. m-1: per layer [B, WINDOW, 576] and their validity [B, WINDOW] (the
        same positions in every layer; a row counts only if all three layers hold it)."""
        x = m.view(-1, 1) - self.spec.window + torch.arange(self.spec.window, device=m.device).view(1, -1)
        out = [r.read(slot, uid, x) for r in self.rings]
        valid = out[0][1]
        for o in out[1:]:
            valid = valid & o[1]
        if TAG_MODE == "count":  # rows of real positions a draft finds masked (0 unless a hit cut into the window)
            want = (x >= 0) & (slot.view(-1, 1) >= 0)
            _counter(m.device, "ring/draft").add_(torch.stack([(want & ~valid).sum(), want.sum()]).float())
        return [o[0] for o in out], valid

    def draft_step(self, tok, m, slot, uid):
        """One draft call from the rings: tok [B] = x_m, m [B], slot [B], uid [B] -> drafts [B, depth]."""
        vals, valid = self.read_rings(slot, uid, m)
        return self.draft(tok, m, vals, valid)

    # ------------------------------------------------------------------ graphs
    def graph(self, B: int):
        """One HPU graph per batch bucket (DSV 20): rings read + 3 layers + head + RNN chain + argmax collectives."""
        g = self._graphs.get(B)
        if g is None:
            if TAG_MODE == "count":  # the ring counter must exist before any capture: made inside one, it would
                _counter(self.mk_w1.device, "ring/draft")  # be the graph's own buffer (counts lost / reset)
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph

            class _G(nn.Module):
                def __init__(s, d):
                    super().__init__()
                    s.d = d

                def forward(s, tok, m, slot, uid):
                    return s.d.draft_step(tok, m, slot, uid)

            g = self._graphs[B] = wrap_in_hpu_graph(_G(self))
        return g

    def warm(self, max_seqs: int) -> list:
        """Capture every bucket's draft graph (slot -1: nothing read). All TP ranks run the same calls."""
        dev = self.mk_w1.device
        bs, B = [], 1
        while True:
            t = torch.randint(1000, 100000, (B,), generator=torch.Generator().manual_seed(B)).to(dev)
            m = torch.full((B,), 300, dtype=torch.long, device=dev)
            none = torch.full((B,), -1, dtype=torch.long, device=dev)
            self.graph(B)(t, m, none, none.to(torch.int32)).cpu()
            bs.append(B)
            if B >= max_seqs:
                return bs
            B *= 2

    # ------------------------------------------------------------------ accounting
    def weight_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def buffer_bytes(self) -> int:
        n = sum(r.nbytes() for r in self.rings)
        if self.taps is not None:
            n += self.taps.nbytes()
        return n


def _sync():
    import habana_frameworks.torch as ht

    ht.hpu.synchronize()


def median_ms(fn, n: int = 30, warm: int = 5, sync=_sync) -> float:
    ts = []
    for i in range(n + warm):
        sync()
        t = time.perf_counter()
        fn()
        sync()
        if i >= warm:
            ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]
