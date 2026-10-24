# SPDX-License-Identifier: Apache-2.0
"""CSA2 attention for DeepSeek-V4.1, position-exact over paged caches.

Ground truth: DeepSeek's reference (inference/model.py in the checkpoint: Attention, Compressor, Indexer,
select_candidate_blocks, and kernel.py sparse_attn). The reference only runs one-shot prefill (start_pos 0) and
1-token decode, holding state in ring buffers. This module computes the same function for ANY chunking: every
token at absolute position p writes its rows, then every query reads what it may see, in the same forward. Prefill
chunks and decode tokens share one path.

Per layer l (compress ratio r = compress_ratios[l]):
  window     all layers: row p of swa[l] = fake_fp8(rope_p(kv_norm(wkv(x_p))))
  compressor kv sources {2,8,14,20}: entry j of comp[l], idx_k[l]
               r=1: latent_j = norm(wkv_c(x_j))                       (token j)
               r=2: latent_j = norm(bf16(softmax-gated pool of x_2j, x_2j+1)), fp32;
                    even tokens park their fp32 (kv, score) in pend[l]
             comp[l][j]  = fake_fp4_e4m3_16(rope_{j*r}(latent_j))
             idx_k[l][j] = fake_fp4_ue8m0_32(rope_{j*r}(k_norm(wk(latent_j))))
  indexer    index sources {2,8,14,20,24,28,32,36}: top-512 entries of idx_k[S] (S = latest kv source <= l);
             layer 20 also picks candidate blocks that layers 24..36 are restricted to
  reuse      other ratio>0 layers take the latest index source's top-k and the latest kv source's comp cache
  attention  single-KV-head MQA over [128-token window + selected entries] with a per-head sink; P in bf16 before
             P.V as in the kernel; running max floored at -1e30; inverse RoPE on the output; grouped wo_a; wo_b.

Deliberate differences from the reference:
  * On even decode steps the reference's layers 2/8/14 read layer 20's index K through a stale shared pointer
    (model.py, Indexer). Here every index source reads its own kv source's cache, as upstream vLLM does.
  * GEMM input activations are not FP8-fake-quantized (the reference does it for every FP8/FP4 linear); the
    checkpoint's FP8 linears are dequantized to bf16 offline, and bf16 inputs are the higher-precision direction.
  * The online-softmax block order is not emulated: P is rounded to bf16 against the final running max, not the
    per-64-block running max.

Switches (off by default unless noted; each one's off path is the reference form above):
  DSV41_DET_TOPK=1        top-k as a stable descending sort: highest score first, ties to the LOWER index. Index
                          scores are bf16 sums of relu'd dots, so exact ties are common and torch.topk breaks them
                          differently per device, shape and rank; the stable sort makes every rank, chunking and tile
                          pick the same set.
  DSV41_TOPK_SELECT=thresh  prompt rows of at least TOPK_THRESH_MIN entries take select_topk_thresh(): the same set
                          as DSV41_DET_TOPK=1 without sorting the row.
  DSV41_PREFILL_ATTN=1    prompt chunks (S > 1) attend in two parts under one softmax instead of gathering a
                          [S, 128 + 512, 512] key list per query: the window as one band of S + 127 rows read once and
                          scored as a plain GEMM; the compressed part densely for caches of at most PREFILL_DENSE_MAX
                          entries, gathered otherwise. Same function; only the fp32 summation order differs.
  DSV41_INDEX_BF16_DOT=1  the prompt indexer dot as a bf16 GEMM (fp32 accumulation, one bf16 rounding) instead of an
                          fp32 einsum rounded to bf16. q and K are fake-FP4 values, so every product is exact and the
                          scores are bit-equal.
  DSV41_INDEX_DECODE=1    a decode step scores all index heads in one bf16 matmul [B, nh, d] @ [B, d, Lc] instead of
                          nh/4 fp32 einsums and an fp32 copy of the index-K table. Same scores (see
                          _index_scores_decode).
  DSV41_WOA_LINEAR=1      on a decode step with one local output group (TP 8), wo_a as a plain linear instead of the
                          grouped einsum, which transposes the 8 MB weight every step. Bit-equal.
  DSV41_INDEX_STEP_TILES=N  (default 8; 0 = off) a prefill's indexer marks a lazy step after every N query tiles.
                          Unrolled in one graph, every tile's intermediates were live at once: a full-stack prefill
                          (prompt_logprobs) past 64K context needed ~9.5 GiB above live memory in one layer. Same ops
                          in the same order, so outputs are bit-identical.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .numerics import (
    LEAN,
    fake_fp4_e4m3,
    fake_fp4_ue8m0,
    fake_fp8_ue8m0,
    rms_norm,
    rope_inv_freq,
    rope_tables,
    rope_tail,
)
from .paged import BatchCtx

NEG_FLOOR = -1e30
DET_TOPK = os.environ.get("DSV41_DET_TOPK", "0") == "1"
TOPK_SELECT = os.environ.get("DSV41_TOPK_SELECT", "sort")
# shorter rows keep the sort: the threshold select has a roughly fixed compaction cost (k x 256 per row) and only
# wins on long rows (one card, 4096 rows: 8K entries 11 vs 65 ms, 64K 126 vs 85, 128K 286 vs 109)
TOPK_THRESH_MIN = 65536
PREFILL_ATTN = os.environ.get("DSV41_PREFILL_ATTN", "0") == "1"
PREFILL_DENSE_MAX = 8192
INDEX_BF16_DOT = os.environ.get("DSV41_INDEX_BF16_DOT", "0") == "1"
INDEX_DECODE = os.environ.get("DSV41_INDEX_DECODE", "0") == "1"
WOA_LINEAR = os.environ.get("DSV41_WOA_LINEAR", "0") == "1"
INDEX_STEP_TILES = int(os.environ.get("DSV41_INDEX_STEP_TILES", "8"))
_MARK_STEP: list = []


def _mark_step() -> None:
    """htorch.core.mark_step in HPU lazy mode, else nothing."""
    if not _MARK_STEP:
        try:
            import habana_frameworks.torch as htorch

            _MARK_STEP.append(htorch.core.mark_step if htorch.utils.internal.is_lazy() else None)
        except ImportError:
            _MARK_STEP.append(None)
    if _MARK_STEP[0] is not None:
        _MARK_STEP[0]()


def _index_scores_decode(q: torch.Tensor, K: torch.Tensor, w: torch.Tensor, hchunk: int) -> torch.Tensor:
    """DSV41_INDEX_DECODE: index scores of one decode token per row, q [B, 1, nh, d], K [B, Lc, d], w [B, 1, nh] ->
    fp32 [B, 1, Lc]. q and K are fake-FP4 values, so the bf16 GEMM (fp32 accumulation, one bf16 rounding) equals
    the fp32 dot rounded to bf16. The served loop computes relu(d) * w in fp32 (the graph compiler fuses the product
    into the head sum and drops its bf16 rounding); this writes that out with the same per-4-head sums added in the
    same order."""
    B, _, nh, ihd = q.shape
    d = torch.matmul(q.reshape(B, nh, ihd), K.transpose(1, 2)).view(B, 1, nh, K.shape[1])
    c = (d.relu().float() * w.float().unsqueeze(-1)).unflatten(2, (nh // hchunk, hchunk)).sum(3)
    score = torch.zeros_like(c[:, :, 0])
    for i in range(nh // hchunk):
        score = score + c[:, :, i]
    return score


def select_topk_thresh(score: torch.Tensor, k: int, block: int = 256):
    """DSV41_TOPK_SELECT=thresh: the same k entries as the stable descending sort (highest first, ties to the
    lower index), without sorting the rows. Returns (values, indices) with the indices in ASCENDING order, not
    by value (callers use the set: a key mask or a gather). score: bf16 [..., L], L >= k.
      1. per row, the k-th largest value v: binary search over the 2^16 bf16 bit patterns in float order
         (16 compare-and-count passes from -inf up; compares are float compares, so -0 ties +0 as in the sort);
      2. take every entry > v, and the lowest-index entries == v until k (a running count of the ties);
      3. compaction without a scatter over the row: per `block` entries a count, its running sum, then output
         slot j finds its block (count of block ends <= j) and its place inside it (running count in the block).
    Counts and positions run in fp32 (exact below 2^24), which Gaudi's TPC does far faster than int32."""
    assert score.dtype == torch.bfloat16, score.dtype
    *lead, L = score.shape
    pad = -L % block
    if pad:  # -inf padding ranks after every real entry, -inf ones included (higher index)
        score = torch.nn.functional.pad(score, (0, pad), value=float("-inf"))
    Lp = L + pad
    dev = score.device
    # keys -32768..-32642 are negative NaN patterns; -32641 is -inf, which every entry reaches (no NaN scores)
    lo = torch.full((*lead, 1), -32641, dtype=torch.int32, device=dev)
    hi = torch.full((*lead, 1), 32767, dtype=torch.int32, device=dev)
    kf = float(k)

    def val(key):  # float-order key -> bf16 value with that bit pattern
        bits = torch.where(key >= 0, key, key ^ 0x7FFF)
        return bits.to(torch.int16).view(torch.bfloat16)

    for _ in range(16):  # invariant: count(score >= val(lo)) >= k > count(score >= val(hi + 1))
        mid = (lo + hi + 1) >> 1
        ok = (score >= val(mid)).sum(-1, keepdim=True, dtype=torch.float32) >= kf
        lo = torch.where(ok, mid, lo)
        hi = torch.where(ok, hi, mid - 1)
    v = val(lo)
    gt = score > v
    eq = score == v
    need = kf - gt.sum(-1, keepdim=True, dtype=torch.float32)
    take = (gt | (eq & (torch.cumsum(eq.to(torch.float32), -1) <= need))).to(torch.float32)
    nb = Lp // block
    tb = take.view(*lead, nb, block)
    ends = torch.cumsum(tb.sum(-1), -1)  # [..., nb] inclusive
    j = torch.arange(k, device=dev, dtype=torch.float32)
    b = (ends.unsqueeze(-2) <= j.view(k, 1)).sum(-1, dtype=torch.float32)  # [..., k] block of slot j
    bl = b.long()
    start = torch.gather(ends, -1, (bl - 1).clamp(min=0)) * (bl > 0)  # trues before block b
    r = j - start  # place inside the block
    rows = torch.gather(tb, -2, bl.unsqueeze(-1).expand(*lead, k, block))  # [..., k, block]
    inner = (torch.cumsum(rows, -1) <= r.unsqueeze(-1)).sum(-1, dtype=torch.float32)
    idx = (b * block + inner).long()
    return torch.gather(score, -1, idx), idx


def select_topk(score: torch.Tensor, k: int, prompt: bool = False):
    if prompt and TOPK_SELECT == "thresh" and score.shape[-1] >= TOPK_THRESH_MIN:
        if score.dtype != torch.bfloat16:  # candidate-block maxima: fp32 copies of bf16 scores (+-inf), exact
            v, i = select_topk_thresh(score.to(torch.bfloat16), k)
            return v.to(score.dtype), i
        return select_topk_thresh(score, k)
    if DET_TOPK:
        v, i = torch.sort(score, dim=-1, descending=True, stable=True)
        return v[..., :k], i[..., :k]
    t = score.topk(k, dim=-1)
    return t.values, t.indices


# --------------------------------------------------------------------- topology
@dataclass(frozen=True)
class LayerRole:
    layer_id: int
    ratio: int  # 0: window only
    is_kv_source: bool
    is_index_source: bool
    kv_source: int | None  # layer whose comp/idx_k caches this layer reads
    index_source: int | None  # layer whose top-k this layer uses
    is_candidate_source: bool
    uses_candidates: bool


def csa_roles(cfg) -> list[LayerRole]:
    """CSA2 modes from config alone, as the reference derives them (model.py:653-661).
    Full = kv source (compresses + owns index K), Reindex = index source that is not a
    kv source, Reuse = every other ratio>0 layer."""
    n = cfg.num_hidden_layers
    ratios = list(cfg.compress_ratios)[:n]
    kv_src, idx_src = set(cfg.kv_source_layer_ids), set(cfg.index_source_layer_ids)
    cand = getattr(cfg, "candidate_source_layer_id", -1)
    roles, last_kv, last_idx = [], None, None
    for lid in range(n):
        r = ratios[lid]
        is_kv = r > 0 and lid in kv_src
        is_idx = r > 0 and lid in idx_src
        if is_kv:
            last_kv = lid
        if is_idx:
            last_idx = lid
        if r > 0:
            assert last_kv is not None and last_idx is not None, f"layer {lid} has no source"
            assert ratios[last_kv] == r, f"layer {lid} ratio {r} != its kv source {last_kv}"
        roles.append(
            LayerRole(
                lid,
                r,
                is_kv,
                is_idx,
                last_kv if r > 0 else None,
                last_idx if r > 0 else None,
                lid == cand,
                is_idx and 0 <= cand < lid,
            )
        )
    return roles


# -------------------------------------------------------------------- per-forward
@dataclass
class Shared:
    """What layers hand down the stack within ONE forward (never a module buffer,
    so nothing persists between forwards or leaks into an HPU graph's state)."""

    topk: dict = field(default_factory=dict)  # index source -> [B, S, K] entry ids (-1 none)
    cand_blocks: torch.Tensor | None = None  # [B, S, n_blocks] bool, from the candidate source


def _linear(mod, x):
    out = mod(x)
    return out[0] if isinstance(out, tuple) else out


_ROPE_TABLES: dict = {}


class RopeCache(nn.Module):
    """cos/sin tables built on CPU in fp32 (HPU sin/cos is wrong above 196608) and
    shared by every layer with the same rope parameters -- V4.1 has two sets
    (compressed layers: theta 160000 + YaRN; window-only layers: theta 1e4). They are
    moved to the device once, by materialize(), never inside a forward that an HPU
    graph could record."""

    def __init__(self, dim, max_pos, base, original_seq_len, factor, beta_fast, beta_slow):
        super().__init__()
        self.key = (dim, max_pos, float(base), int(original_seq_len), float(factor), float(beta_fast), float(beta_slow))
        if (self.key, "cpu") not in _ROPE_TABLES:
            _ROPE_TABLES[(self.key, "cpu")] = rope_tables(
                max_pos, rope_inv_freq(dim, base, original_seq_len, factor, beta_fast, beta_slow)
            )

    def materialize(self, device) -> None:
        k = (self.key, str(device))
        if k not in _ROPE_TABLES:
            cos, sin = _ROPE_TABLES[(self.key, "cpu")]
            _ROPE_TABLES[k] = (cos.to(device), sin.to(device))

    def at(self, pos: torch.Tensor, ctx=None, name=None) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin at pos; with a DSV41_DECODE_HOIST memo on ctx and a name for pos, once per forward per table."""
        if ctx is not None and name is not None and ctx.memo is not None:
            return ctx.shared(("rope", self.key, name), lambda: self.at(pos))
        cos, sin = _ROPE_TABLES.get((self.key, str(pos.device))) or _ROPE_TABLES[(self.key, "cpu")]
        p = pos.clamp(min=0)
        return cos[p], sin[p]


def sparse_attention(q, keys, valid, sink, scale):
    """q [B,S,H,D] bf16; keys [B,S,N,D] (K = V); valid [B,S,N]; sink [H] fp32.
    The reference sparse_attn: fp32 scores, max floored at -1e30, exp in fp32,
    P rounded to bf16 for P.V, fp32 denominator plus exp(sink - max)."""
    s = torch.einsum("bshd,bsnd->bshn", q.float(), keys.float()) * scale
    s = s.masked_fill(~valid.unsqueeze(2), float("-inf"))
    m = s.amax(-1, keepdim=True).clamp_min(NEG_FLOOR)
    p = torch.exp(s - m)
    denom = p.sum(-1, keepdim=True) + torch.exp(sink.float().view(1, 1, -1, 1) - m)
    o = torch.einsum("bshn,bsnd->bshd", p.to(q.dtype).float(), keys.float()) / denom
    return o.to(q.dtype)


class DSV41Attention(nn.Module):
    """One CSA2 attention layer. Linears come from `linear(name, in, out, kind, dtype)`
    so the same code runs with plain modules in tests and vLLM parallel layers in
    serving: kind in {"replicated", "column", "row"}."""

    def __init__(self, cfg, role: LayerRole, linear, tp_size: int = 1, max_pos: int = 1 << 20, q_tile: int = 512):
        super().__init__()
        self.role, self.cfg, self.q_tile = role, cfg, q_tile
        D, H, hd = cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim
        self.rd = cfg.qk_rope_head_dim
        self.eps = cfg.rms_norm_eps
        self.window = cfg.sliding_window
        assert H % tp_size == 0 and cfg.o_groups % tp_size == 0
        self.n_local_heads = H // tp_size
        self.n_local_groups = cfg.o_groups // tp_size
        self.head_dim, self.o_lora_rank = hd, cfg.o_lora_rank
        self.scale = hd**-0.5

        self.wq_a = linear("wq_a", D, cfg.q_lora_rank, "replicated", None)
        self.q_norm = nn.Parameter(torch.ones(cfg.q_lora_rank), requires_grad=False)
        self.wq_b = linear("wq_b", cfg.q_lora_rank, H * hd, "column", None)
        self.wkv = linear("wkv", D, hd, "replicated", None)
        self.kv_norm = nn.Parameter(torch.ones(hd), requires_grad=False)
        # [o_groups * o_lora_rank, H * hd / o_groups], used as a grouped einsum, never called
        self.wo_a = nn.Parameter(
            torch.empty(self.n_local_groups * cfg.o_lora_rank, H * hd // cfg.o_groups, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.wo_b = linear("wo_b", cfg.o_groups * cfg.o_lora_rank, D, "row", None)
        self.attn_sink = nn.Parameter(torch.zeros(self.n_local_heads, dtype=torch.float32), requires_grad=False)

        r = role.ratio
        rs = getattr(cfg, "rope_scaling", None) or {}
        if r:
            self.rope = RopeCache(
                self.rd,
                max_pos,
                cfg.compress_rope_theta,
                rs.get("original_max_position_embeddings", 0),
                rs.get("factor", 1),
                rs.get("beta_fast", 32),
                rs.get("beta_slow", 1),
            )
        else:  # window-only layers: plain theta, no YaRN (model.py:685-687)
            self.rope = RopeCache(self.rd, max_pos, cfg.rope_theta, 0, 1, 32, 1)

        if role.is_kv_source:
            # r > 1 pools in fp32, so those weights are fp32 (model.py:444-448)
            wdt = torch.float32 if r > 1 else torch.bfloat16
            self.c_wkv = linear("compressor.wkv", D, hd, "replicated", wdt)
            if r > 1:
                self.c_wgate = linear("compressor.wgate", D, hd, "replicated", wdt)
            self.c_norm = nn.Parameter(torch.ones(hd), requires_grad=False)
            self.i_wk = linear("indexer.wk", hd, cfg.index_head_dim, "replicated", torch.bfloat16)
            self.i_k_norm = nn.Parameter(torch.ones(cfg.index_head_dim), requires_grad=False)
        if role.is_index_source:
            # replicated on every rank (as upstream), so no [S, L] score all-reduce
            self.i_wq_b = linear(
                "indexer.wq_b", cfg.q_lora_rank, cfg.index_n_heads * cfg.index_head_dim, "replicated", None
            )
            self.i_weights_proj = linear("indexer.weights_proj", D, cfg.index_n_heads, "replicated", torch.bfloat16)

    # ------------------------------------------------------------------ pieces
    def _window_rows(self, x, pos, ctx=None):
        kv = rms_norm(_linear(self.wkv, x), self.kv_norm, self.eps)
        cos, sin = self.rope.at(pos, ctx, "pos")
        return fake_fp8_ue8m0(rope_tail(kv, cos, sin, self.rd))

    def _compress(self, x, ctx: BatchCtx, caches):
        """Write this kv source's completed entries (comp, idx_k) for the tokens in x."""
        r, pos, valid = self.role.ratio, ctx.positions, ctx.valid
        hd = self.head_dim
        if r == 1:
            latent = rms_norm(_linear(self.c_wkv, x), self.c_norm, self.eps)
            entry, done = pos, valid
        else:
            assert r == 2, f"compress ratio {r} not implemented"
            x32 = x.float()
            kv, sc = _linear(self.c_wkv, x32), _linear(self.c_wgate, x32)
            even = valid & (torch.remainder(pos, 2) == 0)
            pend = caches["pend"]
            # the fp32 (kv, score) pair is bit-cast into the cache's dtype, losslessly
            pend.write(ctx, pos, torch.cat([kv, sc], -1).contiguous().view(pend.data.dtype), even, key="pos")
            odd = valid & (torch.remainder(pos, 2) == 1)
            prev, got = pend.read(ctx, pos - 1, odd, key="pos-1")
            prev = prev.contiguous().view(torch.float32)
            kv0, sc0 = prev[..., :hd], prev[..., hd:]
            w = torch.softmax(torch.stack([sc0, sc], dim=-2), dim=-2)
            pooled = (torch.stack([kv0, kv], dim=-2) * w).sum(-2)
            latent = rms_norm(pooled.to(x.dtype), self.c_norm, self.eps)
            entry, done = torch.div(pos, 2, rounding_mode="floor"), odd & got
        cos, sin = self.rope.at(entry * r, ctx, f"entry{r}")  # an entry stands for its group's first token
        k = rms_norm(_linear(self.i_wk, latent), self.i_k_norm, self.eps)
        caches["idx_k"].write(ctx, entry, fake_fp4_ue8m0(rope_tail(k, cos, sin, self.rd), 32), done, key=f"entry{r}")
        caches["comp"].write(ctx, entry, fake_fp4_e4m3(rope_tail(latent, cos, sin, self.rd), 16), done, key=f"entry{r}")

    def _index(self, x, qr, ctx: BatchCtx, caches_all, shared: Shared):
        """Top-512 compressed entries per query (Indexer.forward, model.py:527-580)."""
        role, cfg = self.role, self.cfg
        r, S_src = role.ratio, role.kv_source
        pos, B, S = ctx.positions, *ctx.positions.shape
        Lc = ctx.max_len // r
        nh, ihd = cfg.index_n_heads, cfg.index_head_dim

        def _ent():
            ent = torch.arange(Lc, device=pos.device).view(1, Lc).expand(B, Lc)
            # every entry this batch could reach; which ones each query may see is below
            reach = torch.div(pos.clamp(min=0) + 1, r, rounding_mode="floor")  # [B, S]
            return ent, reach, reach.amax(-1, keepdim=True)

        ent, reach, kmax = ctx.shared(("ent", r, Lc), _ent)
        K, kvalid = caches_all[S_src]["idx_k"].read(
            ctx, ent, ent < kmax, key=("ent", r, Lc), share=True
        )  # [B, Lc, 128]

        cos, sin = self.rope.at(pos, ctx, "pos")
        q = _linear(self.i_wq_b, qr).unflatten(-1, (nh, ihd))
        q = fake_fp4_ue8m0(rope_tail(q, cos.unsqueeze(-2), sin.unsqueeze(-2), self.rd), 32)
        w = _linear(self.i_weights_proj, x) * (ihd**-0.5 * nh**-0.5)  # [B, S, nh] bf16

        k_out = min(cfg.index_topk, Lc)
        # DSV41_LEAN_OPS: a single query tile (decode) writes its results directly, no empty/zeros + slice copy
        one_tile = LEAN and max(1, min(self.q_tile, (64 << 20) // max(1, B * 4 * Lc))) >= S
        topk = None if one_tile else torch.empty(B, S, k_out, dtype=torch.long, device=pos.device)
        cand_blocks = None
        if role.is_candidate_source and not one_tile:
            nblk = -(-Lc // cfg.candidate_block_size)
            cand_blocks = torch.zeros(B, S, nblk, dtype=torch.bool, device=pos.device)
        # Tile queries so the fp32 per-head dot block [B, tile, hchunk, Lc] stays ~256 MB.
        hchunk = 4
        tile = max(1, min(self.q_tile, (64 << 20) // max(1, B * hchunk * Lc)))
        bf16_dot = INDEX_BF16_DOT and S > 1
        dec_dot = INDEX_DECODE and S == 1
        Kf = K if (bf16_dot or dec_dot) else K.float()
        n_done = 0
        for s0 in range(0, S, tile):
            s1 = min(S, s0 + tile)
            # The reference's bf16 einsum is a GEMM with fp32 accumulation and ONE bf16
            # rounding per dot; the HPU's bf16 einsum is not (41% of scores differed), so
            # the dot runs in fp32 and is rounded here. relu and the weight product stay
            # bf16; heads are summed with fp32 accumulation and one bf16 rounding.
            if dec_dot:
                score = _index_scores_decode(q, K, w, hchunk)
            else:
                score = torch.zeros(B, s1 - s0, Lc, dtype=torch.float32, device=pos.device)
            for h0 in range(0, 0 if dec_dot else nh, hchunk):
                if bf16_dot:  # [B, tile*hc, D] @ [B, D, Lc]: a plain matmul (einsum permutes K)
                    qq = q[:, s0:s1, h0 : h0 + hchunk]
                    d = torch.matmul(qq.reshape(B, -1, ihd), Kf.transpose(1, 2)).view(B, s1 - s0, -1, Kf.shape[1])
                else:
                    d = torch.einsum("bshd,btd->bsht", q[:, s0:s1, h0 : h0 + hchunk].float(), Kf).to(q.dtype)
                score += (d.relu() * w[:, s0:s1, h0 : h0 + hchunk].unsqueeze(-1)).float().sum(2)
            score = score.to(q.dtype)
            see = (ent.unsqueeze(1) < reach[:, s0:s1].unsqueeze(-1)) & kvalid.unsqueeze(1)
            score = score.masked_fill(~see, float("-inf"))
            if role.is_candidate_source:
                cb = self._candidate_blocks(score, reach[:, s0:s1], prompt=S > 1)
                if one_tile:
                    cand_blocks = cb
                else:
                    cand_blocks[:, s0:s1] = cb
            elif role.uses_candidates:
                cb = shared.cand_blocks[:, s0:s1]
                allowed = cb.repeat_interleave(cfg.candidate_block_size, dim=-1)[..., :Lc]
                score = score.masked_fill(~allowed, float("-inf"))
            idx = select_topk(score, k_out, prompt=S > 1)[1]
            ok = torch.gather(see, 2, idx)
            if role.uses_candidates:
                ok = ok & torch.gather(allowed, 2, idx)
            if one_tile:
                topk = torch.where(ok, idx, -1)
            else:
                topk[:, s0:s1] = torch.where(ok, idx, torch.full_like(idx, -1))
                n_done += 1
                if INDEX_STEP_TILES and S > 1 and s1 < S and n_done % INDEX_STEP_TILES == 0:
                    _mark_step()  # DSV41_INDEX_STEP_TILES: bound the graph's live intermediates
        if role.is_candidate_source:
            shared.cand_blocks = cand_blocks
        shared.topk[role.layer_id] = topk

    def _candidate_blocks(self, score, reach, prompt: bool = False):
        """select_candidate_blocks (model.py:583-610) on a masked score tile."""
        bs, n_top = self.cfg.candidate_block_size, self.cfg.candidate_topk_blocks
        L = score.shape[-1]
        pad = -L % bs
        blk = F.pad(score.float(), (0, pad), value=float("-inf")).unflatten(-1, (-1, bs)).amax(-1)
        nblk = blk.shape[-1]
        last = torch.div(reach - 1, bs, rounding_mode="floor").unsqueeze(-1)  # newest block, pinned
        blk = blk.masked_fill(torch.arange(nblk, device=score.device) == last, float("inf"))
        tv, ti = select_topk(blk, min(n_top, nblk), prompt)
        keep = torch.zeros_like(blk, dtype=torch.bool).scatter_(-1, ti, tv > float("-inf"))
        return keep

    # ----------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, ctx: BatchCtx, caches_all: dict, shared: Shared) -> torch.Tensor:
        role = self.role
        pos, valid = ctx.positions, ctx.valid
        B, S, _ = x.shape
        caches = caches_all[role.layer_id]

        qr = rms_norm(_linear(self.wq_a, x), self.q_norm, self.eps)
        cos, sin = self.rope.at(pos, ctx, "pos")
        q = _linear(self.wq_b, qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = rope_tail(q, cos.unsqueeze(-2), sin.unsqueeze(-2), self.rd)

        # 1. writes: this chunk's window rows, then (kv sources) its completed entries
        caches["swa"].write(ctx, pos, self._window_rows(x, pos, ctx), valid, key="pos")
        if role.is_kv_source:
            self._compress(x, ctx, caches)
        # 2. index sources pick entries (after the writes, so same-chunk entries count)
        if role.is_index_source:
            self._index(x, qr, ctx, caches_all, shared)

        # 3. attention, tiled over queries
        if PREFILL_ATTN and S > 1:
            out = self._attend_prefill(q, ctx, caches, caches_all, shared)
            return self._out_proj(out, cos, sin, B, S)
        one_tile = LEAN and self.q_tile >= S
        out = None if one_tile else torch.empty(B, S, self.n_local_heads, self.head_dim, dtype=q.dtype, device=x.device)
        for s0 in range(0, S, self.q_tile):
            s1 = min(S, s0 + self.q_tile)

            def _win(s0=s0, s1=s1):
                woff = torch.arange(-(self.window - 1), 1, device=x.device)  # [W]
                wx = pos[:, s0:s1].unsqueeze(-1) + woff  # [B, St, W]
                return wx, (wx >= 0) & valid[:, s0:s1].unsqueeze(-1)

            wx, wwant = ctx.shared(("win", s0, s1), _win)
            wk, wvalid = caches["swa"].read(ctx, wx, wwant, key=("win", s0, s1))
            keys, kvalid = wk, wvalid
            if role.ratio:
                idx = shared.topk[role.index_source][:, s0:s1]  # [B, St, K]
                ck, cvalid = caches_all[role.kv_source]["comp"].read(
                    ctx, idx.clamp(min=0), idx >= 0, key=("topk", role.index_source, s0, s1), share=True
                )
                keys = torch.cat([wk, ck], dim=2)
                kvalid = torch.cat([wvalid, cvalid], dim=2)
            o_t = sparse_attention(q[:, s0:s1], keys, kvalid, self.attn_sink, self.scale)
            if one_tile:
                out = o_t
            else:
                out[:, s0:s1] = o_t
        return self._out_proj(out, cos, sin, B, S)

    def _out_proj(self, out, cos, sin, B, S):
        o = rope_tail(out, cos.unsqueeze(-2), sin.unsqueeze(-2), self.rd, inverse=True)
        if WOA_LINEAR and S == 1 and self.n_local_groups == 1:
            o = F.linear(o.reshape(B, S, -1), self.wo_a)
        else:
            o = o.view(B, S, self.n_local_groups, -1)
            wo_a = self.wo_a.view(self.n_local_groups, self.o_lora_rank, -1)
            o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        o = _linear(self.wo_b, o.flatten(2))
        return o

    def _attend_prefill(self, q, ctx: BatchCtx, caches, caches_all, shared: Shared):
        """DSV41_PREFILL_ATTN: sparse_attention() over [window rows + selected entries] for a prompt chunk,
        as a banded window GEMM plus a dense (short caches) or gathered (long caches) compressed part."""
        role, W = self.role, self.window
        pos = ctx.positions
        B, S, Hn, Dh = q.shape
        dev = q.device
        out = torch.empty_like(q)
        dense = None
        if role.ratio:
            Lc = ctx.max_len // role.ratio
            if Lc <= PREFILL_DENSE_MAX:
                ent = torch.arange(Lc, device=dev).view(1, Lc).expand(B, Lc)
                reach = torch.div(pos.clamp(min=0) + 1, role.ratio, rounding_mode="floor")
                ck_all, cv_all = caches_all[role.kv_source]["comp"].read(ctx, ent, ent < reach.amax(-1, keepdim=True))
                dense = (ck_all.float(), cv_all, Lc)
        for s0 in range(0, S, self.q_tile):
            s1 = min(S, s0 + self.q_tile)
            St = s1 - s0
            p = pos[:, s0:s1]  # [B, St]
            qf = q[:, s0:s1].float()
            # window: band of positions p0-(W-1) .. p0+St-1 (prompt rows are contiguous), each row read once
            bp = p[:, :1] + torch.arange(-(W - 1), St, device=dev).view(1, -1)  # [B, Nb]
            # want only rows up to the tile's last real position: a padded tile's band reaches past the row's end
            # (never written); those rows are causally masked anyway, but asking for them would count as tag
            # violations in DSV41_TAG_CHECK=count. Same values.
            bk, bv = caches["swa"].read(ctx, bp, (bp >= 0) & (bp <= p.amax(-1, keepdim=True)))
            pi, bj = p.unsqueeze(-1), bp.unsqueeze(1)  # [B, St, 1], [B, 1, Nb]
            mw = bv.unsqueeze(1) & (bj <= pi) & (bj > pi - W) & (pi >= 0)  # [B, St, Nb]
            # plain (batched) matmuls, not einsum: einsum's lowering permutes the key tensors (~2x the tile time)
            bkf = bk.float()
            sw = torch.matmul(qf.view(B, St * Hn, Dh), bkf.transpose(1, 2)).view(B, St, Hn, -1) * self.scale
            sw = sw.masked_fill(~mw.unsqueeze(2), float("-inf"))
            m = sw.amax(-1, keepdim=True)
            sc = ckf = None
            if role.ratio:
                idx = shared.topk[role.index_source][:, s0:s1]  # [B, St, K]
                if dense is not None:
                    ckf, cv_all, Lc = dense
                    sel = (
                        torch.zeros(B, St, Lc, dtype=torch.float32, device=dev).scatter_add_(
                            2, idx.clamp(min=0), (idx >= 0).float()
                        )
                        > 0
                    )
                    mc = sel & cv_all.unsqueeze(1)
                    sc = torch.matmul(qf.view(B, St * Hn, Dh), ckf.transpose(1, 2)).view(B, St, Hn, -1) * self.scale
                else:
                    ck, mc = caches_all[role.kv_source]["comp"].read(ctx, idx.clamp(min=0), idx >= 0)
                    ckf = ck.float()
                    sc = torch.matmul(qf, ckf.transpose(-1, -2)) * self.scale
                sc = sc.masked_fill(~mc.unsqueeze(2), float("-inf"))
                m = torch.maximum(m, sc.amax(-1, keepdim=True))
            m = m.clamp_min(NEG_FLOOR)
            pw = torch.exp(sw - m)
            denom = pw.sum(-1, keepdim=True) + torch.exp(self.attn_sink.float().view(1, 1, -1, 1) - m)
            o = torch.matmul(pw.to(q.dtype).float().view(B, St * Hn, -1), bkf).view(B, St, Hn, Dh)
            if sc is not None:
                pc = torch.exp(sc - m)
                denom = denom + pc.sum(-1, keepdim=True)
                pcf = pc.to(q.dtype).float()
                if dense is not None:
                    o = o + torch.matmul(pcf.view(B, St * Hn, -1), ckf).view(B, St, Hn, Dh)
                else:
                    o = o + torch.matmul(pcf, ckf)
            out[:, s0:s1] = (o / denom).to(q.dtype)
        return out
