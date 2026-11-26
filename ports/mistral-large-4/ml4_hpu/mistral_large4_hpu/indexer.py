"""The DSA lightning indexer of Mistral Large 4, replicated on every rank ().

Ground truth: the PR's MistralIndexer + SparseAttnIndexer (FP8 path) and HF's MistralLarge4Indexer for the meaning:
  qi  = wq_b(LayerNorm_q(a))                 [32, 128]   a = the RMSNorm'ed attention input (LayerNorm eps 1e-6)
  qi *= lambda_idx(pos)                       SSMax with the indexer's own s, b [32], before RoPE
  ki  = LayerNorm_k(wk a)                     [128]
  RoPE layers rotate dims [0:64] of qi and ki (interleaved, [rope | nope]); NoPE layers (match_nope) do not rotate
  qi, ki -> FP8 e4m3fn, one power-of-two scale per 128 (per token and head for q, per token for k)
  w   = weights_proj(a) * 128^-0.5 * 32^-0.5
  I[t, s] = sum_h w[t, h] * ReLU(qi[t, h] . ki[s])      s <= t;   topk(t) = the 2048 largest I[t, s]

On the HPU the FP8 values are fake-quantized in bf16 (numerics.fake_fp8_ue8m0: the e4m3fn grid, exact in bf16, no
device FP8 cast) and the index-K cache stores them as bf16 (256 B/token/layer; U8 + scale is). The PR folds
the q scale into w; with the value q8 * s_q in place of q8 that is the same product. Each head's dot runs in fp32
(fake-FP8 products are exact in fp32), ReLU and the w product in fp32, heads summed in fp32 (DeepGEMM's fp8 MQA
logits are fp32 too); the score row is rounded to bf16 once for the threshold top-k (DSV's select_topk_thresh: the
same set as a stable descending sort, ties to the lower position). That rounding is the gate's "top-k overlap
>= 0.99 in bf16" against the fp32 oracle: it only moves ties at the selection boundary.

Short rows: a query at position p <= 2047 sees at most 2048 keys, so its top-k is every visible key whatever the
scores are. Those rows get [0 .. p] directly (exact by construction; the PR runs the indexer anyway and pads with
-1). When the runner knows every row of the batch is that short (BatchCtx.host_max_pos, host side, never a .item()),
the scores are not computed at all; the index-K rows are still written (later queries need them).

Shared-indexer layers own nothing here: they take the top-k tensor of the closest preceding full layer, handed down
the layer loop in Shared (attention.py), never a module buffer.
"""

from __future__ import annotations

import os

import torch
from torch import nn

from .numerics import fake_fp8_ue8m0, layer_norm, ssmax_apply, ssmax_lambda
from .rope import RopeCache, rope_head

N_HEADS = 32
HEAD_DIM = 128
TOPK = 2048
NORM_EPS = 1e-6
# ML4_TOPK_SELECT=sort: the stable descending sort instead of the threshold select (same set; tests and A/B).
TOPK_SELECT = os.environ.get("ML4_TOPK_SELECT", "thresh")
# fp32 bytes of one [B, tile, heads, keys] dot block (DSV: ~256 MB keeps the query-tile intermediates bounded)
TILE_BYTES = int(os.environ.get("ML4_INDEX_TILE_BYTES", str(256 << 20)))
# Prompt rows: the threshold select's compaction holds [rows, k, block] and [rows, k, Lk / block] fp32 tensors
# (~2-4 MB per row at k 2048; a 2048-row tile at 4K keys was ~8 GiB live, the 64K-prompt peak sat 0.3 GiB under the
# allocator limit). ML4_INDEX_SELECT_BYTES (e.g. 536870912) bounds a tile's select
# tensors and ML4_INDEX_STEP_BYTES the tiles per lazy graph (mark_step between). 0 (default) = the previous tiling
# (score dot only, a flush every 8 tiles). Experimental: on the server only through the control file until its
# PPL A/B is done.
SELECT_BYTES = int(os.environ.get("ML4_INDEX_SELECT_BYTES", "0"))
STEP_BYTES = int(os.environ.get("ML4_INDEX_STEP_BYTES", str(1 << 30)))
SELECT_BLOCK = 256
HCHUNK = 4
# ML4_INDEX_DECODE_FAST=1: decode rows (S <= 8) score all 32 heads in one dot and add the 4-head partial sums in the
# same order as the chunk loop, and find the k-th value with one topk instead of the 16-step bit search (the
# loop and the search were ~200 lazy nodes per full layer). Same values and indices (--unit).
DECODE_FAST = os.environ.get("ML4_INDEX_DECODE_FAST", "0") == "1"
DECODE_ROWS = 8
# ML4_INDEX_DECODE_SELECT (control key index_decode_select; read at graph capture): how decode / verify rows (S <= 8,
# under ML4_INDEX_DECODE_FAST) turn their bf16 score row into the top-k indices.
#   key     (default) select_topk_key: torch.topk alone, on a key that makes every entry distinct with ties to the
#           lower position (chunks of 65536 + a merge above). The same set in the same (ascending) order as the threshold
#           select; per full layer 0.13 -> 0.04 / 0.17 -> 0.07 / 0.24 -> 0.11 ms at 8K / 64K / 128K keys for one row, 0.27
#           -> 0.05 / 0.37 -> 0.13 / 0.52 -> 0.25 for four (the threshold select's block compaction).
#   thresh  select_topk_thresh(kth_by_topk=True), the previous path.
DECODE_SELECT = os.environ.get("ML4_INDEX_DECODE_SELECT", "key")
UKEY_MAX = 1 << 16  # select_topk_key: one topk over the packed key up to this many keys
# Prompt chunks. ML4_PROMPT_MASK=1 (default): the unabsorbed prompt form gets the top-k as
# a mask (same set, no index compaction; one k-th value search per chunk instead of per tile): TTFT 16K 50.6 -> 11.4 s,
# 64K 344.8 -> 146.0 s; bit-identical to the index path at the same dot tile ("0" restores it). ML4_PROMPT_ABSORB_KEYS: above this many keys a prompt chunk runs the absorbed
# form instead (flat ~80 ms per layer-chunk vs ~0.5 ms per 1K keys unabsorbed; 0 = never). ML4_INDEX_BF16_DOT=1:
# the score dot in bf16 with fp32 accumulation (the fake-FP8 operands are exact in bf16; the sum order differs:
# not bit-identical and no faster in the bench, off).
PROMPT_MASK = os.environ.get("ML4_PROMPT_MASK", "1") == "1"
PROMPT_ABSORB_KEYS = int(os.environ.get("ML4_PROMPT_ABSORB_KEYS", "0")) or (1 << 62)
INDEX_BF16_DOT = os.environ.get("ML4_INDEX_BF16_DOT", "0") == "1"
# ML4_INDEX_MASK_BYTES (control key index_mask_bytes): the prompt mask's k-th value search runs over at most this many
# bytes of [rows, Lk] fp32 at once (its cumsum and compare temporaries scale with it); longer chunks are split into row
# blocks with a flush between them. 1 GiB keeps a 2048-row chunk whole up to 128K keys and
# splits it in two at 256K, so the search's temporaries stay the size they have at 128K.
MASK_BYTES = int(os.environ.get("ML4_INDEX_MASK_BYTES", str(1 << 30)))
# ML4_INDEX_SPLIT=1 (default; control key index_split): a prompt chunk's mask is built by the TP ranks together, each scoring
# 1/n of the query rows (the indexer weights and index-K cache are the same on every rank, so each rank used to repeat
# the whole chunk), then the bit-packed rows are all-gathered (the GLM-5.3 DSA prefill, 2026-10).
# The same rows go through the same ops, so the mask is the replicated one bit for bit. Not inside a prompt HPU graph.
INDEX_SPLIT = os.environ.get("ML4_INDEX_SPLIT", "1") == "1"
_BITS: dict = {}
FORCED_FORM = os.environ.get("ML4_ATTN_FORM", "")
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


def select_topk_thresh(score: torch.Tensor, k: int, block: int = 256, kth_by_topk: bool = False):
    """dsv41_hpu.csa.select_topk_thresh: the same k entries as the stable descending sort (highest first, ties to
    the lower index), without sorting the rows. Returns (values, indices) with the indices in ASCENDING order.
      1. per row, the k-th largest value v: binary search over the 2^16 bf16 bit patterns in float order;
      2. take every entry > v, and the lowest-index entries == v until k (a running count of the ties);
      3. compaction without a scatter: per `block` entries a count and its running sum locate each output slot.
    Counts and positions in fp32 (exact below 2^24), which Gaudi's TPC runs far faster than int32."""
    assert score.dtype == torch.bfloat16, score.dtype
    *lead, L = score.shape
    pad = -L % block
    if pad:
        score = torch.nn.functional.pad(score, (0, pad), value=float("-inf"))
    Lp = L + pad
    dev = score.device
    lo = torch.full((*lead, 1), -32641, dtype=torch.int32, device=dev)  # -inf
    hi = torch.full((*lead, 1), 32767, dtype=torch.int32, device=dev)
    kf = float(k)

    def val(key):
        bits = torch.where(key >= 0, key, key ^ 0x7FFF)
        return bits.to(torch.int16).view(torch.bfloat16)

    if kth_by_topk:
        # the k-th largest value itself (padding is -inf, so Lp >= k always): the value the search converges to
        v = torch.topk(score, k, dim=-1, sorted=False)[0].amin(-1, keepdim=True)
    else:
        for _ in range(16):
            mid = (lo + hi + 1) >> 1
            ok = (score >= val(mid)).sum(-1, keepdim=True, dtype=torch.float32) >= kf
            lo = torch.where(ok, mid, lo)
            hi = torch.where(ok, hi, mid - 1)
        v = val(lo)
    take = _take(score, v, kf).to(torch.float32)
    nb = Lp // block
    tb = take.view(*lead, nb, block)
    ends = torch.cumsum(tb.sum(-1), -1)
    j = torch.arange(k, device=dev, dtype=torch.float32)
    b = (ends.unsqueeze(-2) <= j.view(k, 1)).sum(-1, dtype=torch.float32)
    bl = b.long()
    start = torch.gather(ends, -1, (bl - 1).clamp(min=0)) * (bl > 0)
    r = j - start
    rows = torch.gather(tb, -2, bl.unsqueeze(-1).expand(*lead, k, block))
    inner = (torch.cumsum(rows, -1) <= r.unsqueeze(-1)).sum(-1, dtype=torch.float32)
    idx = (b * block + inner).long()
    return torch.gather(score, -1, idx), idx


def decode_select() -> str:
    from .control import get as ctl_get

    return ctl_get("index_decode_select", DECODE_SELECT)


def _ukey(score: torch.Tensor) -> torch.Tensor:
    """A distinct fp32 key per entry of bf16 score rows of <= 65536 entries whose order is the stable descending sort's:
    the score's bf16 bits in the high half (-0.0 as +0.0, so the two zeros tie as they compare; -inf as the lowest
    finite bf16, -3.39e38, never a score, as -inf's own bits would make NaN keys) and a tie-break in the low half that
    ranks the lower index higher at an equal score (65535 - j at or above zero, j below: the bit order of negative
    floats runs the other way). Scalar fills only: a zeros_like / full_like [R, L] constant inside a decode graph stays
    allocated for the graph's life (+8 bytes per score entry per layer per captured shape)."""
    b = score.view(torch.int16).to(torch.int32)
    b = b.masked_fill(b == -32768, 0).masked_fill(b == -128, -129)  # -0.0 -> +0.0; -inf -> -max (0xFF7F)
    j = torch.arange(score.shape[-1], device=score.device, dtype=torch.int32)
    return ((b << 16) | torch.where(b >= 0, (UKEY_MAX - 1) - j, j)).view(torch.float32)


def select_topk_key(score: torch.Tensor, k: int) -> torch.Tensor:
    """select_topk_thresh's indices (the k largest bf16 scores per row, ties to the lower index, -inf last) in the same
    ascending order, from torch.topk alone (decode / verify rows, ML4_INDEX_DECODE_SELECT=key). k < L.
      L <= 65536: one topk over _ukey's distinct keys, the k indices sorted ascending.
      L > 65536 (an index needs more than 16 bits): the row in chunks of 65536 (the last padded with -inf), each
        chunk's top k as above (every row winner is among its chunk's top k: fewer than k entries of the chunk beat
        it), then the top k of the chunks' candidates, laid out in ascending position, by _ukey over the candidate
        order (ties to the lower candidate = the lower position)."""
    L = score.shape[-1]
    if L <= UKEY_MAX:
        return torch.sort(torch.topk(_ukey(score), k, dim=-1, sorted=False)[1], dim=-1)[0]
    nc = -(-L // UKEY_MAX)
    assert nc * k <= UKEY_MAX, (L, k)
    sp = torch.nn.functional.pad(score, (0, nc * UKEY_MAX - L), value=float("-inf")) if nc * UKEY_MAX > L else score
    lead = sp.shape[:-1]
    loc = torch.sort(torch.topk(_ukey(sp.view(*lead, nc, UKEY_MAX)), k, dim=-1, sorted=False)[1], dim=-1)[0]
    off = torch.arange(nc, device=sp.device, dtype=loc.dtype).view(nc, 1) * UKEY_MAX
    cand = (loc + off).flatten(-2)  # [..., nc k] ascending positions
    pos = torch.sort(torch.topk(_ukey(torch.gather(sp, -1, cand)), k, dim=-1, sorted=False)[1], dim=-1)[0]
    return torch.gather(cand, -1, pos)


def _take(score: torch.Tensor, v: torch.Tensor, kf: float) -> torch.Tensor:
    """The selected entries as a bool mask: every entry > v, and the lowest-index entries == v until k."""
    gt = score > v
    eq = score == v
    need = kf - gt.sum(-1, keepdim=True, dtype=torch.float32)
    return gt | (eq & (torch.cumsum(eq.to(torch.float32), -1) <= need))


def select_mask_thresh(score: torch.Tensor, k: int) -> torch.Tensor:
    """select_topk_thresh's set as a bool mask over the row (no compaction): the prompt form only needs the mask
    (the compaction was ~90% of the indexer's prompt time)."""
    assert score.dtype == torch.bfloat16, score.dtype
    lo = torch.full((*score.shape[:-1], 1), -32641, dtype=torch.int32, device=score.device)
    hi = torch.full_like(lo, 32767)

    def val(key):
        return torch.where(key >= 0, key, key ^ 0x7FFF).to(torch.int16).view(torch.bfloat16)

    for _ in range(16):
        mid = (lo + hi + 1) >> 1
        ok = (score >= val(mid)).sum(-1, keepdim=True, dtype=torch.float32) >= float(k)
        lo = torch.where(ok, mid, lo)
        hi = torch.where(ok, hi, mid - 1)
    return _take(score, val(lo), float(k))


def prompt_mask(S: int, Lk: int) -> bool:
    """Prompt chunks whose attention runs the unabsorbed form take the top-k as a [B, S, Lk] mask; above
    PROMPT_ABSORB_KEYS keys they run the absorbed form, which needs indices. attention.py uses the same rule."""
    from .control import get as ctl_get

    return ctl_get("prompt_mask", PROMPT_MASK) and not FORCED_FORM and S > DECODE_ROWS and Lk <= absorb_keys()


def absorb_keys() -> int:
    from .control import get as ctl_get

    return ctl_get("prompt_absorb_keys", PROMPT_ABSORB_KEYS) or (1 << 62)


def select_topk(score: torch.Tensor, k: int, kth_by_topk: bool = False) -> torch.Tensor:
    """Indices of the k largest bf16 scores per row (ties to the lower index); order unspecified."""
    if TOPK_SELECT == "thresh":
        return select_topk_thresh(score, k, kth_by_topk=kth_by_topk)[1]
    return torch.sort(score, dim=-1, descending=True, stable=True)[1][..., :k]


def dense_topk(pos: torch.Tensor, k: int) -> torch.Tensor:
    """The exact top-k of rows with pos <= k - 1: every visible key [0 .. pos], -1 after (and for padding)."""
    j = torch.arange(k, device=pos.device).view(*([1] * pos.dim()), k)
    return torch.where(j <= pos.unsqueeze(-1), j, -1)


def dcp_rows(cache, S: int) -> bool:
    """Rows that run the DCP decode form (paged.py): the cache has the 1/n layout and the chunk is decode / verify."""
    from .paged import DCP_ROWS

    return getattr(cache, "dcp", None) is not None and S <= DCP_ROWS


def dcp_dense(pos: torch.Tensor, k: int, n: int, r: int) -> torch.Tensor:
    """dense_topk under DCP: this rank's share of [0 .. pos] (positions j n + r), -1 after: [B, S, ceil(k / n)]."""
    w = -(-k // n)
    j = torch.arange(w, device=pos.device).view(*([1] * pos.dim()), w) * n + r
    return torch.where(j <= pos.unsqueeze(-1), j, -1)


def short_topk(ctx, k: int, cache) -> torch.Tensor:
    """The top-k when every row of the batch is at position <= k - 1 (no scores): dense_topk, or under DCP decode
    rows this rank's share of it (attention.absorbed_dcp takes the rank's own keys)."""
    pos = ctx.positions
    if dcp_rows(cache, pos.shape[1]):
        n, r = cache.dcp
        return ctx.shared(("dcp_dense", k), lambda: dcp_dense(pos, k, n, r))
    return ctx.shared(("dense_topk", k), lambda: dense_topk(pos, k))


class ML4Indexer(nn.Module):
    """One full indexer (44 per model). Linears come from `linear(name, in, out, kind, dtype)` as in attention.py;
    all of them are replicated."""

    def __init__(self, cfg, layer_id: int, use_rope: bool, linear, topk: int = TOPK):
        super().__init__()
        D = cfg.hidden_size
        self.layer_id, self.use_rope, self.topk = layer_id, use_rope, topk
        self.q_norm_w = nn.Parameter(torch.ones(D, dtype=torch.bfloat16), requires_grad=False)
        self.q_norm_b = nn.Parameter(torch.zeros(D, dtype=torch.bfloat16), requires_grad=False)
        self.wq_b = linear("indexer.wq_b", D, N_HEADS * HEAD_DIM, "replicated", torch.bfloat16)
        self.wk = linear("indexer.wk", D, HEAD_DIM, "replicated", torch.bfloat16)
        self.weights_proj = linear("indexer.weights_proj", D, N_HEADS, "replicated", torch.bfloat16)
        self.k_norm_w = nn.Parameter(torch.ones(HEAD_DIM, dtype=torch.float32), requires_grad=False)
        self.k_norm_b = nn.Parameter(torch.zeros(HEAD_DIM, dtype=torch.float32), requires_grad=False)
        self.ssmax_s = nn.Parameter(torch.zeros(N_HEADS, dtype=torch.bfloat16), requires_grad=False)
        self.ssmax_b = nn.Parameter(torch.ones(N_HEADS, dtype=torch.bfloat16), requires_grad=False)

    def qk(self, a: torch.Tensor, cos, sin, lp, enc: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(q [B, S, 32, 128] fake-FP8 bf16, k [B, S, 128] fake-FP8 bf16, w [B, S, 32] fp32 incl. both scales).
        enc (ML4_R6_FFK, the cache takes encoded rows): k may come back as (index-K rows [B, S, 66],)."""
        from . import numerics as _nu, r5k, r6k

        pf = (
            r6k.take(a) if r6k.DUAL not in ("0", False) else None
        )  # ML4_R6_DUAL: the fp32 product the compiler would hand layer_norm
        if pf is not None:
            qn = torch.nn.functional.layer_norm(
                pf, (pf.shape[-1],), self.q_norm_w.float(), self.q_norm_b.float(), NORM_EPS
            ).to(a.dtype)
        else:
            qn = layer_norm(a, self.q_norm_w, self.q_norm_b, NORM_EPS)
        q = _lin(self.wq_b, qn).unflatten(-1, (N_HEADS, HEAD_DIM))
        if r5k.ff8_on(q) and r5k.FF8 == "x" and not self.use_rope and _nu.FAKE_QUANT:
            # ML4_R5_FF8=x on a NoPE layer: quantize from the fp32 values ahead of the bf16 casts
            lam = ssmax_lambda(lp, self.ssmax_s, self.ssmax_b, r5=True)
            qf = q.float() * lam.float().unsqueeze(-1)
            kf = torch.nn.functional.layer_norm(
                _lin(self.wk, a).float(), (HEAD_DIM,), self.k_norm_w.float(), self.k_norm_b.float(), NORM_EPS
            )
            w = _lin(self.weights_proj, a).float() * (HEAD_DIM**-0.5 * N_HEADS**-0.5)
            if enc and r6k.ffk_on(kf, False):
                return r5k.ff8_f32(qf), (r6k.ffk(kf),), w
            return r5k.ff8_f32(qf), r5k.ff8_f32(kf), w
        q = ssmax_apply(q, ssmax_lambda(lp, self.ssmax_s, self.ssmax_b, r5=True))
        k = layer_norm(_lin(self.wk, a), self.k_norm_w, self.k_norm_b, NORM_EPS)
        if self.use_rope:
            q = rope_head(q, cos.unsqueeze(-2), sin.unsqueeze(-2), r5=True)
            k = rope_head(k, cos, sin, r5=True)
        w = _lin(self.weights_proj, a).float() * (HEAD_DIM**-0.5 * N_HEADS**-0.5)
        if enc and self.use_rope and r6k.ffk_on(k, True):  # ML4_R6_FFK: k straight to its cache row
            kr = (r6k.ffk(k),)
            if r5k.ff8_on(q) and _nu.FAKE_QUANT:
                return r5k.ff8(q), kr, w
            return fake_fp8_ue8m0(q), kr, w
        if r5k.ff8_on(q) and _nu.FAKE_QUANT and (self.use_rope or r5k.FF8 == "bf16"):  # ML4_R5_FF8
            return r5k.ff8(q), r5k.ff8(k), w
        return fake_fp8_ue8m0(q), fake_fp8_ue8m0(k), w

    def forward(self, a: torch.Tensor, ctx, caches: dict, rope: RopeCache) -> torch.Tensor:
        """Write this chunk's index-K rows, then return topk [B, S, k] key positions (-1 = none), k = min(2048,
        table length). a [B, S, D]."""
        pos, valid = ctx.positions, ctx.valid
        B, S, _ = a.shape
        cos, sin, lp = ctx.shared(("rope",), lambda: rope.at(pos))
        from . import r6k

        q, k, w = self.qk(a, cos, sin, lp, enc=r6k.FFK != "0" and r6k.ffk_cache_ok(caches["index_k"]))
        if isinstance(k, tuple):  # ML4_R6_FFK: encoded index-K rows
            caches["index_k"].write(ctx, pos, k[0], valid, key="pos", encoded=True)
        else:
            caches["index_k"].write(ctx, pos, k, valid, key="pos")
        Lk = ctx.max_len
        k_out = min(self.topk, Lk)
        host_max = getattr(ctx, "host_max_pos", None)
        if host_max is not None and host_max <= k_out - 1:
            return short_topk(ctx, k_out, caches["index_k"])  # every row short: no scores
        from . import k1ix

        if k1ix.ENABLED and k1ix.eligible(S, Lk, k_out, caches["index_k"]):
            return k1ix.select(q, w, ctx, caches["index_k"], k_out)  # ML4_INDEX_KERNEL=1: same indices, same order
        return self.select(q, w, ctx, caches["index_k"], k_out)

    def scores(self, q, w, K, s0, s1) -> torch.Tensor:
        """fp32 [B, s1 - s0, Lk]: sum_h w * relu(q_h . K), heads in chunks of HCHUNK (fp32 dot, fp32 sum)."""
        B, Lk = K.shape[0], K.shape[1]
        Kt = K.float().transpose(1, 2)  # [B, 128, Lk]
        score = torch.zeros(B, s1 - s0, Lk, dtype=torch.float32, device=K.device)
        if DECODE_FAST and s1 - s0 <= DECODE_ROWS:
            St, nc = s1 - s0, N_HEADS // HCHUNK
            d = torch.matmul(q[:, s0:s1].float().reshape(B, -1, HEAD_DIM), Kt).view(B, St, N_HEADS, Lk)
            part = (d.relu() * w[:, s0:s1].unsqueeze(-1)).view(B, St, nc, HCHUNK, Lk).sum(3)
            for c in range(nc):
                score = score + part[:, :, c]
            return score
        from .control import get as ctl_get

        Kb = K.transpose(1, 2) if ctl_get("index_bf16_dot", INDEX_BF16_DOT) else None  # [B, 128, Lk] bf16
        for h0 in range(0, N_HEADS, HCHUNK):
            if Kb is not None:
                qq = q[:, s0:s1, h0 : h0 + HCHUNK].reshape(B, -1, HEAD_DIM)
                d = torch.matmul(qq, Kb).float().view(B, s1 - s0, -1, Lk)
            else:
                qq = q[:, s0:s1, h0 : h0 + HCHUNK].float().reshape(B, -1, HEAD_DIM)
                d = torch.matmul(qq, Kt).view(B, s1 - s0, -1, Lk)
            score = score + (d.relu() * w[:, s0:s1, h0 : h0 + HCHUNK].unsqueeze(-1)).sum(2)
        return score

    def select(self, q, w, ctx, cache, k_out: int) -> torch.Tensor:
        """topk [B, S, k_out] key positions (-1 none), or for prompt chunks under prompt_mask() a bool [B, S, Lk]
        mask of the same set (rows at p <= k_out - 1: every visible key)."""
        pos = ctx.positions
        B, S = pos.shape
        if dcp_rows(cache, S):
            return self._select_dcp(q, w, ctx, cache, k_out)
        Lk = ctx.max_len
        ent = torch.arange(Lk, device=pos.device).view(1, Lk).expand(B, Lk)
        reach = pos.amax(-1, keepdim=True) + 1
        K, kvalid = cache.read(ctx, ent, ent < reach, key=("ent", Lk), share=True)  # [B, Lk, 128]
        dot_row = B * HCHUNK * Lk * 4
        tile = max(1, min(S, TILE_BYTES // max(1, dot_row)))
        if prompt_mask(S, Lk):
            return self._select_mask(q, w, K, kvalid, ent, pos, tile, k_out)
        per_step = 8
        from .control import get as ctl_get

        sel_bytes, step_bytes = ctl_get("index_select_bytes", SELECT_BYTES), ctl_get("index_step_bytes", STEP_BYTES)
        if sel_bytes > 0 and S > DECODE_ROWS:
            nb = -(-Lk // SELECT_BLOCK)
            sel_row = 2 * B * k_out * (SELECT_BLOCK + nb) * 4
            tile = max(1, min(tile, sel_bytes // sel_row))
            per_step = max(1, step_bytes // (tile * (sel_row + dot_row)))
        n_tiles = -(-S // tile)
        # with a flush inside the loop, tiles are collected and concatenated: slice writes into one torch.empty
        # across mark_steps gave wrong rows on the HPU (ML4_T4_PFEQ)
        parts = [] if n_tiles > per_step else None
        out = (
            None if (S <= tile or parts is not None) else torch.empty(B, S, k_out, dtype=torch.long, device=pos.device)
        )
        n = 0
        for s0 in range(0, S, tile):
            s1 = min(S, s0 + tile)
            score = self.scores(q, w, K, s0, s1).to(torch.bfloat16)
            p = pos[:, s0:s1]
            see = (ent.unsqueeze(1) <= p.unsqueeze(-1)) & kvalid.unsqueeze(1)
            score = score.masked_fill(~see, float("-inf"))
            fast = DECODE_FAST and s1 - s0 <= DECODE_ROWS
            if fast and k_out < Lk and TOPK_SELECT == "thresh" and decode_select() == "key":
                idx = select_topk_key(score, k_out)  # the same indices in the same order, faster
            else:
                idx = select_topk(score, k_out, kth_by_topk=fast)
            t = torch.where(torch.gather(see, 2, idx), idx, -1)
            t = torch.where((p <= k_out - 1).unsqueeze(-1), dense_topk(p, k_out), t)  # exact for short rows
            if parts is not None:
                parts.append(t)
                if len(parts) % per_step == 0 and s1 < S:
                    _mark_step()
                continue
            if out is None:
                out = t
            else:
                out[:, s0:s1] = t
                n += 1
                if n % per_step == 0 and s1 < S:
                    _mark_step()  # bound the lazy graph's live tile intermediates (DSV41_INDEX_STEP_TILES)
        return torch.cat(parts, 1) if parts is not None else out

    def _select_dcp(self, q, w, ctx, cache, k_out: int) -> torch.Tensor:
        """Decode / verify rows under DCP: this rank's selected keys [B, S, <= ML4_DCP_KLOC] (positions, -1 none),
        the rank's share of the replicated select's set. Scores over this rank's keys only (1/n of the dot), the
        rank's own top-k candidates (every global winner is among its rank's top k: fewer than k entries beat it
        anywhere), then from every rank's candidates (one all-gather of [B, S, k, 2] fp32) the global k-th value v
        and the tie cut: entries > v, and of the entries == v the lowest positions until k, as select_topk_thresh."""
        from .paged import DCP_KLOC, dcp_all_gather, dcp_count_overflow

        n, r = cache.dcp
        pos = ctx.positions
        B, S = pos.shape
        Lk = ctx.max_len
        L = Lk // n
        xl = (torch.arange(L, device=pos.device) * n + r).view(1, L).expand(B, L)
        K, kvalid = cache.read(
            ctx, xl, xl <= pos.amax(-1, keepdim=True), key=("dcp_ent", Lk), share=True, local=True
        )  # [B, L, 128]
        see = (xl.unsqueeze(1) <= pos.unsqueeze(-1)) & kvalid.unsqueeze(1)  # [B, S, L]
        score = self.scores(q, w, K, 0, S).to(torch.bfloat16).masked_fill(~see, float("-inf"))
        kl = min(k_out, L)
        lv, li = select_topk_thresh(score, kl, kth_by_topk=DECODE_FAST)  # ascending local index
        lp = li * n + r
        ok = torch.gather(see, 2, li)
        lvf = lv.float()
        got = dcp_all_gather(torch.stack([lvf, lp.float()], -1))  # [n, B, S, kl, 2]; positions < 2^24
        cv = got[..., 0].permute(1, 2, 0, 3).reshape(B, S, n * kl)
        cp = got[..., 1].permute(1, 2, 0, 3).reshape(B, S, n * kl)
        kk = min(k_out, n * kl)
        v = torch.topk(cv, kk, dim=-1, sorted=False)[0].amin(-1, keepdim=True)
        need = kk - (cv > v).sum(-1, keepdim=True)  # >= 1 ties at v are taken
        tied = torch.where(cv == v, cp, float(Lk))
        low = -torch.topk(-tied, kk, dim=-1, sorted=True)[0]  # tie positions, ascending
        cut = torch.gather(low, -1, (need - 1).clamp(0, kk - 1))
        win = ok & ((lvf > v) | ((lvf == v) & (lp.float() <= cut)) | (pos <= k_out - 1).unsqueeze(-1))
        wl = min(DCP_KLOC, kl)
        dcp_count_overflow((win.sum(-1) > wl).sum())
        idx = torch.topk(win.to(torch.float32), wl, dim=-1, sorted=False)[1]
        return torch.where(torch.gather(win, -1, idx), torch.gather(lp, -1, idx), -1)

    def _select_mask(self, q, w, K, kvalid, ent, pos, tile, k_out) -> torch.Tensor:
        """Prompt chunk -> bool [B, S, Lk], the same set as the index path. The scores are built per dot tile into
        one bf16 [B, S, Lk] row block and the k-th value search runs ONCE over the whole chunk: in lazy mode the
        index path's per-tile search (16 passes x tiles) was mostly host time (99 of 152 ms per layer-chunk at 16K).
        Bounded by the unabsorbed form's key range (ML4_PROMPT_ABSORB_KEYS)."""
        from . import prof

        prof.lap("ix_pre")
        B, S = pos.shape
        Lk = K.shape[1]
        from .control import get as ctl_get

        rows = max(1, min(S, int(ctl_get("index_mask_bytes", MASK_BYTES)) // (B * Lk * 4)))
        n = _split_ranks(S, Lk) if ctl_get("index_split", INDEX_SPLIT) else 1
        if n > 1:
            from .paged import dcp_group

            r = dcp_group()[1]
            r0, r1 = r * (S // n), (r + 1) * (S // n)
        else:
            r0, r1 = 0, S
        if rows >= r1 - r0:
            m = self._mask_rows(q, w, K, kvalid, ent, pos, tile, k_out, r0, r1)
        else:
            parts = []  # row blocks, a flush between them; collected and concatenated (see select())
            for a in range(r0, r1, rows):
                if a > r0 and q.device.type == "hpu":
                    _mark_step()
                parts.append(self._mask_rows(q, w, K, kvalid, ent, pos, tile, k_out, a, min(r1, a + rows)))
            m = torch.cat(parts, 1)
        prof.lap("ix_rows")
        return _gather_rows(m, n) if n > 1 else m

    def _mask_rows(self, q, w, K, kvalid, ent, pos, tile, k_out, r0, r1) -> torch.Tensor:
        """_select_mask for query rows [r0, r1): the rows are independent, so any split gives the same mask."""
        p = pos[:, r0:r1]
        score = torch.cat(
            [self.scores(q, w, K, s0, min(r1, s0 + tile)).to(torch.bfloat16) for s0 in range(r0, r1, tile)], 1
        )
        see = (ent.unsqueeze(1) <= p.unsqueeze(-1)) & kvalid.unsqueeze(1)
        score = score.masked_fill(~see, float("-inf"))
        return see & (select_mask_thresh(score, k_out) | (p <= k_out - 1).unsqueeze(-1))


def _split_ranks(S: int, Lk: int) -> int:
    """How many TP ranks share a prompt chunk's mask rows (1 = each rank builds all of them)."""
    from . import moe as _moe
    from .paged import dcp_group

    if _moe.GRAPH_PROMPT[0] or Lk % 8:
        return 1
    try:
        n = dcp_group()[2]
    except Exception:  # noqa: BLE001  (no process group: one rank)
        return 1
    return n if n > 1 and S % n == 0 and S > DECODE_ROWS else 1


def _bit_weights(dtype, device) -> torch.Tensor:
    key = (dtype, str(device))
    if key not in _BITS:
        _BITS[key] = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=dtype, device=device)
    return _BITS[key]


def _gather_rows(m: torch.Tensor, n: int) -> torch.Tensor:
    """This rank's mask rows [B, S/n, Lk] -> every rank's rows in rank order [B, S, Lk]: 8 keys per byte (a bf16
    matmul with weights 1..128, exact: the sums are integers <= 255), one all-gather of [n, B, S/n, Lk/8] uint8."""
    from .paged import dcp_all_gather

    B, s, Lk = m.shape
    packed = (
        torch.matmul(m.reshape(B, s, Lk // 8, 8).to(torch.bfloat16), _bit_weights(torch.bfloat16, m.device).view(8, 1))
        .view(B, s, Lk // 8)
        .to(torch.uint8)
    )
    got = dcp_all_gather(packed)  # [n, B, s, Lk / 8]
    got = got.permute(1, 0, 2, 3).reshape(B, n * s, Lk // 8)
    return (torch.bitwise_and(got.unsqueeze(-1), _bit_weights(torch.uint8, m.device)) != 0).view(B, n * s, Lk)


def _lin(mod, x):
    out = mod(x)
    return out[0] if isinstance(out, tuple) else out
