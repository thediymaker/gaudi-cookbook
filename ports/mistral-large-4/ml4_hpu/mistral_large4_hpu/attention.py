"""Sparse MLA for Mistral Large 4, model-local over paged caches .

Ground truth: the PR's MistralAttention + sparse MLA (HF for the unabsorbed meaning). Per layer l, for a
token at absolute position p with attention input a = RMSNorm_attn(x):
  q      = wq a                     [H, 128]   RoPE layers: [nope 64 | rope 64], rotate the last 64 (interleaved);
                                               NoPE layers: 128 nope dims, no rotation
  q     *= lambda(p) = s_h log(p+1) + b_h      SSMax, bf16, after RoPE, all 128 dims
  owner: kv = wkv_a a = [c_raw 512 | k_pe 64 (RoPE only)];  c = RMSNorm_kv(c_raw);  k_pe = RoPE(k_pe) or 0
         latent[l][p] = [c | k_pe]   (576 bf16; NoPE owners store k_pe = 0)
  kv-shared (39, 43, ..., 71): no wkv_a, no write; read latent[35] (layer 35 wrote this chunk earlier in the forward)
  wkv_b [H (P + 128), 512] per head [W_UK (P) ; W_UV (128)],  P = 64 (RoPE) or 128 (NoPE)
  keys of query t: top-k(t) from the layer's index source (its own indexer, or the closest preceding full one)
  logit  = q . [W_UK c_s | k_pe_s] * 128^-0.5         (NO YaRN mscale: the 3.106x trap)
  o      = sum_s softmax(logit) W_UV c_s;   out = wo concat_h o

Two forms of the same function:
  unabsorbed (prompt chunks): k_nope, v = wkv_b(c) for every key the chunk can see (the HF form; bf16 linear
    output), q . k over 128 dims, fp32 scores, mask = causal & valid & (row p <= 2047 or key in top-k(p)), fp32
    softmax, P rounded to bf16 (HF eager), P . V accumulated in fp32. Rows at p <= 2047 are dense (exact).
  absorbed (decode rows): gather the top-k latent rows [k, 576], ql = W_UK^T q_nope (bf16, the PR's bmm output),
    fp32 logits over [ql | q_pe] . [c | k_pe], fp32 softmax, P in bf16, ol = P . c (bf16), o = W_UV ol.
By default both forms keep every intermediate in fp32 (k/v, P, ql, ol) and round once at the output: the forms then
differ only by fp32 summation order. ML4_ATTN_BF16_POINTS=1 rounds at the oracle's bf16 points instead (ref/
attention.py; HF eager and the PR's bmm outputs). gates the two forms against each other at cos >= 0.99999 and
each against the oracle given the same key selection.

TP: heads split over ranks (wq column, wkv_b by head, SSMax s/b by head, wo row + the caller's all-reduce); wkv_a,
kv_a_norm and the indexer are replicated (every rank holds the same latent and computes the same top-k).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .indexer import TOPK, ML4Indexer, absorb_keys, dcp_dense, dcp_rows, dense_topk
from .paged import dcp_all_gather, dcp_group, dcp_overflow  # noqa: F401  (dcp_overflow: tools)
from .numerics import rms_norm, ssmax_apply, ssmax_lambda
from . import prof
from .rope import RopeCache, rope_tail, rotate

KV_RANK = 512
ROPE_DIM = 64
HEAD_DIM = 128  # q.k head dim of every layer (RoPE: 64 + 64; NoPE: 128)
V_DIM = 128
RMS_EPS = 1e-5
NEG_FLOOR = -1e30
# Deliberately broken variants for the controls (tests only). Each one must make a gate FAIL.
#   mscale: the DeepSeek-V3 MLA softmax scale x mscale(2048, mscale_all_dim=1)^2 = 3.1063
#   no_kv_share: a KV-shared layer reads an empty cache of its own instead of its source's (control)
FAULTS = {
    "mscale": os.environ.get("ML4_FAULT_MSCALE", "0") == "1",
    "no_kv_share": os.environ.get("ML4_FAULT_NO_KV_SHARE", "0") == "1",
}
# ML4_ATTN_FORM=absorbed|unabsorbed forces one form for every row (tests: the both-forms gate); default: unabsorbed
# for prompt chunks (S > DECODE_S), absorbed for decode / verify rows.
FORM = os.environ.get("ML4_ATTN_FORM", "")
DECODE_S = int(os.environ.get("ML4_DECODE_S", "8"))
Q_TILE = int(os.environ.get("ML4_ATTN_Q_TILE", "256"))
# ML4_ATTN_BF16_POINTS=1: round at the oracle's bf16 points (k/v, P, ql, ol); default 0 keeps every intermediate
# in fp32 and rounds once at the output (the two forms then agree to ~1e-7 instead of ~1e-5).
BF16_POINTS = os.environ.get("ML4_ATTN_BF16_POINTS", "0") == "1"
# Prompt Q tiles per lazy graph: a tile holds ~4 fp32 [B, Hl, Q_TILE, Lk] tensors (2 GiB at 64K keys); flush after as
# many tiles as fit in ML4_ATTN_STEP_BYTES (0 = never: the default, the previous behavior). Experimental: on the
# server only through the control file until its PPL A/B is done.
ATTN_STEP_BYTES = int(os.environ.get("ML4_ATTN_STEP_BYTES", "0"))
# ML4_ATTN_TILE_CAP (control key attn_tile_cap): the prompt Q tile shrinks so one fp32 [B, Hl, tile, Lk] tensor stays
# within this many bytes (each tile holds ~4 of them). 1 GiB: 256-row tiles up to 64K keys, 128 at 128K (256 before),
# 64 at 256K without key blocks; rows are independent, so the values do not change.
ATTN_TILE_CAP = int(os.environ.get("ML4_ATTN_TILE_CAP", str(1 << 30)))
# Prompt chunks over more than ML4_ATTN_KEY_BLOCK_MIN keys (default 131072: never at 128K) run the unabsorbed form in
# key blocks of ML4_ATTN_KEY_BLOCK (32768): k / v through wkv_b for one block at a time (fp32 [Lk, Hl, 256] is ~2 GiB at
# 256K keys, plus its permuted copies) and an online softmax over the blocks per Q tile (running max, sum and P.V in
# fp32). Blocks past the batch's largest position (host_max_pos) are skipped. Same function; the sums run in another
# order. Control keys attn_key_block, attn_key_block_min.
ATTN_KEY_BLOCK = int(os.environ.get("ML4_ATTN_KEY_BLOCK", "32768"))
ATTN_KEY_BLOCK_MIN = int(os.environ.get("ML4_ATTN_KEY_BLOCK_MIN", "131072"))
# ML4_ATTN_FSDPA=1 (default; control key attn_fsdpa): prompt chunks run the unabsorbed form through FusedSDPA (unabsorbed_fsdpa):
# k / v through wkv_b in bf16 (the HF form, as ML4_ATTN_BF16_POINTS), the selection as an additive mask, the kernel's
# accurate softmax ("None"; "fast" is badly wrong on deep layers, the GLM-5.3 DSA finding), in row blocks of
# ML4_ATTN_FSDPA_ROWS (control key attn_fsdpa_rows). The fp32 [Hl, rows, Lk] scores are never materialized.
# Not the same sums: pooled PPL 1.033 / 97.0% / KL 0.037 against the fp32 path's 1.006 /
# 97.6% / 0.021, inside the re-based bars and where a harmless fp32 reorder lands (1.034 / 97.2% / 0.031); code and
# techdoc unchanged; needles 10/10. "0": the fp32 path (unabsorbed tiles). Neither this nor ML4_ATTN_SOFT_LEAN runs
# inside a prompt HPU graph (<= 2048 keys, where they save nothing): short prompts keep the fp32 path.
ATTN_FSDPA = os.environ.get("ML4_ATTN_FSDPA", "1") == "1"
# ML4_ATTN_SOFT_LEAN=1 (default; control key attn_soft_lean): the fp32 prompt attention (ML4_ATTN_FSDPA=0) takes the
# softmax scale on q and divides by the row sums after P.V: the same fp32 math in another
# rounding order, two fewer passes over the [Hl, rows, Lk] scores. ~110K TTFT 321 -> 233 s, 32K 29.2 -> 27.8 s.
ATTN_SOFT_LEAN = os.environ.get("ML4_ATTN_SOFT_LEAN", "1") == "1"
ATTN_FSDPA_ROWS = int(os.environ.get("ML4_ATTN_FSDPA_ROWS", "512"))
# ML4_DCP=1: decode context parallel with the 1/tp paged layout (paged.py). Decode / verify rows run absorbed_dcp.
# ML4_ATTN_DECODE_MM=1 (control key attn_decode_mm): decode / verify rows (S <= DECODE_S) run the absorbed form as
# batched matmuls over the (row, query) pairs, with the gathered rows cast to fp32 once and the latent / rope parts as
# transposed views, instead of einsums. Same products and the same 512 + 64 split; one card
# in an HPU graph: bit-identical to the einsum form at B 1 / 4 / 8; on the server the absorbed form's PPL 01-04 is
# bit-identical in both forms (prompt chunks under attn_mm_prompt), and decode greedy splits from the einsum form where
# the junk-draft batch-shape control splits. 4 streams 16.9 -> 18.1 tok/s, 8 streams 12.0 -> 13.1. On by default.
DECODE_MM = os.environ.get("ML4_ATTN_DECODE_MM", "1") == "1"


def decode_mm() -> bool:
    from .control import get as ctl_get

    return bool(ctl_get("attn_decode_mm", DECODE_MM))


def _mm_prompt() -> bool:
    """Test key attn_mm_prompt: prompt chunks that run the absorbed form (prompt_absorb_keys) use absorbed_mm too."""
    from .control import get as ctl_get

    return bool(ctl_get("attn_mm_prompt", False))


def _tile_flush(S: int) -> None:
    """Test key attn_prompt_tile_flush: a lazy-graph flush after every absorbed-form Q tile of a prompt chunk (the
    absorbed form keeps every tile's fp32 rows in one graph: a 2048-token chunk ran out of device memory)."""
    if S > DECODE_S:
        from .control import get as ctl_get

        if ctl_get("attn_prompt_tile_flush", False):
            from .indexer import _mark_step

            _mark_step()


@dataclass(frozen=True)
class LayerRole:
    layer_id: int
    use_rope: bool
    kv_source: int  # layer whose latent cache this layer reads (itself if it owns one)
    has_indexer: bool  # "full": runs its own indexer (and owns an index-K cache)
    index_source: int  # layer whose top-k this layer uses (itself if full)
    is_moe: bool

    @property
    def owns_kv(self) -> bool:
        return self.kv_source == self.layer_id


def _cyc(spec: str, n: int) -> list[str]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return [parts[i % len(parts)] for i in range(n)]


def layer_roles(params: dict, n_layers: int | None = None) -> list[LayerRole]:
    """Per-layer roles from a Mistral-native params.json dict, as the PR's config adapter + get_layer_specs derive
    them (CFG:390-443, LS): NoPE pattern expanded cyclically; dsa.sharing ("full"/"shared"); kv_sharing source list;
    match_nope; first_k_dense_replace. A shared layer reuses the closest preceding full layer's top-k."""
    n = n_layers or params["n_layers"]
    nope = [t == "nope" for t in _cyc(params.get("nope") or "rope", n)]
    dsa = params.get("dsa") or {}
    sharing = _cyc(dsa.get("sharing") or "full", n)
    ks = (params.get("kv_sharing") or {}).get("source_layers_str")
    kv_src = [int(v) for v in ks.split(",")][:n] if ks else list(range(n))
    first_dense = (params.get("moe") or {}).get("first_k_dense_replace", 0)
    roles, last_full = [], None
    for i in range(n):
        full = sharing[i] == "full"
        if full:
            last_full = i
        assert last_full is not None, f"layer {i} shares an indexer but no full layer precedes it"
        assert nope[kv_src[i]] == nope[i], f"layer {i} and its KV source {kv_src[i]} differ in NoPE"
        roles.append(LayerRole(i, not nope[i], kv_src[i], full, last_full, i >= first_dense))
    return roles


@dataclass
class Shared:
    """What layers hand down the stack within ONE forward (never a module buffer)."""

    topk: dict = field(default_factory=dict)  # full layer -> [B, S, k] key positions (-1 none)


def _lin(mod, x):
    out = mod(x)
    return out[0] if isinstance(out, tuple) else out


def softmax_scale() -> float:
    s = HEAD_DIM**-0.5
    if FAULTS["mscale"]:
        s *= (0.1 * 1.0 * math.log(2048) + 1.0) ** 2
    return s


class ML4Attention(nn.Module):
    """One MLA layer. Linears come from `linear(name, in, out, kind, dtype)` so the same code runs with plain
    modules in tests and vLLM parallel layers in serving: kind in {"replicated", "column", "row"}; "row" returns
    the rank's partial sum (the caller all-reduces once per layer with the FFN, as GLM/DSV)."""

    def __init__(self, cfg, role: LayerRole, linear, tp_size: int = 1):
        super().__init__()
        self.role, self.cfg = role, cfg
        D, H = cfg.hidden_size, cfg.num_attention_heads
        assert H % tp_size == 0
        self.Hl = H // tp_size
        self.P = HEAD_DIM - ROPE_DIM if role.use_rope else HEAD_DIM  # W_UK rows per head
        self.wq = linear("wq", D, H * HEAD_DIM, "column", torch.bfloat16)
        if role.owns_kv:
            self.wkv_a = linear(
                "wkv_a_with_mqa", D, KV_RANK + (ROPE_DIM if role.use_rope else 0), "replicated", torch.bfloat16
            )
            self.kv_a_norm = nn.Parameter(torch.ones(KV_RANK, dtype=torch.bfloat16), requires_grad=False)
        # [Hl (P + V), 512], this rank's heads; used as a matrix, never called
        self.wkv_b = nn.Parameter(
            torch.empty(self.Hl * (self.P + V_DIM), KV_RANK, dtype=torch.bfloat16), requires_grad=False
        )
        self.wo = linear("wo", H * V_DIM, D, "row", torch.bfloat16)
        self.ssmax_s = nn.Parameter(torch.zeros(self.Hl, dtype=torch.bfloat16), requires_grad=False)
        self.ssmax_b = nn.Parameter(torch.ones(self.Hl, dtype=torch.bfloat16), requires_grad=False)
        self.indexer = ML4Indexer(cfg, role.layer_id, role.use_rope, linear) if role.has_indexer else None

    # ------------------------------------------------------------------ pieces
    def project_q(self, a, cos, sin, lp):
        q = _lin(self.wq, a).unflatten(-1, (self.Hl, HEAD_DIM))
        if self.role.use_rope:
            q = rope_tail(q, cos.unsqueeze(-2), sin.unsqueeze(-2), r5=True)
        return ssmax_apply(q, ssmax_lambda(lp, self.ssmax_s, self.ssmax_b, r5=True))

    def latent_rows(self, a, cos, sin, kv=None):
        """[c | k_pe] [B, S, 576] bf16 of an owner layer."""
        kv = _lin(self.wkv_a, a) if kv is None else kv
        from . import r5k

        if r5k.rms_on(kv, KV_RANK) and kv.is_contiguous():  # ML4_R5_RMS: reads the first 512 columns of kv
            c = r5k.rms(kv, self.kv_a_norm, RMS_EPS, KV_RANK)
        else:
            c = rms_norm(kv[..., :KV_RANK], self.kv_a_norm, RMS_EPS)
        if self.role.use_rope:
            kpe = rotate(kv[..., KV_RANK:], cos, sin)
        else:
            kpe = torch.zeros(*c.shape[:-1], ROPE_DIM, dtype=c.dtype, device=c.device)
        return torch.cat([c, kpe], -1)

    def w_uk_uv(self):
        w = self.wkv_b.view(self.Hl, self.P + V_DIM, KV_RANK)
        return w[:, : self.P], w[:, self.P :]  # [Hl, P, 512], [Hl, 128, 512]

    # ----------------------------------------------------------------- forward
    def forward(self, a: torch.Tensor, ctx, caches_all: dict, shared: Shared, rope: RopeCache) -> torch.Tensor:
        """a [B, S, D] (RMSNorm'ed) -> this rank's partial attention output [B, S, D] (before the all-reduce)."""
        role = self.role
        pos, valid = ctx.positions, ctx.valid
        B, S, _ = a.shape
        cos, sin, lp = ctx.shared(("rope",), lambda: rope.at(pos))  # once per decode graph (graphs.HOIST)
        q = self.project_q(a, cos, sin, lp)
        if role.owns_kv:
            from . import r5k

            lc = caches_all[role.layer_id]["latent"]
            kv = _lin(self.wkv_a, a) if r5k.LAT else None
            if kv is not None and r5k.lat_on(kv, lc):  # ML4_R5_LAT: the encoded row in one kernel, same bytes
                lc.write(
                    ctx,
                    pos,
                    r5k.latent(kv, self.kv_a_norm, RMS_EPS, cos, sin, role.use_rope),
                    valid,
                    key="pos",
                    encoded=True,
                )
            else:
                lc.write(ctx, pos, self.latent_rows(a, cos, sin, kv), valid, key="pos")
        prof.lap("q")
        if self.indexer is not None:
            shared.topk[role.layer_id] = self.indexer(a, ctx, caches_all[role.layer_id], rope)
            prof.lap("index")
        topk = shared.topk[role.index_source]
        src = caches_all[role.kv_source]["latent"]
        if FAULTS["no_kv_share"] and not role.owns_kv:
            from .paged import TaggedCache

            src = TaggedCache(torch.zeros_like(src.data), torch.full_like(src.tags, -1), "latent", role.layer_id)
        # prompt chunks over more than ML4_PROMPT_ABSORB_KEYS keys run the absorbed form
        form = FORM or ("absorbed" if S <= DECODE_S or ctx.max_len > absorb_keys() else "unabsorbed")
        o = self.absorbed(q, topk, ctx, src) if form == "absorbed" else self.unabsorbed(q, topk, ctx, src)
        return _lin(self.wo, o.flatten(2))

    def unabsorbed(self, q, topk, ctx, src) -> torch.Tensor:
        """Prompt form: keys expanded through wkv_b. q [B, S, Hl, 128] -> o [B, S, Hl, 128] bf16."""
        pos = ctx.positions
        B, S = pos.shape
        Lk = ctx.max_len
        dev = q.device
        ent = torch.arange(Lk, device=dev).view(1, Lk).expand(B, Lk)
        rows, kvalid = src.read(ctx, ent, ent <= pos.amax(-1, keepdim=True), key=("ent", Lk), share=True)
        from .control import get as ctl_get
        from . import moe as _moe

        if dev.type == "hpu" and not _moe.GRAPH_PROMPT[0] and ctl_get("attn_fsdpa", ATTN_FSDPA):
            return self.unabsorbed_fsdpa(q, topk, ctx, rows, kvalid, ent)
        kb = int(ctl_get("attn_key_block", ATTN_KEY_BLOCK))
        if not BF16_POINTS and kb > 0 and Lk > int(ctl_get("attn_key_block_min", ATTN_KEY_BLOCK_MIN)) and Lk % kb == 0:
            return self.unabsorbed_kblocks(q, topk, ctx, rows, kvalid, ent, kb)
        c, kpe = rows[..., :KV_RANK], rows[..., KV_RANK:]
        if BF16_POINTS:
            kvb = F.linear(c, self.wkv_b).view(B, Lk, self.Hl, self.P + V_DIM)  # bf16, the HF kv_b_proj output
        else:
            kvb = F.linear(c.float(), self.wkv_b.float()).view(B, Lk, self.Hl, self.P + V_DIM)
        k = kvb[..., : self.P]
        if self.role.use_rope:
            k = torch.cat([k, kpe.unsqueeze(2).expand(B, Lk, self.Hl, ROPE_DIM)], -1)
        kf = k.float().permute(0, 2, 3, 1)  # [B, Hl, 128, Lk]
        vf = kvb[..., self.P :].float().transpose(1, 2)  # [B, Hl, Lk, 128]
        prof.lap("a_kv")
        scale = softmax_scale()
        k_sel = (
            topk.shape[-1]
            if topk is not None and topk.dtype != torch.bool
            else (min(TOPK, Lk) if topk is not None else TOPK)
        )
        from .control import get as ctl_get

        step_bytes = ctl_get("attn_step_bytes", ATTN_STEP_BYTES)
        qt = int(ctl_get("attn_q_tile", Q_TILE))  # test key: the Q tile (rows are independent: same values)
        # at most ML4_ATTN_TILE_CAP bytes per fp32 [B, Hl, tile, Lk] score tensor: 256 rows up to 64K keys, 128 at 128K
        qt = max(16, min(qt, int(ctl_get("attn_tile_cap", ATTN_TILE_CAP)) // (B * self.Hl * Lk * 4)))
        per_step = max(1, step_bytes // (B * self.Hl * qt * Lk * 16)) if step_bytes > 0 else 0
        # flushing inside the loop: collect tiles and concatenate (slice writes into one torch.empty across
        # mark_steps gave wrong rows on the HPU, ML4_T4_PFEQ)
        parts = [] if (per_step and dev.type == "hpu" and -(-S // qt) > per_step) else None
        lean = not BF16_POINTS and not _moe.GRAPH_PROMPT[0] and bool(ctl_get("attn_soft_lean", ATTN_SOFT_LEAN))
        out = None if parts is not None else torch.empty(B, S, self.Hl, V_DIM, dtype=q.dtype, device=dev)

        def tile(s0, s1):
            # a function, so the tile's scores / P / mask are freed before the next flush: names still bound at a
            # mark_step become graph outputs and stay allocated while the launch queue is ahead of the device
            p = pos[:, s0:s1]
            allowed = (ent.unsqueeze(1) <= p.unsqueeze(-1)) & kvalid.unsqueeze(1) & (p >= 0).unsqueeze(-1)
            if topk is not None and topk.dtype == torch.bool:
                sel = topk[:, s0:s1]  # the indexer's mask (prompt_mask)
                allowed = allowed & (sel | (p <= k_sel - 1).unsqueeze(-1))
            elif topk is not None:
                t = topk[:, s0:s1]
                sel = (
                    torch.zeros(B, s1 - s0, Lk, dtype=torch.float32, device=dev).scatter_add_(
                        2, t.clamp(min=0), (t >= 0).float()
                    )
                    > 0
                )
                allowed = allowed & (sel | (p <= k_sel - 1).unsqueeze(-1))
            prof.lap("a_mask")
            qf = q[:, s0:s1].float().transpose(1, 2)  # [B, Hl, St, 128]
            if lean:
                # the scale on q and the normalisation after P.V: two fewer passes over the fp32 [Hl, St, Lk]
                s = torch.matmul(qf * scale, kf).masked_fill(~allowed.unsqueeze(1), float("-inf"))
                prof.lap("a_qk")
                e = torch.exp(s - s.amax(-1, keepdim=True).clamp_min(NEG_FLOOR))
                l = e.sum(-1, keepdim=True).clamp_min(1e-30)
                prof.lap("a_soft")
                o = (torch.matmul(e, vf) / l).transpose(1, 2).to(q.dtype)
                prof.lap("a_pv")
                return o
            s = torch.matmul(qf, kf) * scale  # [B, Hl, St, Lk]
            prof.lap("a_qk")
            s = s.masked_fill(~allowed.unsqueeze(1), float("-inf"))
            m = s.amax(-1, keepdim=True).clamp_min(NEG_FLOOR)
            e = torch.exp(s - m)
            pr = e / e.sum(-1, keepdim=True).clamp_min(1e-30)
            if BF16_POINTS:
                pr = pr.to(q.dtype).float()  # P in bf16 before P.V (HF eager)
            prof.lap("a_soft")
            o = torch.matmul(pr, vf).transpose(1, 2).to(q.dtype)
            prof.lap("a_pv")
            return o

        for s0 in range(0, S, qt):
            if parts is not None and s0 and (s0 // qt) % per_step == 0:
                from .indexer import _mark_step

                _mark_step()
            s1 = min(S, s0 + qt)
            if parts is not None:
                parts.append(tile(s0, s1))
            else:
                out[:, s0:s1] = tile(s0, s1)
        return torch.cat(parts, 1) if parts is not None else out

    def unabsorbed_fsdpa(self, q, topk, ctx, rows, kvalid, ent) -> torch.Tensor:
        """unabsorbed() through FusedSDPA (ML4_ATTN_FSDPA). q [B, S, Hl, D] -> o [B, S, Hl, 128] bf16."""
        from habana_frameworks.torch.hpex.kernels import FusedSDPA
        from .control import get as ctl_get

        pos = ctx.positions
        B, S = pos.shape
        Lk = ent.shape[1]
        c, kpe = rows[..., :KV_RANK], rows[..., KV_RANK:]
        # test key attn_fsdpa_dtype: "bf16" (default), "f32kv" (k / v through wkv_b in fp32, then bf16), "f32" (the
        # kernel in fp32)
        dt = str(ctl_get("attn_fsdpa_dtype", "bf16"))
        if dt == "bf16":
            kvb = F.linear(c.to(self.wkv_b.dtype), self.wkv_b)
        else:
            kvb = F.linear(c.float(), self.wkv_b.float())
            if dt == "f32kv":
                kvb = kvb.to(q.dtype)
        kvb = kvb.view(B, Lk, self.Hl, self.P + V_DIM)
        k = kvb[..., : self.P]
        if self.role.use_rope:
            k = torch.cat([k, kpe.to(k.dtype).unsqueeze(2).expand(B, Lk, self.Hl, ROPE_DIM)], -1)
        D = k.shape[-1]
        assert q.shape[-1] == D, (q.shape, D)
        v = kvb[..., self.P :]
        if D > V_DIM:
            v = F.pad(v, (0, D - V_DIM))  # the kernel takes one head dim
        k, v = k.transpose(1, 2), v.transpose(1, 2)  # [B, Hl, Lk, D]
        prof.lap("attn_kv")
        k_sel = (
            topk.shape[-1]
            if topk is not None and topk.dtype != torch.bool
            else (min(TOPK, Lk) if topk is not None else TOPK)
        )
        rb = max(1, int(ctl_get("attn_fsdpa_rows", ATTN_FSDPA_ROWS)))
        outs = []
        for s0 in range(0, S, rb):
            s1 = min(S, s0 + rb)
            if s0:
                from .indexer import _mark_step

                _mark_step()
            p = pos[:, s0:s1]
            allowed = (ent.unsqueeze(1) <= p.unsqueeze(-1)) & kvalid.unsqueeze(1) & (p >= 0).unsqueeze(-1)
            if topk is not None and topk.dtype == torch.bool:
                allowed = allowed & (topk[:, s0:s1] | (p <= k_sel - 1).unsqueeze(-1))
            elif topk is not None:
                t = topk[:, s0:s1]
                sel = (
                    torch.zeros(B, s1 - s0, Lk, dtype=torch.float32, device=q.device).scatter_add_(
                        2, t.clamp(min=0), (t >= 0).float()
                    )
                    > 0
                )
                allowed = allowed & (sel | (p <= k_sel - 1).unsqueeze(-1))
            m = torch.where(allowed, 0.0, NEG_FLOOR).to(k.dtype).unsqueeze(1)  # [B, 1, St, Lk]
            prof.lap("attn_mask")
            o = FusedSDPA.apply(q[:, s0:s1].transpose(1, 2).to(k.dtype), k, v, m, 0.0, False, softmax_scale(), "None")
            outs.append(o[..., :V_DIM].transpose(1, 2).to(q.dtype))
            prof.lap("attn_sdpa")
        return torch.cat(outs, 1) if len(outs) > 1 else outs[0]

    def unabsorbed_kblocks(self, q, topk, ctx, rows, kvalid, ent, kb) -> torch.Tensor:
        """unabsorbed() over key blocks of kb (ML4_ATTN_KEY_BLOCK): per block, k / v through wkv_b once, then every
        Q tile's scores against the block and an online-softmax update of that tile's running max m, sum l and P.V
        acc (fp32). Tiles' running tensors are replaced, never written in place (slice writes across mark_steps gave
        wrong rows on the HPU), and there is a flush after each (block, Q tile)."""
        from .indexer import _mark_step

        pos = ctx.positions
        B, S = pos.shape
        Lk = ctx.max_len
        Hl, P = self.Hl, self.P
        scale = softmax_scale()
        k_sel = (
            topk.shape[-1]
            if topk is not None and topk.dtype != torch.bool
            else (min(TOPK, Lk) if topk is not None else TOPK)
        )
        qt = max(16, min(Q_TILE, ATTN_TILE_CAP // (B * Hl * kb * 4)))
        tiles = [(s0, min(S, s0 + qt)) for s0 in range(0, S, qt)]
        qf = [q[:, a:b].float().transpose(1, 2) for a, b in tiles]  # [B, Hl, St, 128]
        m = [torch.full((B, Hl, b - a, 1), NEG_FLOOR, dtype=torch.float32, device=q.device) for a, b in tiles]
        l = [torch.zeros(B, Hl, b - a, 1, dtype=torch.float32, device=q.device) for a, b in tiles]
        acc = [torch.zeros(B, Hl, b - a, V_DIM, dtype=torch.float32, device=q.device) for a, b in tiles]
        host_max = getattr(ctx, "host_max_pos", None)
        nblk = Lk // kb if host_max is None else min(Lk // kb, host_max // kb + 1)

        hpu = q.device.type == "hpu"

        def tile(i, s0, s1, k0, k1, kf, vf):
            p = pos[:, s0:s1]
            allowed = (
                (ent[:, k0:k1].unsqueeze(1) <= p.unsqueeze(-1)) & kvalid[:, k0:k1].unsqueeze(1) & (p >= 0).unsqueeze(-1)
            )
            if topk is not None and topk.dtype == torch.bool:
                allowed = allowed & (topk[:, s0:s1, k0:k1] | (p <= k_sel - 1).unsqueeze(-1))
            elif topk is not None:
                t = topk[:, s0:s1]
                sel = (
                    torch.zeros(B, s1 - s0, Lk, dtype=torch.float32, device=q.device).scatter_add_(
                        2, t.clamp(min=0), (t >= 0).float()
                    )[:, :, k0:k1]
                    > 0
                )
                allowed = allowed & (sel | (p <= k_sel - 1).unsqueeze(-1))
            sc = torch.matmul(qf[i], kf) * scale  # [B, Hl, St, kb]
            sc = sc.masked_fill(~allowed.unsqueeze(1), float("-inf"))
            mn = torch.maximum(m[i], sc.amax(-1, keepdim=True))
            e = torch.exp(sc - mn)
            corr = torch.exp(m[i] - mn)
            l[i] = l[i] * corr + e.sum(-1, keepdim=True)
            acc[i] = acc[i] * corr + torch.matmul(e, vf)
            m[i] = mn

        def block(j):
            k0, k1 = j * kb, (j + 1) * kb
            c, kpe = rows[:, k0:k1, :KV_RANK], rows[:, k0:k1, KV_RANK:]
            kvb = F.linear(c.float(), self.wkv_b.float()).view(B, kb, Hl, P + V_DIM)
            k = kvb[..., :P]
            if self.role.use_rope:
                k = torch.cat([k, kpe.float().unsqueeze(2).expand(B, kb, Hl, ROPE_DIM)], -1)
            kf = k.permute(0, 2, 3, 1)  # [B, Hl, 128, kb]
            vf = kvb[..., P:].transpose(1, 2)  # [B, Hl, kb, 128]
            for i, (s0, s1) in enumerate(tiles):
                if i and hpu:
                    _mark_step()
                tile(i, s0, s1, k0, k1, kf, vf)

        # One lazy graph per (block, Q tile), and the tile / block bodies are functions so their scores and k / v are
        # freed before each flush. The bridge keeps a device workspace as large as the largest graph's intermediates
        # for the life of the process: 8 tiles in one graph left 4.6 GiB at 256K keys, a graph per tile 1.0 GiB, at
        # the same speed (one layer, 2048 rows). Names still bound at a flush become graph
        # outputs, and with launches queued ahead of the device every queued block's outputs stayed allocated.
        for j in range(nblk):
            if j and q.device.type == "hpu":
                _mark_step()
            block(j)
        return torch.cat([(acc[i] / l[i].clamp_min(1e-30)).transpose(1, 2) for i in range(len(tiles))], 1).to(q.dtype)

    def absorbed(self, q, topk, ctx, src) -> torch.Tensor:
        """Decode form: gather the top-k latent rows and attend in latent space. q [B, S, Hl, 128] -> [B, S, Hl, 128]."""
        pos = ctx.positions
        B, S = pos.shape
        if dcp_rows(src, S):
            return self.absorbed_dcp(q, topk, ctx, src)
        if topk is None:
            topk = dense_topk(pos, min(TOPK, ctx.max_len))
        W_UK, W_UV = self.w_uk_uv()
        scale = softmax_scale()
        if not BF16_POINTS and q.device.type == "hpu" and decode_mm() and (S <= DECODE_S or _mm_prompt()):
            from . import k2attn

            if k2attn.ENABLED and S <= DECODE_S and k2attn.eligible(self, q, topk, src):
                return k2attn.absorbed(self, q, topk, ctx, src)  # ML4_ATTN_KERNEL=1: same bits as absorbed_mm
            return self.absorbed_mm(q, topk, ctx, src, W_UK, W_UV, scale)
        tile = S if S <= DECODE_S else max(1, Q_TILE // 2)
        out = torch.empty(B, S, self.Hl, V_DIM, dtype=q.dtype, device=q.device)
        for s0 in range(0, S, tile):
            s1 = min(S, s0 + tile)
            t = topk[:, s0:s1]
            # the addressing of one top-k tensor is the same for every layer reading it (memo: decode graphs). Keyed by
            # the tensor: in the dense regime every full layer returns the same memoised dense top-k. Safe within one
            # memo's life: Shared keeps every top-k tensor of the forward alive, so no id is reused.
            rows, ok = src.read(ctx, t.clamp(min=0), t >= 0, key=("topk", id(topk), s0))
            cf = rows[..., :KV_RANK].float()
            qn = q[:, s0:s1, :, : self.P].float()
            ql = torch.einsum("bshp,hpc->bshc", qn, W_UK.float())  # [B, St, Hl, 512]
            if BF16_POINTS:
                ql = ql.to(q.dtype).float()  # the PR's bmm output, bf16
            lg = torch.einsum("bshc,bskc->bshk", ql, cf)
            if self.role.use_rope:
                lg = lg + torch.einsum("bshr,bskr->bshk", q[:, s0:s1, :, self.P :].float(), rows[..., KV_RANK:].float())
            lg = (lg * scale).masked_fill(~ok.unsqueeze(2), float("-inf"))
            m = lg.amax(-1, keepdim=True).clamp_min(NEG_FLOOR)
            e = torch.exp(lg - m)
            pr = e / e.sum(-1, keepdim=True).clamp_min(1e-30)
            if BF16_POINTS:
                pr = pr.to(q.dtype).float()
            ol = torch.einsum("bshk,bskc->bshc", pr, cf)
            if BF16_POINTS:
                ol = ol.to(q.dtype).float()  # latent output, bf16
            out[:, s0:s1] = torch.einsum("bshc,hvc->bshv", ol, W_UV.float()).to(q.dtype)
            _tile_flush(S)
        return out

    def absorbed_mm(self, q, topk, ctx, src, W_UK, W_UV, scale) -> torch.Tensor:
        """absorbed() as batched matmuls over the (row, query) pairs (ML4_ATTN_DECODE_MM): decode / verify rows, and
        prompt chunks in Q tiles only under the test key attn_mm_prompt (the PPL check of this form). The
        einsums lowered to DMA transposes of the gathered [k, 576] rows and grew ~1.6 ms per stream per step."""
        B, S = ctx.positions.shape
        H, P = self.Hl, self.P
        tile = S if S <= DECODE_S else max(1, Q_TILE // 2)
        out = None if tile == S else torch.empty(B, S, H, V_DIM, dtype=q.dtype, device=q.device)
        for s0 in range(0, S, tile):
            s1 = min(S, s0 + tile)
            t = topk if tile == S else topk[:, s0:s1]
            rows, ok = src.read(ctx, t.clamp(min=0), t >= 0, key=("topk", id(topk), s0))  # [B, St, k, 576]
            St, k = s1 - s0, rows.shape[-2]
            r32 = rows.float().reshape(B * St, k, rows.shape[-1])
            cf = r32[..., :KV_RANK]
            qf = (q if tile == S else q[:, s0:s1]).float().reshape(B * St, H, q.shape[-1])
            ql = torch.matmul(qf[..., :P].transpose(0, 1), W_UK.float()).transpose(0, 1)  # [BS, H, 512]
            lg = torch.matmul(ql, cf.transpose(1, 2))  # [BS, H, k]
            if self.role.use_rope:
                lg = lg + torch.matmul(qf[..., P:], r32[..., KV_RANK:].transpose(1, 2))
            lg = (lg * scale).masked_fill(~ok.reshape(B * St, 1, k), float("-inf"))
            m = lg.amax(-1, keepdim=True).clamp_min(NEG_FLOOR)
            e = torch.exp(lg - m)
            pr = e / e.sum(-1, keepdim=True).clamp_min(1e-30)
            ol = torch.matmul(pr, cf)  # [BS, H, 512]
            o = torch.matmul(ol.transpose(0, 1), W_UV.float().transpose(1, 2)).transpose(0, 1)  # [BS, H, 128]
            o = o.reshape(B, St, H, V_DIM).to(q.dtype)
            if out is None:
                return o
            out[:, s0:s1] = o
            _tile_flush(S)
        return out

    def absorbed_dcp(self, q, topk, ctx, src) -> torch.Tensor:
        """absorbed() for decode / verify rows under the DCP layout (paged.py): topk holds this rank's selected keys
        [B, S, w] (indexer._select_dcp / dcp_dense). Every rank attends its own keys for all heads; the normalised
        partial outputs + LSE go back to the head owners (all-to-all) and merge by LSE."""
        import torch.distributed as dist

        group, r, n = dcp_group()
        B, S = ctx.positions.shape
        Hl = self.Hl
        W_UK, W_UV = self.w_uk_uv()
        if topk is None:
            topk = ctx.shared(("dcp_dense", TOPK), lambda: dcp_dense(ctx.positions, min(TOPK, ctx.max_len), n, r))
        rows, ok = src.read(ctx, topk.clamp(min=0), topk >= 0, key=("dcp", id(topk)), local=True)
        mm = not BF16_POINTS and decode_mm()
        # absorbed queries of this rank's heads -> every rank, all heads
        if mm:  # absorbed_mm's batched matmuls over the (row, query) pairs
            qf = q.float().reshape(B * S, Hl, q.shape[-1])
            ql = torch.matmul(qf[..., : self.P].transpose(0, 1), W_UK.float()).transpose(0, 1).reshape(B, S, Hl, -1)
        else:
            ql = torch.einsum("bshp,hpc->bshc", q[..., : self.P].float(), W_UK.float())  # [B, S, Hl, 512]
        if self.role.use_rope:
            ql = torch.cat([ql, q[..., self.P :].float()], -1)  # [B, S, Hl, 576]
        Wq = ql.shape[-1]
        qa = dcp_all_gather(ql).permute(1, 2, 0, 3, 4).reshape(B, S, n * Hl, Wq)  # global head order
        if mm:
            k = rows.shape[-2]
            r32 = rows.float().reshape(B * S, k, rows.shape[-1])
            lg = torch.matmul(qa.reshape(B * S, n * Hl, Wq), r32[..., :Wq].transpose(1, 2)) * softmax_scale()
            lg = lg.masked_fill(~ok.reshape(B * S, 1, k), float("-inf"))
        else:
            lg = torch.einsum("bshc,bskc->bshk", qa, rows[..., :Wq].float()) * softmax_scale()
            lg = lg.masked_fill(~ok.unsqueeze(2), float("-inf"))
        m = lg.amax(-1, keepdim=True).clamp_min(NEG_FLOOR)
        e = torch.exp(lg - m)
        l = e.sum(-1, keepdim=True)
        if mm:
            ol = (torch.matmul(e, r32[..., :KV_RANK]) / l.clamp_min(1e-30)).reshape(B, S, n * Hl, KV_RANK)
            m, l = m.reshape(B, S, n * Hl, 1), l.reshape(B, S, n * Hl, 1)
        else:
            ol = torch.einsum("bshk,bskc->bshc", e, rows[..., :KV_RANK].float()) / l.clamp_min(1e-30)
        lse = (m + torch.log(l)).masked_fill(l == 0, NEG_FLOOR)  # [B, S, H, 1]; no keys: weight 0 (or 0 / 0 rows)
        send = torch.cat([ol, lse], -1).view(B, S, n, Hl, KV_RANK + 1).permute(2, 0, 1, 3, 4).contiguous()
        recv = torch.empty_like(send)
        if send.device.type == "cpu":  # gloo (tests): no all_to_all
            parts = list(recv.unbind(0))
            for j in range(n):
                dist.scatter(parts[j], list(send.unbind(0)) if r == j else None, src=j, group=group)
        else:
            dist.all_to_all_single(recv, send, group=group)  # [src, B, S, Hl, 513]
        w = torch.softmax(recv[..., KV_RANK], dim=0).unsqueeze(-1)
        ol = (w * recv[..., :KV_RANK]).sum(0)  # [B, S, Hl, 512]
        if BF16_POINTS:
            ol = ol.to(q.dtype).float()
        if mm:
            o = torch.matmul(ol.reshape(B * S, Hl, KV_RANK).transpose(0, 1), W_UV.float().transpose(1, 2))
            return o.transpose(0, 1).reshape(B, S, Hl, V_DIM).to(q.dtype)
        return torch.einsum("bshc,hvc->bshv", ol, W_UV.float()).to(q.dtype)
