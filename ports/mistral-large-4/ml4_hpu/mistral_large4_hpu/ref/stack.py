"""Pure-torch Mistral Large 4 text stack (the CPU oracle).

    from mistral_large4_hpu.ref import RefConfig, RefModel, RefOptions, TensorSource
    src = TensorSource(ckpt_dir)
    m = RefModel(src, RefConfig.load(ckpt_dir), RefOptions(dtype="bf16"))
    out = m.forward(ids, chunks=None, taps=(1, 18))   # out.logits [T, V] fp32, out.taps {l: [T, d]}

One sequence per call, positions 0..T-1. `chunks` (list of chunk lengths) runs a chunked prefill:
each chunk goes through all layers with the per-layer caches of the earlier chunks (latent
[c | k_pe] rows, indexer-K rows), exactly as an engine would.
"""

from __future__ import annotations

import dataclasses

import torch

from . import attention as A
from . import indexer as I
from . import moe as M
from . import numerics as N
from .layer_spec import RefConfig, get_layer_specs
from .rope import cos_sin, inv_freq_from_cfg
from .weights import ExpertSource, TensorSource

F32, BF16 = torch.float32, torch.bfloat16


@dataclasses.dataclass
class RefOptions:
    dtype: str = "bf16"  # "fp32" (HF model.float() comparison) | "bf16" (engine rounding)
    fp8_indexer: bool = True  # PR: FP8 e4m3 + ue8m0 fake-quant of indexer q/k (False = HF)
    attn_form: str = "unabsorbed"  # "unabsorbed" (HF / HPU prefill) | "absorbed" (PR / HPU decode)
    act_mxfp4: bool = False  # W4A4: MXFP4 activations into the routed experts (D13)
    skip_short_topk: bool = True  # pos+1 <= topk selects every visible key
    rope_table_dtype: str = "fp32"  # "bf16" reproduces the PR's bf16 cos_sin_cache
    hf_numerics: bool = False  # bf16 only: HF's fp32 indexer weights_proj (PR: bf16 GEMM)
    topk_ties: str = "stable"  # "stable": ties go to the earlier key; "torch": torch.topk (HF)
    # ---- controls: deliberately wrong variants that must fail their gates
    mscale: bool = False  # x3.106 logits (YaRN mscale_all_dim trap)
    no_alpha: bool = False  # LeRoPE off
    no_ssmax: bool = False  # SSMax off (MLA and indexer)
    rope_on_nope: bool = False  # rotate NoPE layers (MLA q/k and their indexer)
    no_kv_share: bool = False  # KV readers see a zero latent cache
    index_share_self: bool = False  # shared-indexer layers recompute top-k (source layer's indexer weights)
    rope_noninterleaved: bool = False  # NeoX half-split rotation instead of GPT-J pairs
    stock_routing: bool = False  # no top-k renormalisation
    bf16_softmax: bool = False  # attention softmax in bf16
    no_softcap: bool = False  # raw lm_head logits

    @property
    def dt(self):
        return F32 if self.dtype == "fp32" else BF16


# CLI control flag -> RefOptions overrides (the control names).
CONTROLS = {
    "mscale": {"mscale": True},
    "no-alpha": {"no_alpha": True},
    "no-ssmax": {"no_ssmax": True},
    "rope-on-nope": {"rope_on_nope": True},
    "no-fp8-indexer": {"fp8_indexer": False},
    "no-kv-share": {"no_kv_share": True},
    "index-share-self": {"index_share_self": True},
    "rope-noninterleaved": {"rope_noninterleaved": True},
    "stock-routing": {"stock_routing": True},
    "act-mxfp4": {"act_mxfp4": True},
    "bf16-softmax": {"bf16_softmax": True},
    "no-softcap": {"no_softcap": True},
}


class SeqState:
    """Per-sequence caches: latent rows c[l] [S, 512], kpe[l] [S, 64] for KV-owning layers,
    indexer-K rows ik[l] [S, D]; kpos [S] absolute positions of the rows."""

    def __init__(self):
        self.c, self.kpe, self.ik = {}, {}, {}
        self.kpos = torch.zeros(0, dtype=torch.long)

    @staticmethod
    def _app(d, l, t):
        d[l] = t if l not in d else torch.cat([d[l], t], 0)


@dataclasses.dataclass
class ForwardOut:
    logits: torch.Tensor | None
    taps: dict
    sel: dict  # layer -> [T, S] selection mask (full layers only; last chunk if chunked)
    hidden: torch.Tensor


class RefModel:
    def __init__(
        self,
        src: TensorSource,
        cfg: RefConfig,
        opts: RefOptions | None = None,
        keep_weights: bool = True,
        expert_cache: bool = False,
    ):
        self.src, self.cfg = src, cfg
        self.opts = opts or RefOptions()
        self.specs = get_layer_specs(cfg)
        self.keep, self.expert_cache = keep_weights, expert_cache
        self.W: dict[int, dict] = {}
        self._alpha = src.f32("lerope.alpha") if src.has("lerope.alpha") else None
        self._emb = self._head = self._norm = None
        # layer -> [T, S] bool selection to use instead of the indexer's (single-pass runs only):
        # parity against another implementation's top-k (HF, or the HPU)
        self.sel_override: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------ weights
    def load_layer(self, l: int) -> dict:
        if l in self.W:
            return self.W[l]
        s, sp, p = self.src, self.specs[l], f"layers.{l}."
        at = p + "attention."
        w = {
            "attn_norm": s.f32(p + "attention_norm.weight"),
            "ffn_norm": s.f32(p + "ffn_norm.weight"),
            "wq": s.linear(at + "wq"),
            "wkv_b": s.linear(at + "wkv_b"),
            "wo": s.linear(at + "wo"),
            "ssmax_s": s.f32(at + "ssmax_scaling.s"),
            "ssmax_b": s.f32(at + "ssmax_scaling.b"),
        }
        if not sp.shares_kv:
            w["wkv_a"] = s.linear(at + "wkv_a_with_mqa")
            w["kv_a_norm"] = s.f32(at + "kv_a_norm.weight")
        if sp.has_indexer:
            ix = at + "indexer."
            w["idx"] = {
                "q_norm_w": s.f32(ix + "q_input_norm.weight"),
                "q_norm_b": s.f32(ix + "q_input_norm.bias"),
                "wq_b": s.linear(ix + "wq_b"),
                "wk": s.linear(ix + "wk"),
                "k_norm_w": s.f32(ix + "k_norm.weight"),
                "k_norm_b": s.f32(ix + "k_norm.bias"),
                "weights_proj": s.linear(ix + "weights_proj"),
                "ssmax_s": s.f32(ix + "ssmax_scaling.s"),
                "ssmax_b": s.f32(ix + "ssmax_scaling.b"),
            }
        if sp.is_moe:
            w["gate"] = s.f32(p + "gate.weight")
            w["shared"] = {k: s.linear(p + f"shared_experts.{k}") for k in ("w1", "w2", "w3")}
            w["experts"] = ExpertSource(s, l, cache=self.expert_cache)
        else:
            w["ffn"] = {k: s.linear(p + f"feed_forward.{k}") for k in ("w1", "w2", "w3")}
        if self.keep:
            self.W[l] = w
        return w

    def drop_layer(self, l: int) -> None:
        self.W.pop(l, None)

    def embedding(self):
        if self._emb is None:
            self._emb = self.src.get("tok_embeddings.weight")
        return self._emb

    def head_weights(self):
        if self._head is None:
            self._head = self.src.f32("output.weight")  # fp32 lm_head (D12)
            self._norm = self.src.f32("norm.weight")
        return self._head, self._norm

    # ------------------------------------------------------------------ pieces
    def inv_freq(self):
        return inv_freq_from_cfg(self.cfg, None if self.opts.no_alpha else self._alpha)

    def rope_tables(self, pos):
        return cos_sin(pos, self.inv_freq(), F32 if self.opts.rope_table_dtype == "fp32" else BF16)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embedding()[ids].to(self.opts.dt)

    def head(self, x: torch.Tensor) -> torch.Tensor:
        """Final RMSNorm (dt), fp32 lm_head, 30*tanh(z/30) in fp32 -> [T, V] fp32."""
        Wh, wn = self.head_weights()
        h = N.rmsnorm(x, wn, self.cfg.rms_norm_eps, self.opts.dt)
        z = h.to(F32) @ Wh.t()
        return z if self.opts.no_softcap else N.softcap(z, self.cfg.softcap)

    def _lam(self, logp, s, b):
        if self.opts.no_ssmax:
            return torch.ones(logp.shape[0], s.shape[0], dtype=self.opts.dt)
        return N.ssmax_scale(logp, s, b, self.opts.dt)

    def layer_forward(
        self, l: int, x: torch.Tensor, pos: torch.Tensor, st: SeqState, sels: dict, record: dict | None = None
    ) -> torch.Tensor:
        """One chunk (rows at absolute positions `pos`) through layer l. Appends this layer's cache
        rows to `st` (callers append st.kpos once per chunk BEFORE layer 0, see forward()). `sels`
        holds the selection mask of each full layer for this chunk (top-k pass-down)."""
        o, cfg, sp, dt = self.opts, self.cfg, self.specs[l], self.opts.dt
        w = self.load_layer(l)
        rd = cfg.qk_rope_head_dim
        inter = not o.rope_noninterleaved
        cos, sin = self.rope_tables(pos)
        logp = N.log_positions(pos)
        a = N.rmsnorm(x, w["attn_norm"], cfg.rms_norm_eps, dt)
        kpos = st.kpos
        # ---- indexer / top-k
        idx_layer = l if sp.has_indexer else (sp.index_source if o.index_share_self else None)
        if idx_layer is not None:
            iw = self.load_layer(idx_layer)["idx"]
            rope_i = sp.indexer_use_rope or o.rope_on_nope
            lam_i = self._lam(logp, iw["ssmax_s"], iw["ssmax_b"])
            qi = I.indexer_q(a, iw, lam_i, cos, sin, rope_i, rd, dt, o.fp8_indexer, inter)
            ki = I.indexer_k(a, iw, cos, sin, rope_i, rd, dt, o.fp8_indexer, inter)
            SeqState._app(st.ik, l, ki)
            wts = I.indexer_weights(a.to(F32) if o.hf_numerics else a, iw, F32 if o.hf_numerics else dt)
            sc = I.indexer_scores(qi, st.ik[l], wts, pos, kpos)
            sel = I.select_topk(sc, pos, kpos, cfg.index_topk, o.skip_short_topk, o.topk_ties)
            if l in self.sel_override:
                sel = self.sel_override[l][pos][:, : kpos.shape[0]] & (kpos[None, :] <= pos[:, None])
            sels[l] = sel
            if record is not None:
                record["index_scores"] = sc
                record["boundary_ties"] = I.boundary_ties(sc, pos, cfg.index_topk)
        else:
            sel = sels[sp.index_source]
        # ---- MLA
        lam = self._lam(logp, w["ssmax_s"], w["ssmax_b"])
        q = A.mla_q(
            a,
            w,
            sp.use_rope,
            cfg.num_heads,
            rd,
            lam,
            cos,
            sin,
            dt,
            inter,
            rope_on_nope=o.rope_on_nope and not sp.use_rope,
        )
        if not sp.shares_kv:
            c, kpe = A.mla_kv(a, w, sp.use_rope, cfg.kv_lora_rank, rd, cos, sin, dt, cfg.rms_norm_eps, inter)
            SeqState._app(st.c, l, c)
            SeqState._app(st.kpe, l, kpe)
        if sp.shares_kv and o.no_kv_share:
            C, KPE = torch.zeros_like(st.c[sp.kv_source]), torch.zeros_like(st.kpe[sp.kv_source])
        else:
            C, KPE = st.c[sp.kv_source], st.kpe[sp.kv_source]
        C, KPE = C[: kpos.shape[0]], KPE[: kpos.shape[0]]
        ron = o.rope_on_nope and not sp.use_rope
        if o.attn_form == "absorbed" and not ron:
            att = A.attend_absorbed(
                q, C, KPE, w["wkv_b"], sel, cfg.v_head_dim, dt, sp.use_rope, o.mscale, o.bf16_softmax
            )
        else:
            att = A.attend_unabsorbed(
                q,
                C,
                KPE,
                w["wkv_b"],
                sel,
                cfg.v_head_dim,
                dt,
                sp.use_rope,
                o.mscale,
                o.bf16_softmax,
                rope_on_nope_kcs=self.rope_tables(kpos) if ron else None,
                interleaved=inter,
            )
        x = N.rnd(x.to(F32) + N.linear(att, w["wo"], dt).to(F32), dt)
        # ---- FFN
        f = N.rmsnorm(x, w["ffn_norm"], cfg.rms_norm_eps, dt)
        if sp.is_moe:
            y, ids, wts = M.moe_forward(
                f,
                w["gate"],
                w["experts"],
                w["shared"],
                cfg.num_experts_per_tok,
                dt,
                renormalize=not o.stock_routing,
                scaling=cfg.routed_scaling_factor,
                act_mxfp4=o.act_mxfp4,
                return_routing=True,
            )
            if record is not None:
                record["router_ids"], record["router_w"] = ids, wts
        else:
            fw = w["ffn"]
            y = M.swiglu_ffn(f, fw["w1"], fw["w3"], fw["w2"], dt)
        if record is not None:
            record["sel"] = sel
        return N.rnd(x.to(F32) + y.to(F32), dt)

    # ------------------------------------------------------------------ driver
    def forward(
        self,
        ids: torch.Tensor,
        chunks: list[int] | None = None,
        taps=(),
        logits: bool = True,
        layers: range | None = None,
        hook=None,
    ) -> ForwardOut:
        """Teacher-forced prefill of one sequence. taps: layer indices whose OUTPUT residual stream
        (pre final norm; = the stream entering layer l+1) is returned, e.g. DSpark (1, 18, 36, 54, 71).
        hook(l, x_in, x_out, record) is called per layer per chunk (per-layer dumps)."""
        T = ids.shape[0]
        chunks = chunks or [T]
        assert sum(chunks) == T
        layers = layers or range(self.cfg.num_layers)
        st = SeqState()
        outs, tap_out, sel_last, start = [], {l: [] for l in taps}, {}, 0
        hid = []
        for n in chunks:
            pos = torch.arange(start, start + n)
            st.kpos = torch.cat([st.kpos, pos])
            x = self.embed(ids[start : start + n])
            sels = {}
            for l in layers:
                rec = {} if hook is not None else None
                x_in = x
                x = self.layer_forward(l, x, pos, st, sels, rec)
                if hook is not None:
                    hook(l, x_in, x, rec)
                if l in tap_out:
                    tap_out[l].append(x)
            sel_last = sels
            hid.append(x)
            if logits:
                outs.append(self.head(x))
            start += n
        return ForwardOut(
            torch.cat(outs) if logits else None, {l: torch.cat(v) for l, v in tap_out.items()}, sel_last, torch.cat(hid)
        )
