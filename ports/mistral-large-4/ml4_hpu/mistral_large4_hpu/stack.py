"""The Mistral Large 4 decoder stack on the HPU, one TP/EP rank.

    x = embed_vocab_parallel(ids)                                   # all-reduce
    for L in 0..n-1:
        a = rmsnorm(x, attention_norm);  x += allreduce(attn_L(a))  # ML4Attention: indexer, MLA, wo partial sum
        f = rmsnorm(x, ffn_norm);        x += allreduce(ffn_L(f))   # L0 dense SwiGLU, else ML4MoE (routed + shared)
    logits = 30 tanh(lm_head_fp32(rmsnorm(x, norm)) / 30)           # vocab-parallel shard [.., V / tp]

Each linear holds this rank's shard (column: out / tp rows; row: in / tp columns, returning the partial sum) and
the stack all-reduces twice per layer (attention, FFN), as GLM/DSV. The residual adds are bf16 (one rounding, as
HF and the PR's fused add + RMSNorm). Top-k tensors pass down the layer loop in attention.Shared, never a module
buffer. The loader (loader.py) fills every parameter; the RoPE / log(p+1) tables are built on the CPU from the
loaded lerope.alpha by rope() (once per process) and materialized on the device before the first forward.

No vLLM imports: the gate runs this directly (TP1 and TP2 via torch.distributed); vllm_model.py wraps it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from . import lean, prof
from .attention import RMS_EPS, LayerRole, ML4Attention, Shared, layer_roles
from . import moe as _moe
from .moe import GatedFFN, ML4MoE
from .numerics import rms_norm, softcap
from .rope import RopeCache


@dataclass
class ML4Dims:
    """The text dims the HPU modules read (attribute names as the HF / vLLM text config)."""

    hidden_size: int = 6144
    num_attention_heads: int = 64
    moe_intermediate_size: int = 6144
    intermediate_size: int = 24576
    vocab_size: int = 204800
    n_shared_experts: int = 1
    num_hidden_layers: int = 72
    softcap: float = 30.0

    @classmethod
    def from_params(cls, p: dict) -> "ML4Dims":
        moe = p.get("moe") or {}
        return cls(
            p["dim"],
            p["n_heads"],
            moe.get("expert_hidden_dim", p["hidden_dim"]),
            p["hidden_dim"],
            p["vocab_size"],
            moe.get("num_shared_experts", 1),
            p["n_layers"],
            float(p.get("softmax_tanh_gating") or 30.0),
        )


class LocalLinear(nn.Module):
    """y = x W^T with W this rank's shard [out_local, in_local]."""

    def __init__(self, n_in: int, n_out: int, dtype=torch.bfloat16, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_out, n_in, dtype=dtype, device=device), requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight)


def local_linear(tp_size: int, device=None):
    """linear(name, in, out, kind, dtype) for the HPU modules."""

    def make(name, n_in, n_out, kind, dtype):
        if kind == "column":
            assert n_out % tp_size == 0, (name, n_out, tp_size)
            n_out //= tp_size
        elif kind == "row":
            assert n_in % tp_size == 0, (name, n_in, tp_size)
            n_in //= tp_size
        return LocalLinear(n_in, n_out, dtype or torch.bfloat16, device)

    return make


# ML4_PREFILL_MARK=1: prompt chunks flush the lazy graph after each layer's attention and FFN (htorch mark_step).
# Without it a prompt chunk's lazy graphs span attention + indexer + MoE of a layer and the next layer's attention
# (the only flush is v8p's); their recipe workspace stays allocated after the first chunk (+8 GiB per card,
# ) and the 64K live peak sat within 0.3 GiB of the allocator limit.
PREFILL_MARK = os.environ.get("ML4_PREFILL_MARK", "0") == "1"
DECODE_S = 8
# ML4_PROMPT_LAYER_SYNC_KEYS (control key prompt_layer_sync_keys; test key, default 0 = never): prompt chunks over a
# key range wider than this synchronize the device after every layer; with the control key mem_layers rank 0 then
# prints each layer's device peak and in-use (how the per-layer growth was found).
PROMPT_LAYER_SYNC_KEYS = int(os.environ.get("ML4_PROMPT_LAYER_SYNC_KEYS", "0"))


def _flush(x):
    if x.shape[1] <= DECODE_S or x.device.type != "hpu" or _moe.GRAPH_PROMPT[0]:
        return  # (inside a prompt HPU graph the flush is never taken: untested there)
    from .control import get as ctl_get

    if ctl_get("prefill_mark", PREFILL_MARK):
        from .indexer import _mark_step

        _mark_step()


class ML4Layer(nn.Module):
    def __init__(self, dims: ML4Dims, role: LayerRole, linear, tp_rank: int, tp_size: int, placement: str):
        super().__init__()
        D = dims.hidden_size
        self.role = role
        self.attn_norm = nn.Parameter(torch.ones(D, dtype=torch.bfloat16), requires_grad=False)
        self.ffn_norm = nn.Parameter(torch.ones(D, dtype=torch.bfloat16), requires_grad=False)
        self.attn = ML4Attention(dims, role, linear, tp_size)
        if role.is_moe:
            self.ffn = ML4MoE(dims, linear, tp_rank, tp_size, placement, role.layer_id)
        else:
            self.ffn = GatedFFN("feed_forward", D, dims.intermediate_size, linear)

    def forward(self, x, ctx, caches, shared: Shared, rope: RopeCache, allreduce):
        if x.shape[1] <= DECODE_S and lean.ready(self) and lean.enabled():
            return lean.layer_forward(self, x, ctx, caches, shared, rope, allreduce)
        h = self.attn(
            rms_norm(
                x,
                self.attn_norm,
                RMS_EPS,
                r5=not self.role.has_indexer,
                dual=("rope" if self.role.use_rope else "nope") if self.role.has_indexer else False,
            ),
            ctx,
            caches,
            shared,
            rope,
        )
        prof.lap("attn")
        x = x + allreduce(h)
        _flush(x)
        prof.lap("ar")
        h = rms_norm(x, self.ffn_norm, RMS_EPS, r5=True)
        y = self.ffn(h, ctx.valid) if self.role.is_moe else self.ffn(h)
        prof.lap("ffn")
        x = x + allreduce(y)
        _flush(x)
        prof.lap("ar")
        return x


class ML4Stack(nn.Module):
    """Text model of one TP rank (EP = TP: experts split over the same ranks)."""

    def __init__(
        self,
        params: dict,
        tp_rank: int = 0,
        tp_size: int = 1,
        placement: str = "A",
        max_pos: int = 65536,
        device=None,
        n_layers: int | None = None,
        group=None,
    ):
        super().__init__()
        self.dims = ML4Dims.from_params(params)
        self.roles = layer_roles(params, n_layers)
        self.tp_rank, self.tp_size, self.max_pos, self.group = tp_rank, tp_size, max_pos, group
        D, V = self.dims.hidden_size, self.dims.vocab_size
        assert V % tp_size == 0
        self.vp = V // tp_size
        lin = local_linear(tp_size, device)
        self.embed = nn.Parameter(torch.empty(self.vp, D, dtype=torch.bfloat16, device=device), requires_grad=False)
        self.layers = nn.ModuleList(ML4Layer(self.dims, r, lin, tp_rank, tp_size, placement) for r in self.roles)
        self.norm = nn.Parameter(torch.ones(D, dtype=torch.bfloat16, device=device), requires_grad=False)
        self.lm_head = nn.Parameter(torch.empty(self.vp, D, dtype=torch.float32, device=device), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(32, dtype=torch.bfloat16), requires_grad=False)  # lerope.alpha
        self._rope: RopeCache | None = None
        # DSpark taps: tap(layer, x) gets the residual stream LEAVING each layer (71: before the final
        # norm) inside the forward; spec.install_taps sets it, None costs nothing
        self.tap = None

    # ------------------------------------------------------------------ helpers
    def allreduce(self, t: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return t
        import torch.distributed as dist

        dist.all_reduce(t, group=self.group)
        return t

    def rope(self) -> RopeCache:
        """The model's one table set, built on the CPU from the loaded alpha, materialized on the stack's device."""
        if self._rope is None:
            self._rope = RopeCache(self.max_pos, self.alpha.detach().cpu())
            self._rope.materialize(self.norm.device)
        return self._rope

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        """Vocab-parallel lookup: rows of other ranks contribute 0, then the all-reduce."""
        if self.tp_size == 1:
            return self.embed[ids]
        loc = ids - self.tp_rank * self.vp
        ok = (loc >= 0) & (loc < self.vp)
        e = self.embed[loc.clamp(0, self.vp - 1)] * ok.unsqueeze(-1).to(self.embed.dtype)
        return self.allreduce(e)

    # ------------------------------------------------------------------ forward
    def forward(self, ids: torch.Tensor, ctx, caches: dict) -> torch.Tensor:
        """ids [B, S] -> final hidden [B, S, D] (pre-norm residual stream after the last layer)."""
        rope = self.rope()
        prof.begin(ids)
        x = self.embed_tokens(ids)
        shared = Shared()
        from .control import get as ctl_get

        sk = int(ctl_get("prompt_layer_sync_keys", PROMPT_LAYER_SYNC_KEYS))
        layer_sync = (
            sk > 0
            and ids.shape[1] > DECODE_S
            and ids.device.type == "hpu"
            and ctx.max_len > sk
            and not _moe.GRAPH_PROMPT[0]
        )
        # a prompt chunk's top-k is a bool mask [B, S, Lk] (0.25 GiB at 128K keys, 2048 rows): drop each one after the
        # last layer that reads it, instead of keeping every full layer's mask to the end of the forward (+0.25 GiB per
        # full layer held at 128K). Not with a memo (decode graphs key it by tensor id, so
        # those keep every top-k alive), and the eager stack returns only x.
        last = self._index_last() if (ids.shape[1] > DECODE_S and ctx.memo is None) else None
        for i, layer in enumerate(self.layers):
            x = layer(x, ctx, caches, shared, rope, self.allreduce)
            if self.tap is not None:
                self.tap(i, x)
            if last is not None:
                for src in last.get(i, ()):
                    shared.topk.pop(src, None)
            if layer_sync:
                from .indexer import _mark_step

                _mark_step()
                torch.hpu.synchronize()
                if ctl_get("mem_layers", False) and self.tp_rank == 0:  # test key: per-layer device peak
                    st = torch.hpu.memory_stats()
                    print(
                        f"ml4: mem layer {i} rows {ids.shape[1]} width {ctx.max_len}: peak "
                        f"{st.get('MaxInUse', 0) / 2**30:.2f} GiB, in use {st.get('InUse', 0) / 2**30:.2f}",
                        flush=True,
                    )
                    torch.hpu.reset_peak_memory_stats()
        prof.end(ids.shape[1], ctx.max_len, self.tp_rank)
        return x

    def _index_last(self) -> dict:
        """{layer index i: [top-k sources whose last reader is layer i]} (roles are fixed: computed once)."""
        if getattr(self, "_last_reader", None) is None:
            last_at = {}
            for i, layer in enumerate(self.layers):
                last_at[layer.role.index_source] = i
            out: dict = {}
            for src, i in last_at.items():
                out.setdefault(i, []).append(src)
            self._last_reader = out
        return self._last_reader

    def logits(self, h: torch.Tensor, gather: bool = True) -> torch.Tensor:
        """Final RMSNorm, fp32 lm_head, soft-cap -> [.., V] fp32 (gather=True all-gathers the vocab shards)."""
        z = softcap(F.linear(rms_norm(h, self.norm, RMS_EPS).float(), self.lm_head), self.dims.softcap)
        if not gather or self.tp_size == 1:
            return z
        import torch.distributed as dist

        parts = [torch.empty_like(z) for _ in range(self.tp_size)]
        dist.all_gather(parts, z.contiguous(), group=self.group)
        return torch.cat(parts, -1)
