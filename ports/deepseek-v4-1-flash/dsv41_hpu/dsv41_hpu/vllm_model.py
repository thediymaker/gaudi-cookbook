# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash as a vLLM model on Gaudi (vllm-gaudi v0.30.0, lazy mode, TP 8 / EP 8).

Wraps the vLLM-independent stack (core.py, csa.py) with:
  * DSV41KVCache       one paged cache per (layer, kind), declared as an MLAAttentionSpec of its own width. vLLM
                       groups them all into ONE UniformTypeKVCacheSpecs group, so every cache shares the request's
                       single block table; the HPU runner allocates each at its own head size.
  * DSV41MoEHPU        routed experts on the native bias_mxfp4_fused_weights op with an exact emulation of V4.1's
                       clamped SwiGLU (k=7), the reference sqrtsoftplus router, and a TP-sharded clamped shared
                       expert; one all-reduce per MoE layer. With DSV41_MXK=1 calls of <= 16 tokens (decode) run on
                       the MXFP4 TPC kernels instead (mxk.py).
  * parallel linears   wq_b column-parallel, wo_b row-parallel, the rest replicated (the indexer replicated as
                       upstream, so no [S, L] score all-reduce).
  * Engram TP          rank r holds hash columns [3r, 3r+3) of each table; looked-up rows are all-gathered back to
                       24 columns before the replicated wkv.
  * load_weights       reads the converted checkpoint directly with ranged safetensors reads: each rank touches
                       only its 48 experts and its engram buckets.

Checkpoint: the output of convert/convert_to_hpu_mxfp4.py. Serving: text only (--language-model-only), no
speculative decoding, no async scheduling (host-side engram hashing needs every sampled token),
VLLM_CONTIGUOUS_PA=false.

Decode graphs (--enforce-eager stays on: a whole-model HPU graph never finished compiling, since compile time grows
faster than linearly with graph size):
  DSV41_LAYER_GRAPHS=1   decode steps (S == 1) replay HPU graphs per layer group; prefill stays eager (lazy-mode
                         graphs cut after every layer). Off: eager decode (~1.5 tok/s).
  DSV41_LAYER_GROUP=N    layers per decode graph (default 1). Bigger groups mean fewer graph launches per step.
DSV41_REPLAY_KEEP=140: prompts run the decoder layers on their last 140 tokens only (replay.py).
"""

from __future__ import annotations

import logging
import os

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from . import mxk, replay, runner_patch
from .core import BlockStep, DSV41Engram, DSV41Stack, GroupStep, run_groups, run_steps
from .csa import csa_roles
from .engram import EngramHasher, bucket_primes, build_compressed_token_map
from .moe import route, swiglu_clamped
from .paged import TAG_MODE, TaggedCache, tag_violations

logger = logging.getLogger("vllm.dsv41_hpu")  # under "vllm", so vLLM's handlers print it

# SwiGLU clamp emulation shift: on saturated input k=7 is within 0.0036 relative error of the exact clamp (k=0 is
# 6.5% off, the unclamped silu op 57-65%)
SWIGLU_K = 7
LAYER_GRAPHS = os.environ.get("DSV41_LAYER_GRAPHS", "0") == "1"
LAYER_GROUP = int(os.environ.get("DSV41_LAYER_GROUP", "1"))
TAG_LOG_EVERY = 512  # count mode: forwards between "tag check" lines


# ------------------------------------------------------------------- KV caches
class DSV41KVCache(nn.Module, AttentionLayerBase):
    """One paged cache view. The HPU runner hands bind_kv_cache a (k, v, k_scales,
    v_scales) tuple; v is None for MLA-type models. Tags are ours, same row count."""

    def __init__(
        self, prefix: str, width: int, dtype: torch.dtype, block_size: int, vllm_config: VllmConfig, ratio: int = 1
    ):
        super().__init__()
        self.prefix, self.width, self.cache_dtype, self.block_size = prefix, width, dtype, block_size
        self.ratio = ratio  # tokens per row: >1 for compressor-entry caches
        ctx = vllm_config.compilation_config.static_forward_context
        if prefix in ctx:
            raise ValueError(f"duplicate layer name {prefix}")
        ctx[prefix] = self
        self.kv_cache = None
        self.tags = None

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        return MLAAttentionSpec(
            block_size=self.block_size, num_kv_heads=1, head_size=self.width, dtype=self.cache_dtype
        )

    def get_attn_backend(self):
        from vllm_gaudi.attention.backends.hpu_attn import HPUMLAAttentionBackend

        return HPUMLAAttentionBackend

    def bind_kv_cache(self, kv_cache) -> None:
        t = kv_cache[0] if isinstance(kv_cache, tuple | list) else kv_cache
        assert t.dim() == 2 and t.shape[1] == self.width, (self.prefix, tuple(t.shape), self.width)
        self.kv_cache = t
        self.tags = torch.full((t.shape[0], 2), -1, dtype=torch.int32, device=t.device)

    def view(self, device) -> TaggedCache:
        if self.kv_cache is None:  # profiling run before allocation: one dump block
            self.kv_cache = torch.zeros(self.block_size, self.width, dtype=self.cache_dtype, device=device)
            self.tags = torch.full((self.block_size, 2), -1, dtype=torch.int32, device=device)
        return TaggedCache(self.kv_cache, self.tags, self.ratio, label=self.prefix.rsplit("cache_", 1)[-1])


# ------------------------------------------------------------------------ MoE
class DSV41MoEHPU(nn.Module):
    def __init__(self, cfg, layer_id: int, prefix: str, tp_rank: int, tp_size: int):
        super().__init__()
        E, inter, D = cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.hidden_size
        assert E % tp_size == 0 and inter % tp_size == 0
        self.E, self.I, self.D = E, inter, D
        self.n_local, self.e0 = E // tp_size, tp_rank * (E // tp_size)
        self.topk, self.route_scale, self.limit = cfg.num_experts_per_tok, cfg.routed_scaling_factor, cfg.swiglu_limit
        self.tp_size = tp_size
        self.gate_w = nn.Parameter(torch.empty(E, D, dtype=torch.bfloat16), requires_grad=False)
        self.gate_bias = nn.Parameter(torch.empty(E, dtype=torch.float32), requires_grad=False)
        u8 = dict(dtype=torch.uint8)
        self.w13_weight = nn.Parameter(torch.empty(self.n_local, 2 * inter, D // 2, **u8), requires_grad=False)
        self.w13_weight_scale = nn.Parameter(torch.empty(self.n_local, 2 * inter, D // 32, **u8), requires_grad=False)
        self.w2_weight = nn.Parameter(torch.empty(self.n_local, D, inter // 2, **u8), requires_grad=False)
        self.w2_weight_scale = nn.Parameter(torch.empty(self.n_local, D, inter // 32, **u8), requires_grad=False)
        self.shared_w13 = MergedColumnParallelLinear(
            D, [inter, inter], bias=False, params_dtype=torch.bfloat16, prefix=f"{prefix}.shared_experts.w13"
        )
        self.shared_w2 = RowParallelLinear(
            inter,
            D,
            bias=False,
            params_dtype=torch.bfloat16,
            reduce_results=False,
            prefix=f"{prefix}.shared_experts.w2",
        )
        self.moe_op = None
        self._mxk = None
        self.layer_id = layer_id

    def finalize(self) -> None:
        """After loading: fold the clamp emulation into the kernel inputs and build the op.
        gate/up scaled by s=2^k through their E8M0 bytes (exact), up rows biased by -1,
        alpha=2^-k, limit=10*2^k; the router weights take the 2^-2k."""
        from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOpMXFP4

        k = SWIGLU_K
        s13 = self.w13_weight_scale.data
        assert int(s13.max()) + k <= 254, "E8M0 scale headroom exceeded"
        s13.add_(k)
        dev = self.w13_weight.device
        b13 = torch.zeros(2 * self.I, dtype=torch.bfloat16, device=dev)
        b13[1::2] = -1.0
        b2 = torch.zeros(self.D, dtype=torch.bfloat16, device=dev)
        op = VllmMixtureOfExpertsOpMXFP4(
            self.E,
            self.n_local,
            self.e0,
            self.e0 + self.n_local - 1,
            block_size=32,
            has_bias=True,
            alpha=2.0**-k,
            limit=self.limit * 2.0**k,
        )
        # Every expert needs its OWN bias tensors: the Synapse MoE multiplexer registers
        # per-expert inputs, and one shared tensor object fails the graph compile with
        # "MOE multiplexer weights were partially registered".
        self._b13 = b13.unsqueeze(0).repeat(self.n_local, 1).contiguous()
        self._b2 = b2.unsqueeze(0).repeat(self.n_local, 1).contiguous()
        for e in range(self.n_local):
            op.w13_list[e].set_weight(self.w13_weight.data[e])
            op.w13_list[e].set_scale(s13[e])
            op.w13_list[e].set_bias(self._b13[e])
            op.w2_list[e].set_weight(self.w2_weight.data[e])
            op.w2_list[e].set_scale(self.w2_weight_scale.data[e])
            op.w2_list[e].set_bias(self._b2[e])
        op._cache_weight_lists()
        self.moe_op = op
        self._wscale = 2.0 ** (-2 * k)
        logger.warning(
            "dsv41: MoE experts %d..%d on bias_mxfp4_fused_weights, clamp emulation k=%d",
            self.e0,
            self.e0 + self.n_local - 1,
            k,
        )
        if mxk.ENABLED:
            self._mxk_setup(s13, k)

    def _mxk_setup(self, s13: torch.Tensor, k: int) -> None:
        """DSV41_MXK=1: serve this layer's calls of <= mxk.MAX_TOKENS tokens with the custom kernels, on the same
        weight tensors as the stock op (s13 already carries the +k E8M0 shift, so the kernels take alpha 2^-k,
        limit L * 2^k and ascale 2^-2k: the exact clamp)."""
        why = mxk.supported(self.n_local, self.D, self.I, self.topk)
        if why is None and not mxk.env_ok():
            why = (
                "kernel library not registered before the device opened (GC_KERNEL_PATH="
                f"{os.environ.get('GC_KERNEL_PATH')}, PT_HPU_LAZY_ACC_PAR_MODE="
                f"{os.environ.get('PT_HPU_LAZY_ACC_PAR_MODE')}; patches/dsv41-kernel-env not applied?)"
            )
        if why is None:
            mxk.load_ops()
            self._mxk = (
                self.w13_weight.data,
                s13,
                self.w2_weight.data,
                self.w2_weight_scale.data,
                (2.0**-k, self.limit * 2.0**k, 0.0, 2.0 ** (-2 * k)),
            )
        if why is not None or not getattr(DSV41MoEHPU, "_mxk_logged", False):
            DSV41MoEHPU._mxk_logged = True
            logger.warning(
                "dsv41: layer %d MXFP4 decode kernels %s (%s; n=%d D=%d I=%d top-%d, <= %d tokens, "
                "experts %d..%d, GC_KERNEL_PATH=%s, recipe cache %s)",
                self.layer_id,
                "ON" if why is None else "OFF, stock op kept: " + why,
                mxk.lib_version(),
                self.n_local,
                self.D,
                self.I,
                self.topk,
                mxk.MAX_TOKENS,
                self.e0,
                self.e0 + self.n_local - 1,
                os.environ.get("GC_KERNEL_PATH"),
                os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG"),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.D)
        w, ids = route(flat, self.gate_w, self.gate_bias, self.topk, self.route_scale)
        T = flat.shape[0]
        if self._mxk is not None and T <= mxk.MAX_TOKENS:
            # DSV41_MXK=1: straight from the packed weights; routes to experts on other ranks add nothing, and
            # the routing weights go in unscaled (the 2^-2k is the kernel's ascale)
            w13, s13, w2, s2, prm = self._mxk
            idf = ids.reshape(T * self.topk).to(torch.int32)
            ap = torch.ops.custom_op.mxk_w13x(mxk.permute(flat), idf, w13, s13, self.topk, self.e0, *prm, True)
            routed = torch.ops.custom_op.mxk_w2x(
                ap, idf, w.reshape(T * self.topk).to(flat.dtype), w2, s2, self.topk, self.e0
            )
        else:
            routed = self.moe_op(flat, ids.to(torch.int32), (w * self._wscale).to(flat.dtype))
        gu = self.shared_w13(flat)[0]
        g, u = gu.chunk(2, dim=-1)
        shared = self.shared_w2(swiglu_clamped(g, u, self.limit).to(flat.dtype))[0]
        out = routed + shared
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out.reshape(shape)


# ---------------------------------------------------------------------- model
def _linear_factory(prefix: str):
    def make(name, i, o, kind, dtype):
        pd = dtype or torch.bfloat16
        p = f"{prefix}.{name}"
        if kind == "replicated":
            return ReplicatedLinear(i, o, bias=False, params_dtype=pd, prefix=p)
        if kind == "column":
            return ColumnParallelLinear(i, o, bias=False, params_dtype=pd, prefix=p)
        if kind == "row":
            return RowParallelLinear(i, o, bias=False, params_dtype=pd, reduce_results=True, prefix=p)
        raise ValueError(kind)

    return make


class DeepseekV41HPUForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        runner_patch.install()
        cfg = vllm_config.model_config.hf_config
        self.cfg, self.vllm_config = cfg, vllm_config
        self.tp_rank, self.tp_size = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
        self.block_size = vllm_config.cache_config.block_size
        max_len = vllm_config.model_config.max_model_len
        assert cfg.num_attention_heads % self.tp_size == 0 and cfg.o_groups % self.tp_size == 0
        n_cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads
        assert n_cols % self.tp_size == 0, "engram hash columns must split evenly over TP"
        self.cols_per_rank = n_cols // self.tp_size
        self.primes = bucket_primes(
            cfg.engram_layer_ids, cfg.engram_max_ngram_size, cfg.engram_n_heads, cfg.engram_vocab_size
        )

        c0 = self.tp_rank * self.cols_per_rank
        gather = (lambda t: tensor_model_parallel_all_gather(t.contiguous(), dim=-2)) if self.tp_size > 1 else None

        def engram(lid):
            return DSV41Engram(
                cfg,
                lid,
                self.primes[list(cfg.engram_layer_ids).index(lid)],
                c0,
                c0 + self.cols_per_rank,
                _linear_factory(f"model.layers.{lid}"),
                gather,
            )

        def moe(lid):
            return DSV41MoEHPU(cfg, lid, f"model.layers.{lid}.ffn", self.tp_rank, self.tp_size)

        self.stack = DSV41Stack(
            cfg,
            _linear_factory("model"),
            moe,
            lambda: VocabParallelEmbedding(
                cfg.vocab_size, cfg.hidden_size, params_dtype=torch.bfloat16, prefix="model.embed"
            ),
            self.tp_size,
            engram,
            max_pos=max_len,
        )
        try:
            import habana_frameworks.torch as htorch

            if htorch.utils.internal.is_lazy():
                self.stack.mark_step = htorch.core.mark_step
        except ImportError:
            pass
        # the reference head is fp32 and so are its logits
        self.lm_head = ParallelLMHead(cfg.vocab_size, cfg.hidden_size, params_dtype=torch.float32, prefix="lm_head")
        self.logits_processor = LogitsProcessor(cfg.vocab_size)
        # vLLM >= 0.30 defaults head_dtype to the model dtype and then casts the fp32 hidden
        # state AND the fp32 head weight to bf16 every step: bf16 logits (one-ulp ties at
        # |logit| 16-64 flipped greedy tokens vs the prefill path) plus a per-step weight copy.
        self.logits_processor.head_dtype = torch.float32

        # caches: one per (layer, kind), each its own MLAAttentionSpec width
        self.caches = nn.ModuleDict()
        W, IW = cfg.head_dim, cfg.index_head_dim
        for r in self.stack.roles:
            kinds = {"swa": W}
            if r.is_kv_source:
                kinds.update(comp=W, idx_k=IW)
                if r.ratio > 1:
                    kinds["pend"] = 2 * W * 2  # fp32 (kv, score) pair as bf16 elements
            for kind, width in kinds.items():
                name = f"model.layers.{r.layer_id}.attn.cache_{kind}"
                ratio = r.ratio if kind in ("comp", "idx_k") else 1  # entry caches: one row per r tokens
                self.caches[f"{r.layer_id}_{kind}"] = DSV41KVCache(
                    name, width, torch.bfloat16, self.block_size, vllm_config, ratio
                )

        tok = _load_tokenizer(vllm_config)
        token_map, size = build_compressed_token_map(tok)
        assert size == cfg.engram_compressed_vocab_size, (size, cfg.engram_compressed_vocab_size)
        hasher = EngramHasher(
            cfg.engram_layer_ids,
            cfg.engram_max_ngram_size,
            cfg.engram_n_heads,
            cfg.engram_vocab_size,
            token_map,
            size,
            cfg.engram_pad_token_id,
        )
        runner_patch.configure(hasher, max_len, self.block_size)
        logger.warning(
            "dsv41: model built; TP %d, %d paged caches, max_model_len %d", self.tp_size, len(self.caches), max_len
        )

    # --------------------------------------------------------------- forward
    def _cache_views(self, device):
        """Views over the bound caches, rebuilt only when a cache tensor changes (the
        profiling run uses 1-block dummies; KV binding replaces them once)."""
        key = tuple(id(c.kv_cache) for c in self.caches.values())
        if getattr(self, "_views_key", None) != key or any(c.kv_cache is None for c in self.caches.values()):
            views = {}
            for k, c in self.caches.items():
                lid, kind = k.split("_", 1)
                views.setdefault(int(lid), {})[kind] = c.view(device)
            self._views = views
            self._views_key = tuple(id(c.kv_cache) for c in self.caches.values())
            self._graph_steps = None
        return self._views

    def _decode_steps(self, views):
        if self._graph_steps is None:
            import habana_frameworks.torch as htorch

            raw = [BlockStep(layer, views, self.block_size) for layer in self.stack.layers]
            if LAYER_GROUP > 1:
                src_ids = [r.layer_id for r in self.stack.roles if r.is_index_source]
                groups = [
                    GroupStep(raw[i : i + LAYER_GROUP], src_ids, self.cfg.engram_layer_ids)
                    for i in range(0, len(raw), LAYER_GROUP)
                ]
                self._graph_groups = [
                    htorch.hpu.wrap_in_hpu_graph(g, disable_tensor_cache=True, dry_run=False) for g in groups
                ]
                logger.warning("dsv41: decode HPU graphs armed: %d groups of %d layers", len(groups), LAYER_GROUP)
            steps = [htorch.hpu.wrap_in_hpu_graph(st, disable_tensor_cache=True, dry_run=False) for st in raw]
            self._graph_steps = steps
            if LAYER_GROUP <= 1:
                logger.warning("dsv41: per-layer HPU graphs armed for decode (%d layers)", len(steps))
        return self._graph_steps

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.stack.embed(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.stack.embed(input_ids)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, intermediate_tensors=None, inputs_embeds=None, **kwargs
    ) -> torch.Tensor:
        md = get_forward_context().attn_metadata
        if TAG_MODE == "count":
            self._forwards = getattr(self, "_forwards", 0) + 1
            if self._forwards % TAG_LOG_EVERY == 0:  # syncs; the isolation gate greps these lines
                v = tag_violations(by_label=True)
                logger.warning(
                    "dsv41: tag check after %d forwards: %d violating rows of %d reads%s",
                    self._forwards,
                    v["rows"],
                    v["reads"],
                    f" {v['by_label']}" if v["by_label"] else "",
                )
        if input_ids.dim() == 1:  # a runner path that flattened anyway: one request row
            input_ids, positions = input_ids.view(1, -1), positions.view(1, -1)
        ctx = runner_patch.batch_ctx_from_metadata(md, positions, self.block_size)
        rows = md.dsv41_engram_rows
        engram_rows = {lid: rows[:, :, gi].long() for gi, lid in enumerate(self.cfg.engram_layer_ids)}
        views = self._cache_views(input_ids.device)
        if LAYER_GRAPHS and input_ids.shape[1] == 1:
            from .mhc import hc_pre, identity_pre
            from .numerics import rms_norm

            steps = self._decode_steps(views)
            e = self.stack.embed(input_ids)
            h = e.unsqueeze(-2).repeat(1, 1, self.cfg.hc_mult, 1)
            if LAYER_GROUP > 1:
                h, pre = run_groups(self._graph_groups, h, identity_pre(h, self.cfg.hc_mult), ctx, engram_rows)
            else:
                h, pre = run_steps(steps, h, identity_pre(h, self.cfg.hc_mult), ctx, engram_rows)
            return rms_norm(hc_pre(h, pre), self.stack.norm, self.cfg.rms_norm_eps)
        keep = getattr(md, "dsv41_replay", None) or 0  # DSV41_REPLAY_KEEP: decoder replay (replay.py)
        if keep and input_ids.shape[1] > 1:
            if getattr(self, "_replay", None) is None:
                self._replay = replay.Replay(self.stack)
            return replay.run(self.stack, self._replay, input_ids, ctx, views, engram_rows, keep)
        return self.stack(input_ids, ctx, views, engram_rows)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states.float())

    # ----------------------------------------------------------------- load
    def load_weights(self, weights) -> set[str]:
        """Ignores the loader's iterator (it would read all 518 GB on every rank) and reads the converted checkpoint
        with ranged slices instead."""
        from .loader import load_dsv41

        load_dsv41(self, self.vllm_config.model_config.model)
        dev = next(self.parameters()).device
        for layer in self.stack.layers:
            layer.attn.rope.materialize(dev)
            layer.ffn.finalize()
        return {n for n, _ in self.named_parameters()}


def _load_tokenizer(vllm_config):
    from transformers import AutoTokenizer

    mc = vllm_config.model_config
    return AutoTokenizer.from_pretrained(mc.tokenizer or mc.model, trust_remote_code=False)


__all__ = ["DeepseekV41HPUForCausalLM", "DSV41KVCache", "DSV41MoEHPU", "csa_roles"]
