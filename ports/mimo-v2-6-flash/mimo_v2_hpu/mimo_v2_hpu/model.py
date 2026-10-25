# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 (MiMoV2ForCausalLM: MiMo-V2.6-Flash and MiMo-V2.6-Pro) on Gaudi, vllm-gaudi v0.26.0. Text path only.

What differs from upstream vLLM's mimo_v2.py, and why:

- Attention heads. Q/K heads are 192 wide and V heads 128. HPUAttentionImpl takes one head size, so V is
  zero-padded to 192 before attention and the extra 64 output columns are dropped after it. Exact: a zero V column
  adds nothing. It costs a third more V cache.
- Sliding-window layers (window 128, learned sink per head) use attention.MiMoSWAHPUAttentionImpl: banded fp32
  prefill with the sink, and decode over exactly the last 128 positions with the sink counted once per sequence.
- Global layers optionally prefill in query tiles (MIMO_GA_TILE=1, attention.MiMoGATiledHPUAttentionImpl).
- RoPE tables are rebuilt on the CPU in float64 (ops.exact_cos_sin).
- Routed experts. The checkpoint stores them as OCP MXFP4; each rank holds E / tp whole experts and the partial
  sums are all-reduced over the TP group. MIMO_EXPERT_DTYPE picks the device format:
  fp8 (default, the Flash recipe): re-encoded at load to per-output-channel FP8 e4m3 with power-of-two scales
  (ops.mxfp4_to_fp8, exact, checked on every tensor), run by the FP8 MoE op with dynamic per-token activation
  scales. FP8 doubles expert memory, which the Pro checkpoint cannot afford on one node.
  mxfp4 (the Pro recipe): the checkpoint bytes as they are, run by the stock MXFP4 MoE op, and with MIMO_MXK=1
  decode-sized calls by the MXFP4 TPC kernels (mxk.py).
- Routing: fp32 gate logits, sigmoid, e_score_correction_bias for the choice only, top-k, renormalised (the
  reference model's math for n_group == 1).
- FP8 block weights (fused qkv_proj, the dense MLP of layer 0) are dequantised to bf16 at load, so the checkpoint
  is served with its quantization_config removed (convert/stage_checkpoint.py).
- The vision tower, audio encoder and MTP layers are skipped at load.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from . import attention, mxk
from .ops import (
    MXFP4_BLOCK,
    dequant_fp8_block,
    dequant_qkv,
    exact_cos_sin,
    mxfp4_expert_slot,
    mxfp4_to_fp8,
    qkv_rank_rows,
)

logger = init_logger("vllm.mimo_v2_hpu")

SKIP_PREFIXES = ("visual.", "audio_encoder.", "speech_embeddings.", "model.mtp.", "mtp.")
EXPERT_DTYPES = ("fp8", "mxfp4")


def expert_dtype() -> str:
    """MIMO_EXPERT_DTYPE: the device format of the routed experts, fp8 (default) or mxfp4."""
    v = os.environ.get("MIMO_EXPERT_DTYPE", "fp8").lower()
    if v not in EXPERT_DTYPES:
        raise ValueError(f"mimo_v2_hpu: MIMO_EXPERT_DTYPE={v!r}, expected one of {EXPERT_DTYPES}")
    return v


def _exact_rope_table(rope, theta: float) -> None:
    """Replace a rotary embedding's cos/sin table with ops.exact_cos_sin. get_rope shares instances, so once each."""
    if getattr(rope, "_mimo_exact", False):
        return
    cache = rope.cos_sin_cache
    exact = exact_cos_sin(cache.shape[0], rope.rotary_dim, theta)
    rope.cos_sin_cache = exact.to(dtype=cache.dtype).to(cache.device)
    rope._mimo_exact = True


def _swa_kv_layer_views(attn_layers):
    """Per-layer attention metadata when window-sized KV groups are on, else None.

    With MIMO_SWA_KV_GROUPS=1 the sliding layers live in their own KV cache groups, so their block ids come from
    their group's block table (a global-table id addresses another layer's K/V in the shared tensor). The runner
    adds, per group id, the group's slot mapping and window block list (None for the global group). Each layer gets
    a copy of the metadata with its group's tensors in slot_mapping and window_block_list, same shapes as the base
    ones, so the compiled decoder layers see exactly the structure they see without groups. This loop runs eagerly
    between the regionally compiled layers, so it adds no graph variants.
    """
    if not is_forward_context_available():
        return None
    md = get_forward_context().attn_metadata
    sm = getattr(md, "kv_group_slot_mapping", None)
    if sm is None:
        return None
    wl = md.kv_group_window_block_list
    clear = {"kv_group_slot_mapping": None, "kv_group_window_block_list": None}
    per_gid = [
        md._replace(**clear)
        if s is None
        else md._replace(slot_mapping=s, window_block_list=None if wl is None else wl[g], **clear)
        for g, s in enumerate(sm)
    ]
    return [per_gid[a._mimo_kv_gid] for a in attn_layers]


class MiMoV2HpuAttention(nn.Module):
    def __init__(self, config, layer_id: int, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.is_swa = config.hybrid_layer_pattern[layer_id] == 1
        if self.is_swa:
            heads, kvh = config.swa_num_attention_heads, config.swa_num_key_value_heads
            hd = config.swa_head_dim
            vhd = getattr(config, "swa_v_head_dim", hd)
            theta = getattr(config, "swa_rope_theta", config.rope_theta)
            window = config.sliding_window_size
            has_sink = getattr(config, "add_swa_attention_sink_bias", False)
        else:
            heads, kvh = config.num_attention_heads, config.num_key_value_heads
            hd = config.head_dim
            vhd = getattr(config, "v_head_dim", hd)
            theta = config.rope_theta
            window = None
            has_sink = getattr(config, "add_full_attention_sink_bias", False)
        if has_sink and not self.is_swa:
            raise NotImplementedError("mimo_v2_hpu: sinks on global-attention layers are not implemented")
        if self.is_swa and not has_sink:
            raise NotImplementedError("mimo_v2_hpu: sliding-window layers without sinks are not implemented")

        tp = get_tensor_model_parallel_world_size()
        self.total_num_heads, self.total_num_kv_heads = heads, kvh
        self.head_dim, self.v_head_dim = hd, vhd
        self.num_heads = heads // tp
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            hd,
            heads,
            kvh,
            bias=config.attention_bias,
            quant_config=None,
            prefix=f"{prefix}.qkv_proj",
            v_head_size=vhd,
        )
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.q_size = self.num_heads * hd
        self.k_size = self.num_kv_heads * hd
        self.v_size = self.num_kv_heads * vhd
        self.o_proj = RowParallelLinear(
            heads * vhd, config.hidden_size, bias=False, quant_config=None, prefix=f"{prefix}.o_proj"
        )
        self.v_scale = getattr(config, "attention_value_scale", None)

        max_pos = min(config.max_position_embeddings, vllm_config.model_config.max_model_len)
        self.rotary_emb = get_rope(
            head_size=hd,
            max_position=max_pos,
            is_neox_style=True,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": theta,
                "partial_rotary_factor": getattr(config, "partial_rotary_factor", 1.0),
            },
        )
        _exact_rope_table(self.rotary_emb, theta)
        self.attention_sink_bias = nn.Parameter(torch.empty(self.num_heads), requires_grad=False) if has_sink else None
        self.attn = Attention(
            self.num_heads,
            hd,
            hd**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=None,
            per_layer_sliding_window=window,
            prefix=f"{prefix}.attn",
            sinks=self.attention_sink_bias,
        )
        impl = self.attn.impl
        if self.is_swa or attention.GA_TILE:
            if type(impl).__name__ != "HPUAttentionImpl":
                raise RuntimeError(f"mimo_v2_hpu needs HPUAttentionImpl, got {type(impl).__name__}")
            impl.__class__ = attention.swa_impl_class() if self.is_swa else attention.ga_impl_class()

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        if self.v_scale is not None:
            v = v * self.v_scale
        pad = self.head_dim - self.v_head_dim
        if pad:
            v = F.pad(v.unflatten(-1, (self.num_kv_heads, self.v_head_dim)), (0, pad)).flatten(-2)
        o = self.attn(q, k, v)
        if pad:
            o = o.unflatten(-1, (self.num_heads, self.head_dim))[..., : self.v_head_dim].flatten(-2)
        out, _ = self.o_proj(o)
        return out


class MiMoV2HpuMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, prefix: str):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False, quant_config=None, prefix=f"{prefix}.gate_up_proj"
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False, quant_config=None, prefix=f"{prefix}.down_proj"
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(x))
        return x


class MiMoV2HpuMoE(nn.Module):
    """Routed experts, E / tp whole experts per rank (expert parallel over the TP group), no shared experts.

    fp8: FP8 doubles expert memory against MXFP4 but skips the per-expert FP4 unpack; an 8-card node holds Flash
    with room for KV. mxfp4: the checkpoint bytes on the stock MXFP4 op, decode calls on the kernels with MIMO_MXK=1.
    """

    def __init__(self, config, prefix: str):
        super().__init__()
        if getattr(config, "n_group", 1) not in (1, None) or getattr(config, "topk_group", 1) not in (1, None):
            raise NotImplementedError("mimo_v2_hpu: grouped routing (n_group > 1) is not implemented")
        if getattr(config, "n_shared_experts", None):
            raise NotImplementedError("mimo_v2_hpu: shared experts are not implemented")
        if config.scoring_func != "sigmoid" or config.hidden_act != "silu":
            raise NotImplementedError(f"mimo_v2_hpu: {config.scoring_func=} {config.hidden_act=}")
        tp = get_tensor_model_parallel_world_size()
        n_exp, hidden, inter = config.n_routed_experts, config.hidden_size, config.moe_intermediate_size
        if n_exp % tp:
            raise ValueError(f"{n_exp} experts are not divisible by tp {tp}")
        self.n_experts, self.top_k = n_exp, config.num_experts_per_tok
        self.n_local = n_exp // tp
        self.e_start = get_tensor_model_parallel_rank() * self.n_local
        self.inter, self.hidden = inter, hidden
        self.norm_topk = config.norm_topk_prob
        self.scaling = config.routed_scaling_factor if config.routed_scaling_factor is not None else 1.0
        self.prefix = prefix

        gate_dtype = getattr(torch, getattr(config, "moe_router_dtype", "float32") or "float32")
        self.gate_weight = nn.Parameter(torch.empty(n_exp, hidden, dtype=gate_dtype), requires_grad=False)
        self.e_score_correction_bias = nn.Parameter(torch.empty(n_exp, dtype=torch.float32), requires_grad=False)
        self.expert_dtype = expert_dtype()
        n, f32, u8 = self.n_local, torch.float32, torch.uint8
        if self.expert_dtype == "fp8":
            f8 = torch.float8_e4m3fn
            self.w13_weight = nn.Parameter(torch.empty(n, 2 * inter, hidden, dtype=f8), requires_grad=False)
            self.w13_weight_scale = nn.Parameter(torch.empty(n, 2 * inter, dtype=f32), requires_grad=False)
            self.w2_weight = nn.Parameter(torch.empty(n, hidden, inter, dtype=f8), requires_grad=False)
            self.w2_weight_scale = nn.Parameter(torch.empty(n, hidden, dtype=f32), requires_grad=False)
        else:
            if hidden % MXFP4_BLOCK or inter % MXFP4_BLOCK:
                raise ValueError(f"mimo_v2_hpu: MXFP4 experts need hidden {hidden} and intermediate {inter} % 32 == 0")
            b = MXFP4_BLOCK
            self.w13_weight = nn.Parameter(torch.empty(n, 2 * inter, hidden // 2, dtype=u8), requires_grad=False)
            self.w13_weight_scale = nn.Parameter(torch.empty(n, 2 * inter, hidden // b, dtype=u8), requires_grad=False)
            self.w2_weight = nn.Parameter(torch.empty(n, hidden, inter // 2, dtype=u8), requires_grad=False)
            self.w2_weight_scale = nn.Parameter(torch.empty(n, hidden, inter // b, dtype=u8), requires_grad=False)
        self._pending: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
        self.moe_op = None
        self.mxk = False

    def load_expert(self, expert_id: int, proj: str, kind: str, w: torch.Tensor) -> str | None:
        """Take one checkpoint expert tensor (packed weight or E8M0 scale). fp8: once both halves of a projection are
        in, re-encode it into its slot and return the parameter name; None while waiting or for another rank's
        expert. mxfp4: copy the bytes into their slot and return the parameter name."""
        e = expert_id - self.e_start
        if not 0 <= e < self.n_local:
            return None
        if self.expert_dtype == "mxfp4":
            return self._copy_mxfp4(e, proj, kind, w)
        entry = self._pending.setdefault((e, proj), {})
        entry[kind] = w
        if "weight" not in entry or "weight_scale" not in entry:
            return None
        del self._pending[(e, proj)]
        dev = self.w13_weight.device
        q, scale = mxfp4_to_fp8(entry["weight"].to(dev), entry["weight_scale"].to(dev))
        if proj == "down_proj":
            self.w2_weight.data[e].copy_(q)
            self.w2_weight_scale.data[e].copy_(scale)
            return "w2_weight"
        rows = slice(0, self.inter) if proj == "gate_proj" else slice(self.inter, 2 * self.inter)
        self.w13_weight.data[e, rows].copy_(q)
        self.w13_weight_scale.data[e, rows].copy_(scale)
        return "w13_weight"

    def _copy_mxfp4(self, e: int, proj: str, kind: str, w: torch.Tensor) -> str:
        name, rows = mxfp4_expert_slot(proj, kind, self.inter)
        dst = getattr(self, name).data[e]
        if rows is not None:
            dst = dst[rows]
        if tuple(w.shape) != tuple(dst.shape) or w.dtype != torch.uint8:
            raise ValueError(
                f"mimo_v2_hpu {self.prefix}: expert {e + self.e_start} {proj}.{kind} is {w.dtype} {tuple(w.shape)}, "
                f"expected uint8 {tuple(dst.shape)}"
            )
        dst.copy_(w)
        return name

    def build_op(self):
        if self.expert_dtype == "mxfp4":
            self._build_mxfp4_op()
            return
        if self._pending:
            raise RuntimeError(f"mimo_v2_hpu {self.prefix}: incomplete experts {sorted(self._pending)[:4]}")
        from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOpFP8PerChannel

        op = VllmMixtureOfExpertsOpFP8PerChannel(
            self.n_experts, self.n_local, self.e_start, self.e_start + self.n_local - 1
        )
        for e in range(self.n_local):
            op.w13_list[e].set_weight(self.w13_weight.data[e])
            op.w13_list[e].set_scale_inv_fp8(self.w13_weight_scale.data[e])
            op.w2_list[e].set_weight(self.w2_weight.data[e])
            op.w2_list[e].set_scale_inv_fp8(self.w2_weight_scale.data[e])
        op._cache_weight_lists()
        self.moe_op = op

    def _build_mxfp4_op(self):
        from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOpMXFP4

        op = VllmMixtureOfExpertsOpMXFP4(
            self.n_experts,
            self.n_local,
            self.e_start,
            self.e_start + self.n_local - 1,
            block_size=MXFP4_BLOCK,
            has_bias=False,
        )
        for e in range(self.n_local):
            op.w13_list[e].set_weight(self.w13_weight.data[e])
            op.w13_list[e].set_scale(self.w13_weight_scale.data[e])
            op.w2_list[e].set_weight(self.w2_weight.data[e])
            op.w2_list[e].set_scale(self.w2_weight_scale.data[e])
        op._cache_weight_lists()
        self.moe_op = op
        if not mxk.ENABLED:
            return
        why = mxk.layer_reason(self.n_local, self.hidden, self.inter, self.expert_dtype)
        if why is not None:
            mxk.warn_once("layer", "%s not used for %s (%s): stock MoE op", mxk.TAG, self.prefix, why)
            return
        self.mxk = mxk.load()
        if self.mxk:
            mxk.register_eager(
                self.w13_weight,
                self.w13_weight_scale,
                self.w2_weight,
                self.w2_weight_scale,
                self.top_k,
                self.e_start,
                self.hidden,
                self.n_local,
                self.inter,
            )

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.gate_weight.float()).sigmoid()
        choice = scores + self.e_score_correction_bias.unsqueeze(0)
        topk_idx = torch.topk(choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_w = scores.gather(1, topk_idx)
        if self.top_k > 1 and self.norm_topk:
            topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx, topk_w * self.scaling

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        topk_idx, topk_w = self.route(x)
        if self.expert_dtype == "mxfp4":
            # Static shapes under torch.compile: the token count is a constant of each graph, so this picks one path
            # per bucket.
            if self.mxk and mxk.covers(x.shape[0], self.top_k) and x.dtype == torch.bfloat16:
                w13, s13, w2, s2 = self.w13_weight, self.w13_weight_scale, self.w2_weight, self.w2_weight_scale
                out = mxk.moe(x, topk_idx, topk_w, w13, s13, w2, s2, self.top_k, self.e_start)
            else:
                out = self.moe_op(
                    x, topk_idx.to(torch.int32), topk_w.to(x.dtype), permuted_weights=True, activation="silu"
                )
            return tensor_model_parallel_all_reduce(out).view(shape)
        from vllm_gaudi.extension.ops import dynamic_quant

        op = self.moe_op
        x_fp8, x_scale = dynamic_quant(x)
        # The explicit overload: the op packet otherwise resolves overloads by trial, and each miss formats
        # repr(x_scale), a device sync per MoE layer.
        out = torch.ops.hpu.mixture_of_experts.fp8_fused_weights_dynamic(
            hidden_states=x_fp8,
            expert_routing_table=topk_idx.to(torch.int64),
            router_weights=topk_w.to(x.dtype),
            w12=op._cached_w13_views,
            w3=op._cached_w2_views,
            d_scale_hidden_states=x_scale,
            d_scale_w12=op._cached_w13_scale_views,
            d_scale_w3=op._cached_w2_scale_views,
            permuted_weights=True,
            activation="silu",
            experts_min=op.experts_min,
            experts_max=op.experts_max,
            **op._get_extra_kwargs(x.shape[0]),
        )
        return tensor_model_parallel_all_reduce(out).view(shape)


class MiMoV2HpuDecoderLayer(nn.Module):
    def __init__(self, config, layer_id: int, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.self_attn = MiMoV2HpuAttention(config, layer_id, vllm_config, prefix=f"{prefix}.self_attn")
        freq = getattr(config, "moe_layer_freq", None)
        is_moe = freq is not None and not isinstance(freq, int) and bool(freq[layer_id])
        self.mlp = (
            MiMoV2HpuMoE(config, prefix=f"{prefix}.mlp")
            if is_moe
            else MiMoV2HpuMLP(config.hidden_size, config.intermediate_size, prefix=f"{prefix}.mlp")
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class MiMoV2HpuModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config.get_text_config()
        window = getattr(config, "sliding_window_size", None)
        block = vllm_config.cache_config.block_size
        if window and window % block:
            # The exact decode window needs whole blocks (fixes/swa-decode-window).
            raise ValueError(f"mimo_v2_hpu: sliding window {window} is not a multiple of --block-size {block}")
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        self.layers = nn.ModuleList(
            [
                MiMoV2HpuDecoderLayer(config, i, vllm_config, prefix=f"{prefix}.layers.{i}")
                for i in range(config.num_hidden_layers)
            ]
        )
        # A plain list, not submodules: each layer's Attention, tagged with its KV cache group id by
        # swa_kv.SwaKvGroups.after_bind. Kept here because regional compilation wraps the entries of self.layers.
        self._kv_attn = [layer.self_attn.attn for layer in self.layers]
        self.norm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        residual = None
        views = _swa_kv_layer_views(self._kv_attn)
        if views is None:
            for layer in self.layers:
                hidden_states, residual = layer(positions, hidden_states, residual)
        else:
            fc = get_forward_context()
            base = fc.attn_metadata
            try:
                for layer, view in zip(self.layers, views, strict=True):
                    fc.attn_metadata = view
                    hidden_states, residual = layer(positions, hidden_states, residual)
            finally:
                fc.attn_metadata = base
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MiMoV2HpuForCausalLM(nn.Module):
    """Registered over vLLM's MiMoV2ForCausalLM, MiMoV2FlashForCausalLM and MiMoV2OmniForCausalLM (__init__.py)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config.get_text_config()
        self.config = config
        self.model = MiMoV2HpuModel(vllm_config=vllm_config, prefix="model")
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix="lm_head")
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self, input_ids, positions, intermediate_tensors: IntermediateTensors | None = None, inputs_embeds=None, **kw
    ):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    # ------------------------------------------------------------------------------------------------ loading
    def _load_dense(self, name: str, w: torch.Tensor, params: dict, loaded: set) -> None:
        tp_rank, tp_size = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
        if name.endswith("self_attn.qkv_proj.weight"):
            attn = self.get_submodule(name.rsplit(".", 2)[0])
            idx = qkv_rank_rows(
                attn.total_num_heads,
                attn.total_num_kv_heads,
                attn.head_dim,
                attn.v_head_dim,
                self.config.num_key_value_heads,
                tp_rank,
                tp_size,
            )
            params[name].data.copy_(w[idx].to(params[name].dtype))
            loaded.add(name)
            return
        if name.endswith("attention_sink_bias"):
            param = params[name]
            n = param.shape[0]
            param.data.copy_(w.narrow(0, tp_rank * n, n).to(param.dtype))
            loaded.add(name)
            return
        for ckpt, ours in ((".mlp.gate.weight", ".mlp.gate_weight"), (".mlp.gate.e_score_correction_bias", None)):
            if ckpt in name:
                pname = name.replace(ckpt, ours or ".mlp.e_score_correction_bias")
                params[pname].data.copy_(w.to(params[pname].dtype))
                loaded.add(pname)
                return
        for shard, ckpt in ((0, ".mlp.gate_proj."), (1, ".mlp.up_proj.")):
            if ckpt in name:
                pname = name.replace(ckpt, ".mlp.gate_up_proj.")
                params[pname].weight_loader(params[pname], w, shard)
                loaded.add(pname)
                return
        if name not in params:
            raise KeyError(f"mimo_v2_hpu: unexpected checkpoint tensor {name}")
        param = params[name]
        getattr(param, "weight_loader", default_weight_loader)(param, w)
        loaded.add(name)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()
        pending: dict[str, dict[str, torch.Tensor]] = {}
        n_expert_tensors = 0
        for name, w in weights:
            if name.startswith(SKIP_PREFIXES) or "rotary_emb.inv_freq" in name:
                continue
            if ".mlp.experts." in name:
                # model.layers.L.mlp.experts.E.{gate,up,down}_proj.{weight,weight_scale}
                head, tail = name.split(".mlp.experts.", 1)
                eid, proj, kind = tail.split(".")
                moe = self.get_submodule(f"{head}.mlp")
                got = moe.load_expert(int(eid), proj, kind, w)
                if got is not None and moe.expert_dtype == "fp8":  # one re-encoded projection, weight and scale
                    loaded.update((f"{head}.mlp.{got}", f"{head}.mlp.{got}_scale"))
                    n_expert_tensors += 1
                elif got is not None:  # mxfp4: one tensor copied, weight or scale
                    loaded.add(f"{head}.mlp.{got}")
                    n_expert_tensors += kind == "weight"
                continue
            if name.endswith(".weight_scale_inv") or (name.endswith(".weight") and w.dtype == torch.float8_e4m3fn):
                base, kind = name.rsplit(".", 1)
                entry = pending.setdefault(base, {})
                entry[kind] = w
                if "weight" in entry and "weight_scale_inv" in entry:
                    del pending[base]
                    wt, s = entry["weight"], entry["weight_scale_inv"]
                    if base.endswith("self_attn.qkv_proj"):
                        deq = dequant_qkv(wt, s, self.config.num_key_value_heads)
                    else:
                        deq = dequant_fp8_block(wt, s)
                    self._load_dense(f"{base}.weight", deq, params, loaded)
                continue
            self._load_dense(name, w, params, loaded)

        if pending:
            raise RuntimeError(f"mimo_v2_hpu: unpaired FP8 tensors: {sorted(pending)[:8]}")
        for m in self.modules():
            if isinstance(m, MiMoV2HpuMoE):
                m.build_op()
        logger.info("mimo_v2_hpu: loaded %d params, %d local expert projections", len(loaded), n_expert_tensors)
        return loaded
