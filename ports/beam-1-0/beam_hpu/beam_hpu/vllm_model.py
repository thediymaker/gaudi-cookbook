# SPDX-License-Identifier: Apache-2.0
"""Beam-1.0 (compressed-tensors FP8) as a vLLM model on Gaudi 2 (vllm-gaudi releases/v0.30.0, lazy mode).

Registered for architecture BeamForCausalLM by beam_hpu.register(). All tensors 2-D [T, ...], residual
stream bf16:

  h = E[id]                                                    bf16
  per layer l (L = l + 1):
    a = rms(h) * w_in                         -> attention (attention.py) -> y [T, H] bf16
    h = round(round(rms(y) * w_post_attn / sqrt(2L)) + round(h * sqrt((2L-1)/2L)))
    f = rms(h) * w_pre_ffn                    -> dense SwiGLU (l < 3) or MoE (moe.py) -> m
    h = round(round(rms(m) * w_post_ffn / sqrt(2L+1)) + round(h * sqrt(2L/(2L+1))))
  x = rms(h) * w_final                                         bf16
  logits = 30 * tanh(lm_head(x) / 30)                          bf16 head, fp32 softcap

Each norm computes in fp32 and rounds once (numerics.py). Weights go through the stock compressed-tensors
HPU methods (FP8 per-channel weights, dynamic per-token activations); router weight and correction bias
stay fp32.

The model returns h with the input's leading shape ([B, S, H] for vllm-gaudi's padded prompt batches):
the runner flattens it for sampling but indexes it by batch row for prompt logprobs.

Prefill run-ahead bound, BEAM_PREFILL_INFLIGHT=K (default 1; 0 = off): in a forward of at least 256
tokens the host records an HPU event after every layer and waits for the event K layers back before
enqueuing more, so in-flight lazy graphs cannot pile up activations past the card's free HBM during a
long prefill. Decode is untouched.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from . import numerics as N
from .attention import BeamHPUAttention, _Weight, install_fsdpa_query_tiling, make_rope
from .moe import BeamHPUMLP, BeamHPUSparseMoeBlock

logger = logging.getLogger("vllm.beam_hpu")  # under "vllm", so its handlers print

PREFILL_INFLIGHT = int(os.environ.get("BEAM_PREFILL_INFLIGHT", "1"))
PREFILL_INFLIGHT_MIN_T = 256


class BeamHPUDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, layer_id: int, rope, prefix: str):
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        H = cfg.hidden_size
        self.layer_id = layer_id
        self.eps = cfg.rms_norm_eps
        self.input_layernorm = _Weight(H)
        self.self_attn = BeamHPUAttention(
            cfg, layer_id, rope, vllm_config.cache_config, vllm_config.quant_config, prefix=f"{prefix}.self_attn"
        )
        self.post_attention_layernorm = _Weight(H) if cfg.enable_sandwich_norm else None
        self.pre_feedforward_layernorm = _Weight(H)
        self.is_moe = cfg.mlp_layer_types[layer_id] == "sparse"
        if self.is_moe:
            self.mlp = BeamHPUSparseMoeBlock(vllm_config, prefix=f"{prefix}.mlp")
        else:
            self.mlp = BeamHPUMLP(
                H, cfg.intermediate_size, cfg.hidden_act, vllm_config.quant_config, prefix=f"{prefix}.mlp"
            )
        self.post_feedforward_layernorm = _Weight(H) if cfg.enable_sandwich_norm else None
        self.attn_coeffs = N.depth_coeffs(layer_id, True) if cfg.self_attn_depth_scaled_residual else None
        self.mlp_coeffs = N.depth_coeffs(layer_id, False) if cfg.mlp_depth_scaled_residual else None

    @staticmethod
    def _w(m):
        return None if m is None else m.weight

    def forward(self, h: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """h [T, H] bf16 residual stream, pos [T] -> h."""
        dt = h.dtype
        a = N.rms_affine(h, self.input_layernorm.weight, self.eps, dt)
        y = self.self_attn(a, pos)
        h = N.boundary(y, h, self._w(self.post_attention_layernorm), self.eps, self.attn_coeffs, dt)
        f = N.rms_affine(h, self.pre_feedforward_layernorm.weight, self.eps, dt)
        m = self.mlp(f)
        return N.boundary(m, h, self._w(self.post_feedforward_layernorm), self.eps, self.mlp_coeffs, dt)


class BeamHPUModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        self.cfg = cfg
        self.eps = cfg.rms_norm_eps
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size, prefix=f"{prefix}.embed_tokens")
        # the ONE rotary module, registered before the layers so the runner's search finds it
        self.rotary_emb = make_rope(cfg, vllm_config.model_config.max_model_len)
        self.layers = nn.ModuleList(
            [
                BeamHPUDecoderLayer(vllm_config, i, self.rotary_emb, prefix=f"{prefix}.layers.{i}")
                for i in range(cfg.num_hidden_layers)
            ]
        )
        self.norm = _Weight(cfg.hidden_size)

    def forward(
        self, ids: torch.Tensor | None, pos: torch.Tensor, inputs_embeds: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.embed_tokens(ids) if inputs_embeds is None else inputs_embeds
        bound = PREFILL_INFLIGHT if (PREFILL_INFLIGHT > 0 and h.shape[0] >= PREFILL_INFLIGHT_MIN_T) else 0
        events = []
        for layer in self.layers:
            h = layer(h, pos)
            if bound:
                import habana_frameworks.torch as htorch

                ev = htorch.hpu.Event()
                ev.record()
                events.append(ev)
                if len(events) > bound:
                    events.pop(0).synchronize()
        return N.rms_affine(h, self.norm.weight, self.eps, h.dtype)


class BeamHPUForCausalLM(nn.Module):
    """Beam causal LM on HPU (text only)."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        install_fsdpa_query_tiling()
        if os.environ.get("PT_HPU_SDPA_QKV_SLICE_MODE_FWD"):
            raise RuntimeError(
                "beam_hpu: PT_HPU_SDPA_QKV_SLICE_MODE_FWD must be unset (its window semantics are unverified)"
            )
        torch.set_float32_matmul_precision("highest")
        cfg = vllm_config.model_config.hf_config
        if cfg.tie_word_embeddings:
            raise NotImplementedError("beam_hpu: tie_word_embeddings=True is not supported")
        if vllm_config.parallel_config.enable_eplb:
            raise NotImplementedError("beam_hpu: EPLB is not supported")
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError("beam_hpu: pipeline parallelism is not supported")
        # Every layer passes its own per_layer_sliding_window (None on full layers); without this,
        # Attention.__init__ falls back to cache_config.sliding_window and the full layers get a window.
        # The runner reads the window from model_config (hf_config.sliding_window), which stays as configured.
        if vllm_config.cache_config is not None:
            vllm_config.cache_config.sliding_window = None
        self.config = cfg
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config
        self.model = BeamHPUModel(vllm_config, prefix="model")
        self.lm_head = ParallelLMHead(cfg.vocab_size, cfg.hidden_size, prefix="lm_head")
        self.logits_processor = LogitsProcessor(cfg.vocab_size, scale=1.0, soft_cap=None)
        # bf16 head as in the reference (bf16 logits, then the fp32 softcap); never let an fp32
        # override cast the 1.5 GB head to fp32 on every step
        self.logits_processor.head_dtype = None
        self.softcap = cfg.final_logit_softcapping
        self._check_rope()
        logger.warning(
            "beam_hpu: model built; %d layers (%d MoE, %d NoPE), tp %d, window %s, softcap %s, "
            "prefill in-flight layers %s, exact bf16 rounding %s",
            cfg.num_hidden_layers,
            sum(layer.is_moe for layer in self.model.layers),
            len(cfg.nope_layer_ids),
            get_tensor_model_parallel_world_size(),
            cfg.sliding_window,
            self.softcap,
            PREFILL_INFLIGHT if PREFILL_INFLIGHT > 0 else "unbounded",
            N.EXACT_ROUNDING,
        )

    def _check_rope(self) -> None:
        """Exactly one *RotaryEmbedding module (the runner refreshes cos/sin per step only for the
        first one it finds)."""
        n = sum(1 for m in self.modules() if type(m).__name__.endswith("RotaryEmbedding"))
        want = 1 if self.model.rotary_emb is not None else 0
        if n != want:
            raise RuntimeError(f"beam_hpu: {n} rotary modules, expected {want}")

    def embed_input_ids(self, input_ids: torch.Tensor, multimodal_embeddings=None, *, is_multimodal=None):
        return self.model.embed_tokens(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        H = self.config.hidden_size
        lead = tuple(input_ids.shape) if input_ids is not None else tuple(inputs_embeds.shape[:-1])
        ids = input_ids.reshape(-1) if input_ids is not None else None
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.reshape(-1, H)
        h = self.model(ids, positions.reshape(-1), inputs_embeds)
        return h.reshape(*lead, H)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        z = self.logits_processor(self.lm_head, hidden_states)
        if z is None:
            return None
        return N.softcap(z, self.softcap)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded, skipped = load_tensors(params, weights, lambda i: self.model.layers[i], self.config.hidden_size)
        expert_params = {n for n in params if ".mlp.experts." in n}
        for layer in self.model.layers:
            if layer.is_moe:
                g = layer.mlp.gate
                if g.weight.dtype != torch.float32 or g.e_score_correction_bias.dtype != torch.float32:
                    raise RuntimeError(
                        f"beam_hpu: layer {layer.layer_id} router weight/bias not fp32 "
                        f"({g.weight.dtype}, {g.e_score_correction_bias.dtype})"
                    )
        missing = sorted(set(params) - loaded - expert_params)
        if missing:
            raise ValueError(
                f"beam_hpu: parameters not loaded from the checkpoint: {missing[:20]}"
                f"{' ...' if len(missing) > 20 else ''} ({len(missing)})"
            )
        logger.warning(
            "beam_hpu: loaded %d parameters (%d expert tensors owned by other ranks)", len(loaded), len(skipped)
        )
        return set(params)


# checkpoint name piece -> (parameter name piece, shard id); dense MLP only (expert names never contain
# ".mlp.gate_proj."); the attention gate is renamed so it can never match a gate_up_proj shard
_STACKED = (
    (".self_attn.q_proj.", ".self_attn.qkv_proj.", "q"),
    (".self_attn.k_proj.", ".self_attn.qkv_proj.", "k"),
    (".self_attn.v_proj.", ".self_attn.qkv_proj.", "v"),
    (".mlp.gate_proj.", ".mlp.gate_up_proj.", 0),
    (".mlp.up_proj.", ".mlp.gate_up_proj.", 1),
)
_RENAMED = ((".self_attn.gate_proj.", ".self_attn.output_gate_proj."),)


def load_tensors(
    params: dict, weights: Iterable[tuple[str, torch.Tensor]], layer_of, hidden: int
) -> tuple[set[str], list[str]]:
    """Route checkpoint tensors (Hugging Face names) into `params` (vLLM names). Experts go through the
    layer's MoE runner load_weights (FusedMoE expert mapping, EP-aware). Returns (loaded parameter names,
    expert tensors no local expert took). Raises on a tensor without a parameter."""
    loaded: set[str] = set()
    skipped: list[str] = []
    for name, w in weights:
        if "rotary_emb" in name:
            continue
        if name.endswith("lm_head.weight") and w.shape[-1] > hidden:
            w = w[..., :hidden].contiguous()  # strip lm_head input padding, as the reference does
        if ".mlp.experts." in name:
            pre, rest = name.split(".mlp.experts.", 1)
            layer = layer_of(int(pre.rsplit(".", 1)[1]))
            got = list(layer.mlp.experts.load_weights([(rest, w)]))
            for p in got:
                loaded.add(f"{pre}.mlp.experts.routed_experts.{p}")
            if not got:
                skipped.append(name)  # an expert owned by another EP rank
            continue
        for old, new in _RENAMED:
            name = name.replace(old, new)
        for old, new, shard in _STACKED:
            if old in name:
                pname = name.replace(old, new)
                param = params[pname]
                param.weight_loader(param, w, shard)
                loaded.add(pname)
                break
        else:
            if name not in params:
                raise KeyError(f"beam_hpu: checkpoint tensor {name} {tuple(w.shape)} has no parameter")
            param = params[name]
            getattr(param, "weight_loader", default_weight_loader)(param, w)
            loaded.add(name)
    return loaded, skipped


__all__ = ["BeamHPUForCausalLM", "BeamHPUModel", "BeamHPUDecoderLayer", "load_tensors"]
