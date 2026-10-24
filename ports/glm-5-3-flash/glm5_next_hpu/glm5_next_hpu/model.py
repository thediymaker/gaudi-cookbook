# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash (`Glm5NextForConditionalGeneration`) for vLLM on Gaudi 2.

Written from the transformers reference, assembled from parts that already exist in vLLM v0.26.0:

* MoE and dense MLP: `DeepseekV2MoE` / `DeepseekV2MLP` (the text config has every field they read), with GLM's
  clamped SwiGLU swapped in (see `_apply_swiglu_clamp`).
* mHC: vLLM's `MHCPreOp` / `MHCPostOp` (pure torch on Gaudi; see mhc.py for the Sinkhorn kernel).
* KDA linear attention: `kda_layer.Glm5NextHPUKDAAttention`, a subclass of vLLM's Kimi KDA layer with an HPU forward.
* NoPE MLA: vLLM's `MultiHeadLatentAttentionWrapper`; the nope-mla patch fixes the one reshape that fails at
  qk_rope_head_dim == 0.

Not built: the DSA lightning indexer (vllm-gaudi runs these layers as dense MLA, which is the same function for prompts
up to 2,048 tokens and an approximation beyond), and the MTP layer. Their checkpoint tensors are skipped and counted.

Differences from upstream vLLM's GPU implementation (vllm-project/vllm#53906): that one runs the kpool sparse
indexer, MTP and Triton / FlashInfer kernels. None of those run on Gaudi; this file keeps the HPU path only.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mhc import MHCPostOp, MHCPreOp
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_v2 import DeepSeekV2FusedQkvAProjLinear, DeepseekV2MLP, DeepseekV2MoE
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from .multimodal import (
    VISION_ENABLED,
    Glm5NextDummyInputsBuilder,
    Glm5NextMultiModalProcessor,
    Glm5NextProcessingInfo,
)

logger = init_logger("vllm.glm5_next_hpu")  # under vllm.* so records reach a handler

# GLM5_NEXT_DECODE_LAYER_GROUP=G (default 1 = off). vllm-gaudi's runner marks a lazy-mode step after every module whose
# class name ends in "DecoderLayer". In decode, the attention and MoE all-reduces already cut each layer's graph, so
# that hook only adds one more small graph launch per layer, and a one-stream decode step is bound by host launch time.
# With G > 1 a pure decode step calls layer.forward() (which skips the hook) and marks a step every G layers and after
# the last; G = 45 gives one mark for the whole stack. Prompt and mixed steps keep the per-layer hook, which bounds the
# host run-ahead and the prompt memory peak. Same ops in the same order: outputs are bit-identical.
DECODE_LAYER_GROUP = max(1, int(os.environ.get("GLM5_NEXT_DECODE_LAYER_GROUP", "1")))


def _decode_layer_group() -> int:
    """G for a pure decode forward, 1 otherwise."""
    if DECODE_LAYER_GROUP <= 1:
        return 1
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return 1
    md = get_forward_context().attn_metadata
    if md is None or isinstance(md, dict) or getattr(md, "is_prompt", True):
        return 1
    return DECODE_LAYER_GROUP


def _mark_step() -> None:
    import habana_frameworks.torch as htorch

    htorch.core.mark_step()


class Glm5NextMLAAttention(nn.Module):
    """NoPE MLA: DeepSeek-V3.2's MLA without RoPE (qk_rope_head_dim == 0) and run dense, without the sparse indexer."""

    def __init__(self, config, quant_config: QuantizationConfig | None, cache_config, prefix: str = ""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.num_heads = config.num_attention_heads

        tp_size = get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0, f"num_attention_heads={self.num_heads} must divide TP={tp_size}"
        self.num_local_heads = self.num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5  # NoPE: no yarn mscale correction
        assert self.q_lora_rank is not None, "MLAModules needs fused_qkv_a_proj when q_lora_rank is set"

        self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
            self.hidden_size,
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=None,  # NoPE
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj,
            kv_a_proj_with_mqa=None,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=None,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
            indexer_rotary_emb=None,
        )
        self.mla_attn = MultiHeadLatentAttentionWrapper(
            hidden_size=self.hidden_size,
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            mla_modules=mla_modules,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states)


def _apply_swiglu_clamp(module: nn.Module, limit: float, where: str) -> int:
    """Replace every `SiluAndMul` under `module` with GLM's clamped SwiGLU and return how many were replaced.

    The reference clamps before the product: `silu(gate.clamp(max=L)) * up.clamp(-L, L)` with L = 10.
    `DeepseekV2MLP` hardcodes `SiluAndMul`, so reuse would drop the clamp on the dense MLPs (layers 0 to 2) and on
    every shared expert. `SiluAndMulWithClamp(limit, alpha=1, beta=0)` is exactly the reference. The 288 routed experts
    run inside the fused FP8 MoE op, whose activation cannot be swapped this way; there the clamp binds on about 1
    element in 10^5 to 10^7.
    """
    replaced = 0
    for mod in module.modules():
        for name, child in list(mod.named_children()):
            if isinstance(child, SiluAndMul):
                setattr(mod, name, SiluAndMulWithClamp(swiglu_limit=limit, alpha=1.0, beta=0.0))
                replaced += 1
    if replaced == 0:
        raise RuntimeError(f"GLM-5.3-Flash SwiGLU clamp: no SiluAndMul found in {where}; vLLM's MLP layout changed")
    return replaced


class Glm5NextDecoderLayer(nn.Module):
    """One layer with an mHC site around attention and another around the MLP:

    post, comb, x = hc_pre(streams); x = attn(norm(x)); streams = post * x + comb^T @ streams;
    then the same for the MLP.
    """

    _vllm_config: VllmConfig | None = None  # set by Glm5NextModel before the layers are built

    def __init__(self, config, layer_idx: int, quant_config, cache_config, parallel_config, prefix: str = ""):
        super().__init__()
        self.layer_idx = layer_idx
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_post_alpha = 2.0  # GLM: post = 2 * sigmoid(...)
        self.is_kda = config.layer_types[layer_idx] == "linear_attention"

        if self.is_kda:
            from .kda_layer import Glm5NextHPUKDAAttention

            self.self_attn = Glm5NextHPUKDAAttention(
                config=config,
                vllm_config=self._vllm_config,
                prefix=f"{prefix}.self_attn",
                gate_lower_bound=config.linear_lower_bound,
            )
        else:
            self.self_attn = Glm5NextMLAAttention(config, quant_config, cache_config, prefix=f"{prefix}.self_attn")

        limit = float(getattr(config, "swiglu_limit", 10.0))
        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = DeepseekV2MoE(
                config=config, parallel_config=parallel_config, quant_config=quant_config, prefix=f"{prefix}.mlp"
            )
            _apply_swiglu_clamp(self.mlp, limit, f"layer {layer_idx} shared expert")
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            _apply_swiglu_clamp(self.mlp, limit, f"layer {layer_idx} dense MLP")

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # mHC parameters must be fp32 (vLLM's mHC op asserts it); the checkpoint stores bf16 and the loader upcasts.
        mix = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * config.hidden_size

        def mk(*shape):
            return nn.Parameter(torch.empty(*shape, dtype=torch.float32), requires_grad=False)

        self.hc_attn_fn = mk(mix, hc_dim)
        self.hc_ffn_fn = mk(mix, hc_dim)
        self.hc_attn_base = mk(mix)
        self.hc_ffn_base = mk(mix)
        self.hc_attn_scale = mk(3)
        self.hc_ffn_scale = mk(3)
        self.mhc_pre = MHCPreOp()
        self.mhc_post = MHCPostOp()

    def _hc_pre(self, streams, fn, scale, base):
        return self.mhc_pre(
            residual=streams,
            fn=fn,
            hc_scale=scale,
            hc_base=base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.hc_post_alpha,
            sinkhorn_repeat=self.hc_sinkhorn_iters,
        )

    def forward(self, positions: torch.Tensor, streams: torch.Tensor) -> torch.Tensor:
        residual = streams
        post, comb, x = self._hc_pre(streams, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.input_layernorm(x)
        if self.is_kda:
            out = torch.empty_like(x)
            self.self_attn(x, positions, out)
            x = out
        else:
            x = self.self_attn(positions, x)
        streams = self.mhc_post(x, residual, post, comb)

        residual = streams
        post, comb, x = self._hc_pre(streams, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return self.mhc_post(x, residual, post, comb)


class Glm5NextModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        top_cfg = vllm_config.model_config.hf_config
        config = getattr(top_cfg, "text_config", top_cfg)
        self.config = config
        self.hc_mult = config.hc_mult
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config

        self.vocab_size = config.vocab_size
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size, quant_config=quant_config, prefix=f"{prefix}.embed_tokens"
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # The KDA layer takes vllm_config; the layer factory below reads it from the class.
        Glm5NextDecoderLayer._vllm_config = vllm_config
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Glm5NextDecoderLayer(
                config,
                int(prefix.rsplit(".", 1)[-1]),
                quant_config,
                cache_config,
                parallel_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size * config.hc_mult
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        # vllm-gaudi hands the model [batch, seq, hidden]; every reused layer expects vLLM's flat [tokens, hidden].
        # Flatten once here and restore the shape on the way out.
        if get_pp_group().is_first_rank:
            emb = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            lead = emb.shape[:-1]
            emb = emb.reshape(-1, emb.shape[-1])
            streams = emb.unsqueeze(-2).expand(-1, self.hc_mult, -1).contiguous()  # one stream becomes hc_mult
        else:
            assert intermediate_tensors is not None
            flat = intermediate_tensors["hidden_states"]
            lead = flat.shape[:-1]
            streams = flat.reshape(-1, self.hc_mult, self.config.hidden_size)

        group = _decode_layer_group()
        n_layers = self.end_layer - self.start_layer
        for i, layer in enumerate(self.layers[self.start_layer : self.end_layer]):
            if group > 1:
                streams = layer.forward(positions, streams)
                if (i + 1) % group == 0 or i + 1 == n_layers:
                    _mark_step()
            else:
                streams = layer(positions, streams)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": streams.flatten(-2).reshape(*lead, -1)})
        # GLM's head is an unweighted mean of the streams (unlike DeepSeek-V4's HCHeadOp).
        out = self.norm(streams.mean(dim=-2))
        return out.reshape(*lead, out.shape[-1])


@MULTIMODAL_REGISTRY.register_processor(
    Glm5NextMultiModalProcessor, info=Glm5NextProcessingInfo, dummy_inputs=Glm5NextDummyInputsBuilder
)
class Glm5NextForConditionalGeneration(nn.Module, HasInnerState, SupportsPP, IsHybrid, SupportsMultiModal):
    # IsHybrid is load-bearing: vLLM reads `is_hybrid` off the class. Without it every layer counts as attention,
    # vllm-gaudi never fills the state fields of its metadata, every KDA layer returns early, and generation still
    # produces plausible tokens.

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        top_cfg = vllm_config.model_config.hf_config
        config = getattr(top_cfg, "text_config", top_cfg)
        # self.config is the top-level config: vllm-gaudi's runner reads `model.config.architectures`, which the
        # nested text config does not have.
        self.config = top_cfg
        self.text_config = config
        self.quant_config = vllm_config.quant_config

        # Keep the attribute name `visual`: vllm-gaudi's vision graph wrap looks for `vision_tower`, and capturing one
        # graph per image geometry would be unbounded device memory. Eager, the tower costs only compile time.
        self.visual = None
        if VISION_ENABLED:
            from .vision import Glm5NextVisionModel

            vision_cfg = getattr(top_cfg, "vision_config", None)
            if vision_cfg is None:
                raise ValueError("GLM5_NEXT_VISION=1 but the config has no vision_config")
            self.visual = Glm5NextVisionModel(vision_cfg)

        self.model = Glm5NextModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        """KDA conv and recurrent state shapes: Kimi Linear's calculator, reading the nested text config."""
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

        top = vllm_config.model_config.hf_config
        lac = getattr(top, "text_config", top).linear_attn_config
        num_spec = vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config else 0
        return MambaStateShapeCalculator.kda_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            lac["num_heads"],
            lac["head_dim"],
            conv_kernel_size=lac["short_conv_kernel_size"],
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        """fp32 recurrent state, model-dtype conv state (as the layer's own get_state_dtype)."""
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator

        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFuncCalculator

        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def get_language_model(self):
        return self.model

    def embed_input_ids(
        self, input_ids: torch.Tensor, multimodal_embeddings=None, *, is_multimodal: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Text only: the language model's embedding. Otherwise the interface's version, which scatters the image rows.
        if multimodal_embeddings is None and is_multimodal is None:
            return self.model.embed_input_ids(input_ids)
        return SupportsMultiModal.embed_input_ids(self, input_ids, multimodal_embeddings, is_multimodal=is_multimodal)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|begin_of_image|><|image|><|end_of_image|>"
        raise ValueError(f"only images are supported, got {modality!r}")

    def embed_multimodal(self, **kwargs):
        """One embedding tensor per image, in request order (vLLM 0.26 calls this name)."""
        pixel_values = kwargs.get("pixel_values")
        grid_thw = kwargs.get("image_grid_thw")
        if pixel_values is None or grid_thw is None:
            return None
        if self.visual is None:
            raise RuntimeError("glm5_next_hpu: image input but the vision tower is not built (GLM5_NEXT_VISION=1)")
        if isinstance(pixel_values, list):
            pixel_values = torch.cat([p.reshape(-1, p.shape[-1]) for p in pixel_values], dim=0)
        if isinstance(grid_thw, list):
            grid_thw = torch.cat([g.reshape(-1, 3) for g in grid_thw], dim=0)
        grid_thw = grid_thw.reshape(-1, 3).to(torch.long)
        param = next(self.visual.parameters())
        out = self.visual(pixel_values.to(device=param.device, dtype=param.dtype), grid_thw.to(param.device))
        # Rows per image = the placeholder count the processor wrote: grid.prod() // merge_size**2.
        merge = self.visual.spatial_merge_size**2
        sizes = [int(g.prod()) // merge for g in grid_thw]
        if sum(sizes) != out.shape[0]:
            raise ValueError(f"vision tower returned {out.shape[0]} rows, grid_thw implies {sum(sizes)} ({sizes})")
        return list(torch.split(out, sizes))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """DeepSeek-V2's loader order, plus the skips and the completeness check below."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
        ]
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.text_config.n_routed_experts,
        )
        params_dict = dict(self.named_parameters())
        vision_params = dict(self.visual.named_parameters()) if self.visual is not None else {}
        loaded: set[str] = set()
        skipped = {"vision": 0, "mtp": 0, "indexer": 0}
        rejected = 0
        mtp_prefix = f"model.language_model.layers.{self.text_config.num_hidden_layers}."

        for name, w in weights:
            if name.startswith("model.visual."):
                if self.visual is None:
                    skipped["vision"] += 1
                    continue
                vname = name[len("model.visual.") :]
                if vname not in vision_params:
                    raise ValueError(f"checkpoint tensor {name} has no parameter in the vision tower")
                default_weight_loader(vision_params[vname], w)
                loaded.add("visual." + vname)
                continue
            if name.startswith(mtp_prefix):
                skipped["mtp"] += 1
                continue
            if ".indexer." in name:
                skipped["indexer"] += 1  # DSA layers run as dense MLA: no indexer is built
                continue

            name = name.replace("model.language_model.", "model.", 1)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # A routed expert's gate_proj / up_proj belongs to FusedMoE (below), not to gate_up_proj.
                if "mlp.experts." in name and name not in params_dict:
                    continue
                mapped = name.replace(weight_name, param_name)
                if param_name == "fused_qkv_a_proj" and mapped not in params_dict:
                    continue
                name = mapped
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                param.weight_loader(param, w, shard_id)
                loaded.add(name)
                break
            else:
                is_expert_weight = False
                for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    # Under expert parallelism another rank may own this expert; that is not an error.
                    is_expert_weight = True
                    mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(mapped, self):
                        continue
                    param = params_dict[mapped]
                    success = param.weight_loader(
                        param, w, mapped, shard_id=shard_id, expert_id=expert_id, return_success=True
                    )
                    # vllm-gaudi's synced weight loader discards the inner return value, so a loaded expert reports
                    # None. Only an explicit False is a rejection.
                    if success is not False:
                        loaded.add(mapped)
                        break
                    rejected += 1
                else:
                    if is_expert_weight:
                        continue
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        raise KeyError(f"unexpected checkpoint tensor {name!r}")
                    param = params_dict[name]
                    if param.dtype == torch.float32 and w.dtype != torch.float32:
                        w = w.float()  # mHC parameters: fp32 in the model, bf16 in the checkpoint
                    getattr(param, "weight_loader", default_weight_loader)(param, w)
                    loaded.add(name)

        if rejected:
            logger.warning("glm5_next_hpu: %d expert tensors belong to other expert-parallel ranks", rejected)
        missing = sorted({n for n, _ in self.named_parameters()} - loaded)
        logger.info(
            "glm5_next_hpu: loaded %d parameters; skipped %d vision, %d MTP, %d indexer tensors",
            len(loaded),
            skipped["vision"],
            skipped["mtp"],
            skipped["indexer"],
        )
        # A parameter never loaded holds uninitialised memory, and the model still talks fluently on most of it, so a
        # partial load must not serve.
        if missing:
            raise RuntimeError(f"glm5_next_hpu: {len(missing)} parameters were never loaded (first: {missing[:5]})")
        return loaded
