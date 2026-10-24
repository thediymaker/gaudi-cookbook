# SPDX-License-Identifier: Apache-2.0
"""`glm5_next` configuration classes for GLM-5.3-Flash.

The transformers release in the v0.26.0 image predates `glm5_next`, so `AutoConfig` cannot read the checkpoint. These
classes use the classic `PretrainedConfig` API (explicit `__init__`, kwargs passed through) and every default is the
value in the published config.json. tests/test_config.py checks that parsing that file reproduces it.

The linear-attention geometry is kept both as the nested `linear_attn_config` dict (what vLLM's Kimi KDA layer
reads) and as flat `linear_*` attributes (what the transformers reference reads).
"""

import contextlib

from transformers.configuration_utils import PretrainedConfig


class Glm5NextTextConfig(PretrainedConfig):
    model_type = "glm5_next_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "n_routed_experts"}

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        num_hidden_layers: int = 45,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 64,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        rms_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        head_dim: int = 0,
        tie_word_embeddings: bool = False,
        # --- per-layer schedules (45 entries each) ---
        layer_types: list[str] | None = None,
        mlp_layer_types: list[str] | None = None,
        indexer_types: list[str] | None = None,
        first_k_dense_replace: int = 3,
        # --- MLA / NoPE ---
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_head_dim: int = 256,
        qk_nope_head_dim: int = 256,
        qk_rope_head_dim: int = 0,
        v_head_dim: int = 256,
        mla_use_nope: bool = True,
        # --- DSA indexer ---
        index_topk: int = 2048,
        index_head_dim: int = 128,
        index_n_heads: int = 32,
        index_kpool: int = 4,
        index_kpool_compress: bool = True,
        index_kpool_always_select_tail: bool = True,
        index_share_for_mtp_iteration: bool = True,
        indexer_rope_interleave: bool = True,
        # --- KDA linear attention ---
        linear_attn_config: dict | None = None,
        # --- mHC ---
        mhc: bool = True,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        # --- MoE ---
        moe_intermediate_size: int = 2048,
        n_routed_experts: int = 288,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 8,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 2.5,
        scoring_func: str = "sigmoid",
        topk_method: str = "noaux_tc",
        n_group: int = 1,
        topk_group: int = 1,
        moe_router_dtype: str = "float32",
        router_aux_loss_coef: float = 0.001,
        output_router_logits: bool = False,
        swiglu_limit: float = 10.0,
        swiglu_alpha: float = 1.0,
        swiglu_beta: float = 0.0,
        # --- MTP ---
        num_nextn_predict_layers: int = 1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim

        self.layer_types = layer_types
        self.mlp_layer_types = mlp_layer_types
        self.indexer_types = indexer_types
        self.first_k_dense_replace = first_k_dense_replace

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_head_dim = qk_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope

        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_kpool = index_kpool
        self.index_kpool_compress = index_kpool_compress
        self.index_kpool_always_select_tail = index_kpool_always_select_tail
        self.index_share_for_mtp_iteration = index_share_for_mtp_iteration
        self.indexer_rope_interleave = indexer_rope_interleave

        self.linear_attn_config = linear_attn_config
        # Flat aliases for transformers-style consumers, derived from linear_attn_config (the single source).
        lac = linear_attn_config or {}
        self.linear_num_heads = lac.get("num_heads")
        self.linear_head_dim = lac.get("head_dim")
        self.linear_conv_kernel_dim = lac.get("short_conv_kernel_size")
        self.linear_lower_bound = lac.get("gate_lower_bound")

        self.mhc = mhc
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_router_dtype = moe_router_dtype
        self.router_aux_loss_coef = router_aux_loss_coef
        self.output_router_logits = output_router_logits
        self.swiglu_limit = swiglu_limit
        # Plain clamped SwiGLU, silu(gate) * up (alpha 1, beta 0). vLLM reads these names and otherwise falls back
        # to the SwiGLU-OAI defaults (1.702, 1.0), a different activation.
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta

        self.num_nextn_predict_layers = num_nextn_predict_layers

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Glm5NextVisionConfig(PretrainedConfig):
    model_type = "glm5_next_vision"

    def __init__(
        self,
        depth: int = 24,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_heads: int = 16,
        in_channels: int = 3,
        image_size: int = 448,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 4096,
        projection_intermediate_size: int = 10240,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-5,
        attention_bias: bool = True,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        swiglu_limit: float = 10.0,
        swiglu_alpha: float = 1.0,
        swiglu_beta: float = 0.0,
        **kwargs,
    ):
        self.depth = depth
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.projection_intermediate_size = projection_intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.swiglu_limit = swiglu_limit
        # Plain clamped SwiGLU, silu(gate) * up (alpha 1, beta 0). vLLM reads these names and otherwise falls back
        # to the SwiGLU-OAI defaults (1.702, 1.0), a different activation.
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        super().__init__(**kwargs)


class Glm5NextConfig(PretrainedConfig):
    model_type = "glm5_next"
    sub_configs = {"text_config": Glm5NextTextConfig, "vision_config": Glm5NextVisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config: dict | PretrainedConfig | None = None,
        vision_config: dict | PretrainedConfig | None = None,
        image_token_id: int = 154854,
        video_token_id: int = 154855,
        image_start_token_id: int = 154830,
        image_end_token_id: int = 154831,
        video_start_token_id: int = 154832,
        video_end_token_id: int = 154833,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            self.text_config = Glm5NextTextConfig(**text_config)
        elif text_config is None:
            # A flat text-only checkpoint keeps the text fields at the top level.
            self.text_config = Glm5NextTextConfig(**kwargs)
        else:
            self.text_config = text_config

        if isinstance(vision_config, dict):
            self.vision_config = Glm5NextVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = Glm5NextVisionConfig()
        else:
            self.vision_config = vision_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id

        # vLLM reads these off the top-level config in many places (scheduler, KV cache sizing, logits), so mirror
        # the text values up.
        for attr in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
            "rms_norm_eps",
            "hidden_act",
            "head_dim",
        ):
            setattr(self, attr, getattr(self.text_config, attr))

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


def register() -> None:
    """Register the configs and processors with the transformers Auto classes. Idempotent."""
    from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, AutoVideoProcessor

    for cfg in (Glm5NextTextConfig, Glm5NextVisionConfig, Glm5NextConfig):
        with contextlib.suppress(ValueError):  # already registered
            AutoConfig.register(cfg.model_type, cfg)

    from .hf.image_processing_glm5_next import Glm5NextImageProcessor
    from .hf.processing_glm5_next import Glm5NextProcessor
    from .hf.video_processing_glm5_next import Glm5NextVideoProcessor

    # The video processor is registered although video is not served: processor_config.json declares one, and
    # AutoProcessor builds no processor at all without it.
    for auto, klass in (
        (AutoImageProcessor, Glm5NextImageProcessor),
        (AutoVideoProcessor, Glm5NextVideoProcessor),
        (AutoProcessor, Glm5NextProcessor),
    ):
        with contextlib.suppress(ValueError):  # already registered
            auto.register(Glm5NextConfig, klass)
