# SPDX-License-Identifier: Apache-2.0
"""Config for model_type `inkling_mm_model`, text only.

The checkpoint nests the language model under `text_config` next to the vision, audio and MTP configs.
This config flattens `text_config` to the top level, drops the towers (they are not built) and adds the
attributes vLLM and the HPU runner key on: `layer_types` and `sliding_window` (interleaved sliding-window
metadata for the 35 local layers), `max_position_embeddings` and the padded vocab size.
"""

from transformers import PretrainedConfig


class InklingHPUConfig(PretrainedConfig):
    model_type = "inkling_mm_model"

    def __init__(self, **kwargs):
        text = dict(kwargs.pop("text_config", None) or {})
        for tower in ("audio_config", "vision_config", "mtp_config"):
            kwargs.pop(tower, None)

        text.setdefault("hidden_size", 4096)
        text.setdefault("num_hidden_layers", 42)
        text.setdefault("num_attention_heads", 32)
        text.setdefault("num_key_value_heads", 8)
        text.setdefault("head_dim", 128)
        text.setdefault("rms_norm_eps", 1e-6)
        text.setdefault("d_rel", 16)
        text.setdefault("rel_extent", 1024)
        text.setdefault("sliding_window_size", 512)
        text.setdefault("local_layer_ids", [])
        text.setdefault("sconv_kernel_size", 4)
        text.setdefault("dense_mlp_idx", 0)
        text.setdefault("dense_intermediate_size", text.get("intermediate_size", 16384))
        for key, value in text.items():
            setattr(self, key, value)

        # vocab_size in the checkpoint is the padded table size; unpadded_vocab_size is the real one.
        padded = text.get("vocab_size", 201024)
        self.padded_vocab_size = padded
        self.vocab_size = min(text.get("unpadded_vocab_size", padded), padded)

        self.max_position_embeddings = text.get("model_max_length", 1048576)
        self.sliding_window = self.sliding_window_size
        local = set(self.local_layer_ids)
        self.layer_types = [
            "sliding_attention" if i in local else "full_attention" for i in range(self.num_hidden_layers)
        ]
        kwargs.setdefault("torch_dtype", text.get("torch_dtype", "bfloat16"))
        super().__init__(**kwargs)

    def get_text_config(self, *args, **kwargs):
        return self


def register_config() -> None:
    """Point both config registries at InklingHPUConfig. vLLM maps inkling_mm_model to its own config class
    and registers that with AutoConfig when it parses a config, so both must be overridden."""
    import vllm.transformers_utils.config as vllm_config_utils
    from transformers import AutoConfig

    AutoConfig.register("inkling_mm_model", InklingHPUConfig, exist_ok=True)
    vllm_config_utils._CONFIG_REGISTRY["inkling_mm_model"] = InklingHPUConfig
