"""quant_method "mistral_large4_hpu_mxfp4": serves AMD's Quark MXFP4 checkpoint of Mistral Large 4 unmodified.

With `--config-format mistral` the quantization config is params.json's `quantization_config`, which the
adapter passes through as is: `quant_method: "quark"`, which the HPU platform rejects. As for Muse
(muse_spark_hpu/quant.py), this config claims it through `override_quantization_method`: vLLM probes custom
(registered) methods before its built-in overrides, and a non-None answer replaces the checkpoint's method.

The claim is narrow: model_type `mistral_large_4` (the text config, or its Pixtral wrapper), Quark, real-quantized
export, and exactly the two formats the loader handles:
  - global (routed experts): fp4 weights, per-group 32, E8M0 scales (MXFP4);
  - per-layer overrides (layer_quant_config): fp8_e4m3 weights in 128x128 blocks with float32 scales, only on
    the non-expert linears the loader dequantizes to BF16 on the host: attention wq / wo, the
    indexer's wq_b, the shared experts and the layer-0 dense FFN. The AMD export has 404 of them.
Anything else (an FP8 per-type override, KV-cache quantization, an override on another module, a non-MXFP4 global
format, another model) is not claimed and still fails closed on HPU, as does this checkpoint on any image without
the plugin.

The model builds its own expert parameters, loads them with its own loader and drives the MXFP4 kernels itself,
so this config only has to hand every linear layer the unquantized (bf16) method.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

NAME = "mistral_large4_hpu_mxfp4"
MODEL_TYPE = "mistral_large_4"
# modules that may carry the FP8 block override (Mistral-native names)
FP8_MODULES = re.compile(r"layers\.\d+\.(attention\.(wq|wo|indexer\.wq_b)|shared_experts\.w[123]|feed_forward\.w[123])")
FP8_WEIGHT = {"dtype": "fp8_e4m3", "qscheme": "per_block", "block_size": [128, 128], "scale_type": "float32"}
FP8_ACT = {"dtype": "fp8_e4m3", "qscheme": "per_group", "group_size": 128, "is_dynamic": True}


def format_problems(hf_quant_cfg: dict[str, Any] | None) -> list[str]:
    """Why a Quark config is not the MXFP4 + FP8-block export we serve ([] when it is)."""
    if not isinstance(hf_quant_cfg, dict):
        return ["no quantization_config"]
    problems = []
    if str(hf_quant_cfg.get("quant_method", "")).lower() != "quark":
        problems.append(f"quant_method {hf_quant_cfg.get('quant_method')!r} != 'quark'")
    weight = (hf_quant_cfg.get("global_quant_config") or {}).get("weight") or {}
    expect = {"dtype": "fp4", "qscheme": "per_group", "group_size": 32, "scale_format": "e8m0"}
    for key, want in expect.items():
        if weight.get(key) != want:
            problems.append(f"global_quant_config.weight.{key} {weight.get(key)!r} != {want!r}")
    if hf_quant_cfg.get("layer_type_quant_config"):
        problems.append("per-type Quark overrides present")
    if hf_quant_cfg.get("kv_cache_quant_config"):
        problems.append("KV cache quantization present")
    for name, layer in (hf_quant_cfg.get("layer_quant_config") or {}).items():
        if not FP8_MODULES.fullmatch(name):
            problems.append(f"override on {name!r}: not an FP8-block linear the loader handles")
            break
        w, a = layer.get("weight") or {}, layer.get("input_tensors") or {}
        bad = [f"weight.{k}" for k, v in FP8_WEIGHT.items() if w.get(k) != v] + [
            f"input_tensors.{k}" for k, v in FP8_ACT.items() if a.get(k) != v
        ]
        if bad:
            problems.append(f"override on {name!r} is not FP8 128x128 block: {', '.join(bad)}")
            break
    fmt = (hf_quant_cfg.get("export") or {}).get("weight_format")
    if fmt != "real_quantized":
        problems.append(f"export.weight_format {fmt!r} != 'real_quantized'")
    return problems


def _model_type(hf_config: Any) -> str | None:
    mt = getattr(hf_config, "model_type", None)
    if mt != MODEL_TYPE:  # the Pixtral wrapper (ML4_VISION=1) carries the text config inside
        mt = getattr(getattr(hf_config, "text_config", None), "model_type", mt)
    return mt


@register_quantization_config(NAME)
class MistralLarge4HPUMxfp4Config(QuantizationConfig):
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.cfg = cfg or {}
        problems = format_problems(self.cfg)
        if problems:
            raise ValueError(f"{NAME}: not the AMD Quark MXFP4 export: {'; '.join(problems)}")

    @classmethod
    def get_name(cls) -> str:
        return NAME

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "MistralLarge4HPUMxfp4Config":
        return cls(config)

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg: dict[str, Any], user_quant: str | None, hf_config: Any = None
    ) -> str | None:
        if user_quant not in (None, NAME):
            return None
        if _model_type(hf_config) != MODEL_TYPE:
            return None
        if format_problems(hf_quant_cfg):
            return None
        return NAME

    def get_quant_method(self, layer, prefix: str):
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None
