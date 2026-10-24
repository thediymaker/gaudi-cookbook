# SPDX-License-Identifier: Apache-2.0
"""quant_method "muse_spark_hpu_mxfp4": serve AMD's Quark MXFP4 checkpoint of Muse Spark Open unmodified.

The checkpoint declares `quant_method: "quark"`, which the HPU platform rejects. Instead of rewriting its
config.json, this config claims it through `override_quantization_method`: vLLM asks registered methods before
its built-in overrides, and a non-None answer replaces the checkpoint's method. The claim is narrow: model_type
`muse_spark`, Quark, and the MXFP4 format the loader and the HPU op expect (fp4 weights, per-group 32, E8M0
scales, real-quantized export). Anything else (another Quark export, `--quantization quark`, another model) is
not claimed and still fails closed on HPU.

The model builds its own expert parameters, loads them with its own loader and drives the HPU MXFP4 MoE op
itself, so this config only hands every linear layer the unquantized (bf16) method.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

NAME = "muse_spark_hpu_mxfp4"
MODEL_TYPE = "muse_spark"


def mxfp4_format_problems(hf_quant_cfg: dict[str, Any] | None) -> list[str]:
    """Why a Quark config is not the MXFP4 export we serve ([] when it is)."""
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
    if hf_quant_cfg.get("layer_type_quant_config") or hf_quant_cfg.get("layer_quant_config"):
        problems.append("per-layer Quark overrides present")
    fmt = (hf_quant_cfg.get("export") or {}).get("weight_format")
    if fmt != "real_quantized":
        problems.append(f"export.weight_format {fmt!r} != 'real_quantized'")
    return problems


@register_quantization_config(NAME)
class MuseSparkHPUMxfp4Config(QuantizationConfig):
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.cfg = cfg or {}
        problems = mxfp4_format_problems(self.cfg)
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
    def from_config(cls, config: dict) -> MuseSparkHPUMxfp4Config:
        return cls(config)

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg: dict[str, Any], user_quant: str | None, hf_config: Any = None
    ) -> str | None:
        if user_quant not in (None, NAME):
            return None
        if getattr(hf_config, "model_type", None) != MODEL_TYPE:
            return None
        if mxfp4_format_problems(hf_quant_cfg):
            return None
        return NAME

    def get_quant_method(self, layer, prefix: str):
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None
