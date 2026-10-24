# SPDX-License-Identifier: Apache-2.0
"""quant_method "dsv41_hpu_mxfp4": the label the converted checkpoint carries (convert/ in this port).

It exists so that the checkpoint FAILS CLOSED on any image without this plugin
(vLLM rejects an unknown quant_method) instead of loading as plain fp8 and
routing FP4 experts to the wrong kernel. The model builds its own expert
parameters and routes them to the native HPU MXFP4 op itself, so the config only
has to hand every linear layer the unquantized (bf16) method.
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

NAME = "dsv41_hpu_mxfp4"


@register_quantization_config(NAME)
class DSV41HPUMxfp4Config(QuantizationConfig):
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.cfg = cfg or {}
        layout = (self.cfg.get("routed_experts") or {}).get("w13_layout", "interleaved_gate_up")
        if layout != "interleaved_gate_up":
            raise ValueError(
                f"dsv41: checkpoint w13 layout {layout!r}; the clamp-capable bias kernel "
                "needs interleaved_gate_up (reconvert with --w13-layout interleaved)"
            )

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
    def from_config(cls, config: dict) -> DSV41HPUMxfp4Config:
        return cls(config)

    def get_quant_method(self, layer, prefix: str):
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None
