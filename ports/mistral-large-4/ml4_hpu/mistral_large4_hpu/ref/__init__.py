"""Pure-torch reference (CPU oracle) for Mistral Large 4. No vllm imports. See README.md."""

from .layer_spec import LayerSpec, RefConfig, get_layer_specs
from .stack import CONTROLS, ForwardOut, RefModel, RefOptions, SeqState
from .weights import ExpertSource, TensorSource

__all__ = [
    "LayerSpec",
    "RefConfig",
    "get_layer_specs",
    "CONTROLS",
    "ForwardOut",
    "RefModel",
    "RefOptions",
    "SeqState",
    "ExpertSource",
    "TensorSource",
]
