"""Mistral Large 4 config: the text-only switch and the per-layer tables derived from params.json.

Text-only switch. With `--config-format mistral`, vLLM parses params.json through
adapt_config_dict (vendored from PR #64534). params.json carries a `vision_encoder` block, so the adapter wraps
the text config as PixtralForConditionalGeneration. The port serves text only: install() wraps the adapter so
that, for a Mistral Large 4 params dict (`dsa` present) and ML4_VISION unset, the `vision_encoder` block is
dropped and the top-level architecture is the text model, MistralLarge4ForCausalLM. The adapter also back-fills
every key of the HF config.json (`defaults`) that params.json lacks; for this checkpoint those are the Pixtral
wrapper's keys (text_config, vision_config, image_token_index, ...), which must not land on a text config, so
in text-only mode only TEXT_DEFAULTS pass. ML4_VISION=1 is refused (no vision path in this port).

Per-layer tables. The adapter expands params.json's patterns into per-layer lists on the text config:
  nope_layer_flags          True for NoPE layers (`nope: rope,rope,rope,nope` -> 3, 7, ..., 71)
  kv_source_layers          layer whose latent cache a layer reads (itself if it owns one; 35 for 39, 43, ..., 71)
  index_sharing             "full" (own indexer) or "shared" (reuse the previous full layer's top-k)
  indexer_nope_layer_flags  NoPE flags of the indexer (= nope_layer_flags: dsa.match_nope)
layer_tables() turns them into what the HPU stack indexes by layer (the PR's common/layer_spec.py, plus the
inverse maps): KV owners and their readers, index source per layer. Pure Python, no torch, no HPU.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("vllm.mistral_large4_hpu")

MODEL_TYPE = "mistral_large_4"  # vLLM's spelling (HF: mistral_large4)
TEXT_ARCH = "MistralLarge4ForCausalLM"
VISION = os.environ.get("ML4_VISION", "0") == "1"
# config.json keys that may back-fill a text-only config; everything else in it describes the Pixtral wrapper
TEXT_DEFAULTS = frozenset({"dtype", "torch_dtype", "tie_word_embeddings", "transformers_version"})
_MARK = "_mistral_large4_hpu_text_only"


def is_ml4_params(config_dict: dict[str, Any]) -> bool:
    """A raw params.json dict of Mistral Large 4 (the adapter's own test: a `dsa` block)."""
    return bool(config_dict.get("dsa"))


def text_only_args(config_dict: dict[str, Any], defaults: dict[str, Any]) -> tuple[dict, dict]:
    """(params dict without vision_encoder, defaults reduced to TEXT_DEFAULTS); inputs are not modified."""
    dropped = sorted(k for k in defaults if k not in TEXT_DEFAULTS)
    if dropped:
        logger.debug("mistral_large4_hpu: text-only, config.json keys not back-filled: %s", dropped)
    return (
        {k: v for k, v in config_dict.items() if k != "vision_encoder"},
        {k: v for k, v in defaults.items() if k in TEXT_DEFAULTS},
    )


def install() -> None:
    """Wrap adapt_config_dict with the text-only switch. Idempotent. The vision tower is not part of this port."""
    if VISION:
        raise RuntimeError("mistral_large4_hpu: ML4_VISION=1 is not supported by this port (text only)")
    import vllm.transformers_utils.configs.mistral as M

    if getattr(M.adapt_config_dict, _MARK, False):
        return
    orig = M.adapt_config_dict

    def adapt_config_dict(config_dict, defaults):
        if is_ml4_params(config_dict):
            config_dict, defaults = text_only_args(config_dict, defaults)
        return orig(config_dict, defaults)

    setattr(adapt_config_dict, _MARK, True)
    adapt_config_dict.__wrapped__ = orig
    # MistralConfigParser imports adapt_config_dict from the module at call time, so the module attribute is
    # the one hook needed
    M.adapt_config_dict = adapt_config_dict


def text_config(config: Any) -> Any:
    """The Mistral Large 4 text config of either the text config itself or the Pixtral wrapper."""
    if getattr(config, "model_type", None) == MODEL_TYPE:
        return config
    tc = getattr(config, "text_config", None)
    if tc is not None and getattr(tc, "model_type", None) == MODEL_TYPE:
        return tc
    raise ValueError(f"not a Mistral Large 4 config: model_type {getattr(config, 'model_type', None)!r}")


@dataclass(frozen=True)
class LayerTables:
    """Per-layer structure of the decoder stack (index = layer id)."""

    num_layers: int
    nope: list[bool]  # NoPE attention (no rotary on q_pe / k_pe)
    kv_source: list[int]  # latent cache this layer reads
    index_full: list[bool]  # runs its own indexer
    index_source: list[int]  # the full-indexer layer whose top-k this layer uses (itself if full)
    indexer_nope: list[bool]  # NoPE indexer (no rotary on its q / k)
    readers: dict[int, list[int]] = field(default_factory=dict)  # KV owner -> layers that read it (incl. itself)

    @property
    def owns_kv(self) -> list[bool]:
        return [s == i for i, s in enumerate(self.kv_source)]

    @property
    def kv_owners(self) -> list[int]:
        return [i for i, s in enumerate(self.kv_source) if s == i]

    @property
    def kv_shared(self) -> list[int]:
        return [i for i, s in enumerate(self.kv_source) if s != i]

    @property
    def index_full_layers(self) -> list[int]:
        return [i for i, f in enumerate(self.index_full) if f]

    @property
    def nope_layers(self) -> list[int]:
        return [i for i, n in enumerate(self.nope) if n]

    def summary(self) -> dict[str, Any]:
        shared_to = sorted({self.kv_source[i] for i in self.kv_shared})
        return {
            "layers": self.num_layers,
            "kv_owners": len(self.kv_owners),
            "kv_shared": len(self.kv_shared),
            "kv_shared_sources": shared_to,
            "index_full": len(self.index_full_layers),
            "index_shared": self.num_layers - len(self.index_full_layers),
            "nope": len(self.nope_layers),
            "indexer_nope": sum(self.indexer_nope),
            "max_readers": max(len(r) for r in self.readers.values()),
        }


def layer_tables(config: Any) -> LayerTables:
    """Derive LayerTables from an adapted config (text config or Pixtral wrapper); checks what the HPU stack
    relies on: a shared layer reads an earlier owner with the same NoPE flag, layer 0 has a full indexer."""
    tc = text_config(config)
    n = int(tc.num_hidden_layers)
    nope = [bool(x) for x in (getattr(tc, "nope_layer_flags", None) or [False] * n)]
    kv_source = [int(x) for x in (getattr(tc, "kv_source_layers", None) or range(n))]
    sharing = list(getattr(tc, "index_sharing", None) or ["full"] * n)
    indexer_nope = [bool(x) for x in (getattr(tc, "indexer_nope_layer_flags", None) or [False] * n)]
    if not (len(nope) == len(kv_source) == len(sharing) == len(indexer_nope) == n):
        raise ValueError(f"per-layer tables do not cover {n} layers")
    if set(sharing) - {"full", "shared"} or sharing[0] != "full":
        raise ValueError(f"index_sharing must be full/shared starting with full: {sharing[:4]}...")
    index_full = [s == "full" for s in sharing]
    index_source, last = [], -1
    for i, full in enumerate(index_full):
        last = i if full else last
        index_source.append(last)
    readers: dict[int, list[int]] = {}
    for i, s in enumerate(kv_source):
        if not (s <= i and kv_source[s] == s and nope[s] == nope[i]):
            raise ValueError(f"layer {i} reads layer {s}: not an earlier KV owner with the same NoPE flag")
        readers.setdefault(s, []).append(i)
    return LayerTables(n, nope, kv_source, index_full, index_source, indexer_nope, readers)
