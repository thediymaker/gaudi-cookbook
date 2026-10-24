# SPDX-License-Identifier: Apache-2.0
"""vLLM multimodal plumbing for GLM-5.3-Flash images.

GLM5_NEXT_VISION=1 builds the vision tower and admits images; with it off the model is text-only and an image request
gets HTTP 400 at admission. The model reads `VISION_ENABLED` from here, so admission and the tower always agree.

GLM5_NEXT_IMAGE_MAX_TOKENS (default 2048) caps the tokens of one image. The checkpoint allows 8,000. On Gaudi each new
prompt length compiles a graph, and an image is caller-sized prompt length, so the cap bounds both the compile cost
and the device memory one image can take. `smart_resize` spends the budget on the whole canvas; no image is dropped.

Video is not served: the processor is registered because processor_config.json requires it, but no video is
admitted.

Token accounting: each image becomes `grid_thw.prod() // merge_size**2` placeholder tokens, which must equal the rows
the tower returns for it (both are derived from grid_thw).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from transformers import BatchFeature
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)

VISION_ENABLED = os.environ.get("GLM5_NEXT_VISION", "0") == "1"

IMAGE_MAX_TOKENS = int(os.environ.get("GLM5_NEXT_IMAGE_MAX_TOKENS", "2048"))
if IMAGE_MAX_TOKENS < 16:
    raise ValueError(f"GLM5_NEXT_IMAGE_MAX_TOKENS={IMAGE_MAX_TOKENS} is below the processor minimum of 16")

# Processor kwargs a request may not set: the patch geometry must match the tower, and do_resize=False would skip
# smart_resize and with it the token cap.
_GEOMETRY_KWARGS = ("patch_size", "merge_size", "temporal_patch_size", "patch_expand_factor", "do_resize")


def clamp_request_kwargs(mm_kwargs: Mapping[str, object]) -> dict[str, object]:
    """A request's processor kwargs with geometry keys dropped and `max_image_tokens` capped (at the top level and in
    `images_kwargs`). A call-time kwarg would otherwise win over the cap set on the processor instance."""
    out = {k: v for k, v in mm_kwargs.items() if k not in _GEOMETRY_KWARGS}
    if "max_image_tokens" in out:
        out["max_image_tokens"] = min(int(out["max_image_tokens"]), IMAGE_MAX_TOKENS)
    nested = out.get("images_kwargs")
    if isinstance(nested, Mapping):
        nested = {k: v for k, v in nested.items() if k not in _GEOMETRY_KWARGS}
        if "max_image_tokens" in nested:
            nested["max_image_tokens"] = min(int(nested["max_image_tokens"]), IMAGE_MAX_TOKENS)
        out["images_kwargs"] = nested
    return out


class Glm5NextProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # Empty means text-only: the chat layer refuses images with a 400 before anything reaches the engine.
        # The serve command lowers the per-prompt limit further (--limit-mm-per-prompt).
        return {"image": 8} if VISION_ENABLED else {}

    def get_hf_processor(self, **kwargs: object):
        proc = super().get_hf_processor(**kwargs)
        # On the instance, so the cap also holds for the profile run and the dummy inputs.
        ip = getattr(proc, "image_processor", None)
        if ip is not None:
            ip.max_image_tokens = min(int(ip.max_image_tokens), IMAGE_MAX_TOKENS)
        return proc

    def get_image_processor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).image_processor

    def get_image_size_with_most_features(self) -> ImageSize:
        """The largest square canvas the token cap allows, for profiling."""
        ip = self.get_image_processor()
        max_tok = int(getattr(ip, "max_image_tokens", 8000))
        side = int(max_tok**0.5) * int(ip.patch_size) * int(ip.merge_size)
        return ImageSize(width=side, height=side)

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int]):
        if not VISION_ENABLED:
            return {}
        return {"image": int(getattr(self.get_image_processor(), "max_image_tokens", 8000))}


class Glm5NextDummyInputsBuilder(BaseDummyInputsBuilder[Glm5NextProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return self.info.get_hf_processor().image_token * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        w, h = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=w, height=h, num_images=mm_counts.get("image", 0), overrides=(mm_options or {}).get("image")
            )
        }


class Glm5NextMultiModalProcessor(BaseMultiModalProcessor[Glm5NextProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        return super()._call_hf_processor(
            prompt=prompt, mm_data=mm_data, mm_kwargs=clamp_request_kwargs(mm_kwargs), tok_kwargs=tok_kwargs
        )

    def _get_mm_fields_config(
        self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]
    ) -> Mapping[str, MultiModalFieldConfig]:
        # pixel_values is one flat patch sequence for the whole request: split it per image by patch count. Batching
        # it by first dim would hand image B's patches to image A with matching shapes and no error.
        grid = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        counts = grid.prod(-1) if grid.numel() else torch.empty((0,))
        return {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes("image", counts),
            "image_grid_thw": MultiModalFieldConfig.batched("image"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        merge_length = int(self.info.get_image_processor(**hf_processor_mm_kwargs).merge_size) ** 2

        def replacement(item_idx: int):
            grid_thw = out_mm_kwargs["image"][item_idx]["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            return [hf.image_token_id] * (int(grid_thw.prod()) // merge_length)

        # The chat template writes <|begin_of_image|><|image|><|end_of_image|>; only the middle token expands, and the
        # begin/end markers stay as the model was trained to see them.
        return [PromptReplacement(modality="image", target=hf.image_token, replacement=replacement)]
