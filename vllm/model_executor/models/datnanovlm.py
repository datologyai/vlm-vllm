# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import ImageEmbeddingItems, ImageProcessorItems
from vllm.multimodal.processing import PromptReplacement, PromptUpdate, PromptUpdateDetails

from .isaac import (
    IsaacDummyInputsBuilder,
    IsaacForConditionalGeneration,
    IsaacMultiModalProcessor,
    IsaacProcessingInfo,
    MultiModalDataItems,
    MultiModalKwargsItems,
    _resolve_vision_token_id,
)

VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
IMAGE_PAD_TOKEN = "<|image_pad|>"
IMAGE_PLACEHOLDER_TOKEN = "<image>"


class DatNanoVLMProcessingInfo(IsaacProcessingInfo):
    """DatNanoVLM processing contract aligned with training data pipeline.

    External placeholder remains `<image>`, but expanded prompt tokens are
    `<|vision_start|><|image_pad|>*N<|vision_end|>`.
    """

    def get_hf_processor(self, **kwargs):
        # Keep external contract stable for prompts/messages.
        kwargs["image_token"] = IMAGE_PLACEHOLDER_TOKEN
        return super().get_hf_processor(**kwargs)


class DatNanoVLMMultiModalProcessor(IsaacMultiModalProcessor):
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)

        def get_replacement_datnanovlm(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )
            if isinstance(images, ImageEmbeddingItems):
                feature_size = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                feature_size = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    image_processor=image_processor,
                )

            repl_full = (
                VISION_START_TOKEN + (IMAGE_PAD_TOKEN * feature_size) + VISION_END_TOKEN
            )
            return PromptUpdateDetails.select_text(repl_full, IMAGE_PAD_TOKEN)

        return [
            PromptReplacement(
                modality="image",
                target=IMAGE_PLACEHOLDER_TOKEN,
                replacement=get_replacement_datnanovlm,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    DatNanoVLMMultiModalProcessor,
    info=DatNanoVLMProcessingInfo,
    dummy_inputs=IsaacDummyInputsBuilder,
)
class DatNanoVLMForConditionalGeneration(IsaacForConditionalGeneration):
    """DatNanoVLM native architecture entrypoint."""

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_PLACEHOLDER_TOKEN
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Vision embeddings must be inserted at <|image_pad|> token positions.
        image_pad_id = _resolve_vision_token_id(
            vllm_config.model_config, IMAGE_PAD_TOKEN
        )
        self.vision_token_id = image_pad_id
        self.config.image_token_id = image_pad_id
