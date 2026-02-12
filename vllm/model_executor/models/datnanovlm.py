# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from transformers.image_processing_utils import BatchFeature

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import ImageEmbeddingItems, ImageProcessorItems
from vllm.multimodal.processing import (
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

from vllm.model_executor.models.utils import WeightsMapper

from .isaac import (
    IsaacDummyInputsBuilder,
    IsaacForConditionalGeneration,
    IsaacMultiModalProcessor,
    IsaacProcessingInfo,
    IsaacProcessor,
    MultiModalDataItems,
    MultiModalKwargsItems,
    _resolve_vision_token_id,
)

VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
IMAGE_PAD_TOKEN = "<|image_pad|>"
IMAGE_PLACEHOLDER_TOKEN = "<image>"


class DatNanoVLMProcessor(IsaacProcessor):
    """Processor wrapper with training-aligned conversation/tag normalization."""

    @staticmethod
    def _is_image_item(content_item: dict[str, Any]) -> bool:
        item_type = content_item.get("type")
        return item_type in {"image", "image_url", "input_image"}

    @staticmethod
    def _is_text_item(content_item: dict[str, Any]) -> bool:
        item_type = content_item.get("type")
        return item_type in {"text", "input_text"}

    def _to_text_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, str]], int]:
        processed: list[dict[str, str]] = []
        total_images = 0

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if isinstance(content, list):
                text_parts: list[str] = []
                for item in content:
                    if self._is_text_item(item):
                        text_parts.append(item.get("text", ""))
                    elif self._is_image_item(item):
                        text_parts.append(IMAGE_PLACEHOLDER_TOKEN)
                        total_images += 1
                processed.append({"role": role, "content": "".join(text_parts)})
                continue

                # no break
            text = str(content)
            processed.append({"role": role, "content": text})

        return processed, total_images

    @staticmethod
    def _count_placeholders(messages: list[dict[str, str]]) -> int:
        return sum(
            msg.get("content", "").count(IMAGE_PLACEHOLDER_TOKEN) for msg in messages
        )

    @staticmethod
    def _strip_all_placeholders(messages: list[dict[str, str]]) -> None:
        for msg in messages:
            msg["content"] = msg.get("content", "").replace(IMAGE_PLACEHOLDER_TOKEN, "")

    @staticmethod
    def _inject_placeholders_first_user(
        messages: list[dict[str, str]], num_images: int
    ) -> bool:
        prefix = (IMAGE_PLACEHOLDER_TOKEN + " ") * num_images
        for msg in messages:
            if msg.get("role") in {"user", "human"}:
                msg["content"] = prefix + msg.get("content", "")
                return True
        return False

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> Any:
        processed_messages, images_from_typed_content = self._to_text_messages(messages)
        placeholder_count = self._count_placeholders(processed_messages)

        if (
            images_from_typed_content > 0
            and placeholder_count != images_from_typed_content
        ):
            self._strip_all_placeholders(processed_messages)
            inserted = self._inject_placeholders_first_user(
                processed_messages, images_from_typed_content
            )
            if not inserted:
                raise ValueError("No user message found to insert image placeholders")

        return self.tokenizer.apply_chat_template(
            processed_messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    def __call__(self, text=None, images=None, **kwargs):
        """Call HF processor and expand `<image>` to training-aligned tokens.

        Isaac expands `<image>` to `<|image_pad|>*N`.
        DatNanoVLM expects `<|vision_start|><|image_pad|>*N<|vision_end|>`.
        """
        result: dict[str, Any] = {}

        if images is not None:
            image_inputs = self.image_processor.preprocess(images, **kwargs)
            image_grid_thw = image_inputs["image_grid_thw"]
            image_num_tiles = image_inputs["image_num_tiles"]
            result.update(image_inputs)

            if text is not None:
                if not isinstance(text, list):
                    text = [text]

                text = text.copy()  # below lines change text in-place
                factor_h, factor_w = self.image_processor.pixel_shuffle_factors
                merge_length = factor_h * factor_w
                tile_index = 0
                source_image_index = 0

                for i in range(len(text)):
                    while self.image_token in text[i]:
                        num_tiles = int(image_num_tiles[source_image_index])
                        total_tokens = 0
                        for _ in range(num_tiles):
                            total_tokens += int(image_grid_thw[tile_index].prod()) // (
                                merge_length
                            )
                            tile_index += 1

                        text[i] = text[i].replace(
                            self.image_token,
                            VISION_START_TOKEN
                            + ("<|placeholder|>" * total_tokens)
                            + VISION_END_TOKEN,
                            1,
                        )
                        source_image_index += 1

                    text[i] = text[i].replace("<|placeholder|>", IMAGE_PAD_TOKEN)

        if text is not None:
            result.update(self.tokenizer(text, **kwargs))

        return BatchFeature(result)


class DatNanoVLMProcessingInfo(IsaacProcessingInfo):
    """DatNanoVLM processing contract aligned with training data pipeline.

    External placeholder remains `<image>`, but expanded prompt tokens are
    `<|vision_start|><|image_pad|>*N<|vision_end|>`.
    """

    def get_hf_processor(self, **kwargs):
        hf_config = self.get_hf_config()
        factor_h, factor_w = (
            hf_config.pixel_shuffle_factor_height,
            hf_config.pixel_shuffle_factor_width,
        )

        processor_kwargs = {
            "image_token": IMAGE_PLACEHOLDER_TOKEN,
            "patch_size": hf_config.video_patch_size,
            "vision_max_num_patches": hf_config.vision_max_num_patches,
            "vision_min_num_patches": hf_config.vision_min_num_patches,
            "pixel_shuffle_factors": (factor_h, factor_w),
            "dynamic_image_size": hf_config.dynamic_image_size,
            "tile_size": hf_config.tile_size,
            "min_num_tiles": hf_config.min_num_tiles,
            "max_num_tiles": hf_config.max_num_tiles,
            "use_thumbnail": hf_config.use_thumbnail,
        }
        processor_kwargs.update(kwargs)
        return self.ctx.get_hf_processor(DatNanoVLMProcessor, **processor_kwargs)


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
                VISION_START_TOKEN
                + (IMAGE_PAD_TOKEN * feature_size)
                + VISION_END_TOKEN
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

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm_head.": "language_model.lm_head.",
            "model.text_model.lm_head.": "language_model.lm_head.",
            "model.text_model.": "language_model.model.",
            "model.vision_embedding.0": "vision_embedding.transformer",
            "model.vision_embedding.1": "vision_embedding.linear_fc1",
            "model.vision_embedding.2": "vision_embedding.act",
            "model.vision_embedding.3": "vision_embedding.linear_fc2",
            "model.vision_embedding.": "vision_embedding.",
            "model.lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
            "llm_backbone.lm_head.": "language_model.lm_head.",
            "llm_backbone.model.": "language_model.model.",
            "llm_backbone.": "language_model.",
            "vision_backbone.model.head.": None,
            "vision_backbone.vision_model.": None,
            "vision_backbone.model.": "vision_embedding.transformer.",
            "projector.layers.0.": "vision_embedding.layers.0.",
            "projector.layers.1.": "vision_embedding.layers.1.",
            "projector.layers.3.": "vision_embedding.layers.3.",
            "projector.layers.4.": "vision_embedding.layers.4.",
            "projector.layers.": None,
        }
    )

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
