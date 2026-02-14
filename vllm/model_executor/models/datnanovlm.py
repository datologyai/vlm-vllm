# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import math
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Annotated, Any

import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.image_processing_utils import BatchFeature
from transformers.models.siglip.image_processing_siglip import SiglipImageProcessor
from transformers.tokenization_utils import TensorType
from typing_extensions import TypedDict, Unpack

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalFieldConfig
from vllm.multimodal.parse import ImageEmbeddingItems, ImageProcessorItems, ImageSize
from vllm.multimodal.processing import (
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .internvl import (
    calculate_internvl_targets,
    dynamic_preprocess_internvl,
    get_internvl_target_ratios,
)
from .isaac import (
    IsaacDummyInputsBuilder,
    IsaacForConditionalGeneration,
    IsaacMultiModalProcessor,
    IsaacProcessingInfo,
    IsaacProcessor,
    MultiModalDataItems,
    MultiModalKwargsItems,
    Siglip2VisionTransformer,
    _resolve_vision_token_id,
    create_cumulative_seq_lengths,
    extract_image_pil,
    patchify_vision,
    prepare_image_tensor,
)

VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
IMAGE_PAD_TOKEN = "<|image_pad|>"
IMAGE_PLACEHOLDER_TOKEN = "<image>"


def _resolve_datnano_pixel_shuffle_factors(
    value: int | Sequence[int] | None = None,
    *,
    config: object | None = None,
) -> tuple[int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(
                "pixel shuffle sequence must contain exactly two integers: "
                "[height_factor, width_factor]"
            )
        return int(value[0]), int(value[1])

    if isinstance(value, int):
        return int(value), int(value)

    if config is None:
        return 1, 1

    factors = getattr(config, "pixel_shuffle_factors", None)
    if isinstance(factors, Sequence) and len(factors) == 2:
        return int(factors[0]), int(factors[1])

    factor_h = getattr(config, "pixel_shuffle_factor_height", None)
    factor_w = getattr(config, "pixel_shuffle_factor_width", None)
    if factor_h is not None or factor_w is not None:
        if factor_h is None:
            factor_h = factor_w
        if factor_w is None:
            factor_w = factor_h
        return int(factor_h), int(factor_w)

    scale = int(
        getattr(
            config,
            "pixel_shuffle_scale_factor",
            getattr(config, "pixel_shuffle_scale", 1),
        )
    )
    return scale, scale


def _create_pixel_shuffle_index_map_datnano(
    seq_sizes: torch.Tensor,
    token_grids: torch.Tensor,
    scale_factor: int | tuple[int, int] = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = seq_sizes.device

    if isinstance(scale_factor, int):
        rh = int(scale_factor)
        rw = int(scale_factor)
    else:
        rh = int(scale_factor[0])
        rw = int(scale_factor[1])

    if rh < 2 and rw < 2:
        raise ValueError("`scale_factor` must be >= 2")

    if not torch.compiler.is_compiling() and not (
        (token_grids[:, 0] % rh == 0).all() and (token_grids[:, 1] % rw == 0).all()
    ):
        raise AssertionError(
            "Every (H,W) in `token_grids` must be divisible by "
            f"scale_factor=({rh}, {rw}), got {token_grids.tolist()}"
        )

    gather_chunks: list[torch.Tensor] = []
    tok_offset = 0

    for seq_len, (h, w) in zip(seq_sizes.tolist(), token_grids.tolist(), strict=False):
        grid = torch.arange(seq_len, device=device, dtype=torch.int64) + tok_offset
        grid = grid.view(h, w)

        grid = grid.view(h, w // rw, rw)
        grid = grid.view(h // rh, rh, w // rw, rw)
        grid = grid.permute(0, 2, 1, 3).contiguous()
        gather_chunks.append(grid.reshape(-1, rh * rw))
        tok_offset += seq_len

    return torch.cat(gather_chunks, dim=0)


def _pixel_shuffle_varlen_datnano(
    x: torch.Tensor,
    token_grids: torch.Tensor,
    scale_factor: int | tuple[int, int] = 1,
) -> torch.Tensor:
    keep_batch_dim = x.dim() == 3
    if keep_batch_dim:
        if x.size(0) != 1:
            raise AssertionError("Packed sequence is expected to have batch_size == 1")
        x_ = x.squeeze(0)
    else:
        x_ = x

    embed_dim = x_.size(-1)
    if isinstance(scale_factor, int):
        rh = int(scale_factor)
        rw = int(scale_factor)
    else:
        rh = int(scale_factor[0])
        rw = int(scale_factor[1])

    seq_sizes = torch.prod(token_grids, dim=-1)
    gather_idx = _create_pixel_shuffle_index_map_datnano(
        seq_sizes=seq_sizes,
        token_grids=token_grids,
        scale_factor=(rh, rw),
        device=x_.device,
    )

    gathered = x_[gather_idx]
    out = gathered.reshape(gathered.size(0), embed_dim * rh * rw)

    if keep_batch_dim:
        out = out.unsqueeze(0)
    return out


def _get_image_size_for_max_num_patches_datnano(
    image_height: int,
    image_width: int,
    patch_size: int,
    max_num_patches: int,
    min_num_patches: int | None = None,
    eps: float = 1e-5,
    pixel_shuffle_scale: int | tuple[int, int] = 1,
) -> tuple[int, int]:
    if isinstance(pixel_shuffle_scale, int):
        shuffle_h = int(pixel_shuffle_scale)
        shuffle_w = int(pixel_shuffle_scale)
    else:
        shuffle_h = int(pixel_shuffle_scale[0])
        shuffle_w = int(pixel_shuffle_scale[1])

    def get_scaled_image_size(scale: float, original_size: int, divisor: int) -> int:
        scaled_size = scale * original_size
        scaled_size = math.ceil(scaled_size / divisor) * divisor
        scaled_size = max(divisor, scaled_size)
        return int(scaled_size)

    divisor_h = patch_size * shuffle_h
    divisor_w = patch_size * shuffle_w
    adjusted_height = math.ceil(image_height / divisor_h) * divisor_h
    adjusted_height = max(divisor_h, adjusted_height)
    adjusted_width = math.ceil(image_width / divisor_w) * divisor_w
    adjusted_width = max(divisor_w, adjusted_width)

    num_patches = (adjusted_height / patch_size) * (adjusted_width / patch_size)

    if min_num_patches is not None and num_patches < min_num_patches:
        scale_min, scale_max = 1.0, 100.0
        while (scale_max - scale_min) >= eps:
            scale = (scale_min + scale_max) / 2
            target_height = get_scaled_image_size(scale, image_height, divisor_h)
            target_width = get_scaled_image_size(scale, image_width, divisor_w)
            num_patches = (target_height / patch_size) * (target_width / patch_size)
            if num_patches >= min_num_patches:
                scale_max = scale
            else:
                scale_min = scale
        scale = scale_max
        target_height = get_scaled_image_size(scale, image_height, divisor_h)
        target_width = get_scaled_image_size(scale, image_width, divisor_w)
        return target_height, target_width

    if num_patches <= max_num_patches:
        return adjusted_height, adjusted_width

    scale_min, scale_max = eps / 10, 1.0
    while (scale_max - scale_min) >= eps:
        scale = (scale_min + scale_max) / 2
        target_height = get_scaled_image_size(scale, image_height, divisor_h)
        target_width = get_scaled_image_size(scale, image_width, divisor_w)
        num_patches = (target_height / patch_size) * (target_width / patch_size)
        if num_patches <= max_num_patches:
            scale_min = scale
        else:
            scale_max = scale

    scale = scale_min
    target_height = get_scaled_image_size(scale, image_height, divisor_h)
    target_width = get_scaled_image_size(scale, image_width, divisor_w)
    return target_height, target_width


def _process_vision_for_patches_datnano(
    images: torch.Tensor,
    patch_size: int,
    max_num_patches: int,
    min_num_patches: int | None = None,
    pixel_shuffle_scale: int | tuple[int, int] = 1,
) -> tuple[torch.Tensor, list[int]]:
    if images.dim() == 3:
        images = images.unsqueeze(0)

    images = images.permute(0, 3, 1, 2)

    _, _, orig_height, orig_width = images.shape
    target_height, target_width = _get_image_size_for_max_num_patches_datnano(
        orig_height,
        orig_width,
        patch_size,
        max_num_patches,
        min_num_patches=min_num_patches,
        pixel_shuffle_scale=pixel_shuffle_scale,
    )

    images = F.interpolate(
        images,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    images = images.permute(0, 2, 3, 1)
    images = prepare_image_tensor(images)
    patches = patchify_vision(images, patch_size=patch_size)

    n_images, h_patches, w_patches, _ = patches.shape
    if isinstance(pixel_shuffle_scale, int):
        shuffle_h = int(pixel_shuffle_scale)
        shuffle_w = int(pixel_shuffle_scale)
    else:
        shuffle_h = int(pixel_shuffle_scale[0])
        shuffle_w = int(pixel_shuffle_scale[1])

    dims_virtual = (
        [1, h_patches, w_patches]
        if shuffle_h == 1 and shuffle_w == 1
        else [n_images, h_patches // shuffle_h, w_patches // shuffle_w]
    )
    return patches, dims_virtual


class DatNanoVLMImageProcessorKwargs(TypedDict, total=False):
    patch_size: int
    max_num_patches: int
    min_num_patches: int
    pixel_shuffle_scale: int
    pixel_shuffle_factors: tuple[int, int]
    dynamic_image_size: bool
    tile_size: int
    min_num_tiles: int
    max_num_tiles: int
    use_thumbnail: bool


class DatNanoVLMImageProcessor:
    patch_size = 16
    max_num_patches = 6144
    min_num_patches = 256
    pixel_shuffle_scale = 2
    dynamic_image_size = False
    tile_size = 384
    min_num_tiles = 1
    max_num_tiles = 12
    use_thumbnail = True

    valid_kwargs = DatNanoVLMImageProcessorKwargs  # type: ignore[assignment]
    model_input_names = ["pixel_values", "image_grid_thw", "image_num_tiles"]

    def __init__(self, kwargs):
        self.patch_size = kwargs.pop("patch_size", self.patch_size)
        self.vision_max_num_patches = kwargs.pop(
            "vision_max_num_patches", self.max_num_patches
        )
        self.vision_min_num_patches = kwargs.pop(
            "vision_min_num_patches", self.min_num_patches
        )
        pixel_shuffle_value = kwargs.pop("pixel_shuffle_factors", None)
        if pixel_shuffle_value is None:
            pixel_shuffle_value = kwargs.pop(
                "pixel_shuffle_scale", self.pixel_shuffle_scale
            )
        self.pixel_shuffle_factors = _resolve_datnano_pixel_shuffle_factors(
            pixel_shuffle_value
        )
        self.dynamic_image_size = kwargs.pop(
            "dynamic_image_size", self.dynamic_image_size
        )
        self.tile_size = kwargs.pop("tile_size", self.tile_size)
        self.min_num_tiles = kwargs.pop("min_num_tiles", self.min_num_tiles)
        self.max_num_tiles = kwargs.pop("max_num_tiles", self.max_num_tiles)
        self.use_thumbnail = kwargs.pop("use_thumbnail", self.use_thumbnail)

        # Use HF SiglipImageProcessor codepath for resize/rescale/normalize parity.
        self._hf_siglip = SiglipImageProcessor(
            do_resize=True,
            size={"height": int(self.tile_size), "width": int(self.tile_size)},
            resample=2,  # PIL bilinear
            do_rescale=True,
            rescale_factor=1.0 / 255.0,
            do_normalize=True,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )

    def _resolve_tiles(self, image: PIL.Image.Image) -> list[PIL.Image.Image]:
        if not self.dynamic_image_size:
            return [image]

        target_ratios = get_internvl_target_ratios(
            self.min_num_tiles, self.max_num_tiles
        )
        return dynamic_preprocess_internvl(
            image=image,
            target_ratios=target_ratios,
            image_size=self.tile_size,
            use_thumbnail=self.use_thumbnail,
        )

    def preprocess(
        self,
        images: list[torch.Tensor],
        return_tensors: str | TensorType | None,
        **kwargs: Unpack[DatNanoVLMImageProcessorKwargs],
    ) -> BatchFeature:
        all_pixel_values: list[torch.Tensor] = []
        all_image_grids: list[torch.Tensor] = []
        image_num_tiles: list[int] = []

        for image in images:
            tiles = self._resolve_tiles(image)
            image_num_tiles.append(len(tiles))

            for tile in tiles:
                # HF SigLIP preprocessing (resize->tile_size, rescale 1/255, normalize mean/std=0.5).
                hf_out = self._hf_siglip(images=tile, return_tensors="pt")
                pixel = hf_out["pixel_values"][0]  # [3,H,W]

                # Extract valid (floor) non-overlapping patches (conv/valid semantics).
                ps = int(self.patch_size)
                patches = (
                    pixel.unfold(1, ps, ps)
                    .unfold(2, ps, ps)
                    .permute(1, 2, 0, 3, 4)
                    .contiguous()
                )
                hp, wp = int(patches.shape[0]), int(patches.shape[1])
                pixel_values = patches.reshape(hp * wp, 3 * ps * ps)

                image_grid_thw = torch.tensor([1, hp, wp]).unsqueeze(0)
                all_pixel_values.append(pixel_values)
                all_image_grids.append(image_grid_thw)

        if all_pixel_values:
            final_pixel_values = torch.cat(all_pixel_values, dim=0)
            final_image_grids = torch.cat(all_image_grids, dim=0)
        else:
            final_pixel_values = torch.empty(0, 0)
            final_image_grids = torch.empty(0, 3)

        return BatchFeature(
            data={
                "pixel_values": final_pixel_values,
                "image_grid_thw": final_image_grids,
                "image_num_tiles": torch.tensor(image_num_tiles, dtype=torch.int32),
            },
            tensor_type=return_tensors,
        )



class DatNanoVLMProcessor(IsaacProcessor):
    """Processor wrapper with training-aligned conversation/tag normalization."""

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        self.image_token = kwargs.pop("image_token", IMAGE_PLACEHOLDER_TOKEN)
        self.image_processor = image_processor or DatNanoVLMImageProcessor(kwargs)
        self.tokenizer = tokenizer

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
        result: dict[str, Any] = {}

        if images is not None:
            image_inputs = self.image_processor.preprocess(images, **kwargs)
            image_grid_thw = image_inputs["image_grid_thw"]
            image_num_tiles = image_inputs["image_num_tiles"]
            result.update(image_inputs)

            if text is not None:
                if not isinstance(text, list):
                    text = [text]

                text = text.copy()
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

                    text[i] = text[i].replace("<|placeholder|>", IMAGE_PAD_TOKEN)

        if text is not None:
            result.update(self.tokenizer(text, **kwargs))

        return BatchFeature(result)


class DatNanoVLMProcessingInfo(IsaacProcessingInfo):
    """DatNanoVLM processing contract aligned with training data pipeline."""

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs):
        hf_config = self.get_hf_config()
        factor_h, factor_w = _resolve_datnano_pixel_shuffle_factors(config=hf_config)

        processor_kwargs = {
            "image_token": IMAGE_PLACEHOLDER_TOKEN,
            "patch_size": int(hf_config.video_patch_size),
            "vision_max_num_patches": int(hf_config.vision_max_num_patches),
            "vision_min_num_patches": hf_config.vision_min_num_patches,
            "pixel_shuffle_factors": (factor_h, factor_w),
            "dynamic_image_size": bool(getattr(hf_config, "dynamic_image_size", False)),
            "tile_size": int(getattr(hf_config, "tile_size", 384)),
            "min_num_tiles": int(getattr(hf_config, "min_num_tiles", 1)),
            "max_num_tiles": int(getattr(hf_config, "max_num_tiles", 12)),
            "use_thumbnail": bool(getattr(hf_config, "use_thumbnail", True)),
        }
        processor_kwargs.update(kwargs)
        return self.ctx.get_hf_processor(DatNanoVLMProcessor, **processor_kwargs)

    def get_image_size_with_most_features(self) -> ImageSize:
        hf_config = self.get_hf_config()
        if getattr(hf_config, "dynamic_image_size", False):
            ratios = get_internvl_target_ratios(
                int(getattr(hf_config, "min_num_tiles", 1)),
                int(getattr(hf_config, "max_num_tiles", 12)),
            )
            width_ratio, height_ratio = max(ratios, key=lambda x: x[0] * x[1])
            tile_size = int(getattr(hf_config, "tile_size", 384))
            return ImageSize(
                width=tile_size * width_ratio,
                height=tile_size * height_ratio,
            )

        factor_h, factor_w = _resolve_datnano_pixel_shuffle_factors(config=hf_config)
        target_height, target_width = _get_image_size_for_max_num_patches_datnano(
            9999999,
            9999999,
            int(hf_config.video_patch_size),
            int(hf_config.vision_max_num_patches),
            min_num_patches=hf_config.vision_min_num_patches,
            pixel_shuffle_scale=(factor_h, factor_w),
        )
        return ImageSize(width=target_width, height=target_height)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor: DatNanoVLMImageProcessor | None = None,
    ) -> int:
        if image_processor is None:
            image_processor = self.get_image_processor()

        factor_h, factor_w = image_processor.pixel_shuffle_factors
        merge_length = factor_h * factor_w
        if image_processor.dynamic_image_size:
            target_ratios = get_internvl_target_ratios(
                image_processor.min_num_tiles,
                image_processor.max_num_tiles,
            )
            num_tiles, _, _ = calculate_internvl_targets(
                orig_width=image_width,
                orig_height=image_height,
                target_ratios=target_ratios,
                image_size=image_processor.tile_size,
                use_thumbnail=image_processor.use_thumbnail,
            )
            patches_per_tile = (
                image_processor.tile_size // image_processor.patch_size
            ) ** 2
            return (patches_per_tile // merge_length) * num_tiles

        target_height, target_width = _get_image_size_for_max_num_patches_datnano(
            image_height=image_height,
            image_width=image_width,
            patch_size=image_processor.patch_size,
            max_num_patches=image_processor.vision_max_num_patches,
            min_num_patches=image_processor.vision_min_num_patches,
            pixel_shuffle_scale=(factor_h, factor_w),
        )
        num_patches = (target_height // image_processor.patch_size) * (
            target_width // image_processor.patch_size
        )
        return num_patches // merge_length

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        target_size = self.get_image_size_with_most_features()
        num_vision_tokens = self.get_num_image_tokens(
            image_width=target_size.width,
            image_height=target_size.height,
        )
        return {"image": num_vision_tokens}


class DatNanoVLMImagePixelInputs(TensorSchema):
    pixel_values: Annotated[
        torch.Tensor,
        TensorShape("np", "d"),
    ]

    image_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("nt", 3),
    ]

    image_num_tiles: Annotated[
        torch.Tensor,
        TensorShape("ni"),
    ]


class DatNanoVLMMultiModalProcessor(IsaacMultiModalProcessor):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        image_num_tiles = hf_inputs.get(
            "image_num_tiles", torch.empty((0,), dtype=torch.int32)
        )

        if len(image_num_tiles) > 0:
            pixel_sizes: list[int] = []
            grid_sizes: list[int] = []
            tile_offset = 0
            for num_tiles in image_num_tiles.tolist():
                tiles_for_image = int(num_tiles)
                grid_chunk = image_grid_thw[tile_offset : tile_offset + tiles_for_image]
                pixel_sizes.append(int(grid_chunk.prod(-1).sum()))
                grid_sizes.append(tiles_for_image)
                tile_offset += tiles_for_image
            pixel_sizes_tensor = torch.tensor(
                pixel_sizes, dtype=torch.int32, device=image_grid_thw.device
            )
            grid_sizes_tensor = torch.tensor(
                grid_sizes, dtype=torch.int32, device=image_grid_thw.device
            )

            return {
                "pixel_values": MultiModalFieldConfig.flat_from_sizes(
                    "image", pixel_sizes_tensor
                ),
                "image_grid_thw": MultiModalFieldConfig.flat_from_sizes(
                    "image", grid_sizes_tensor
                ),
                "image_num_tiles": MultiModalFieldConfig.batched("image"),
            }

        image_grid_sizes = image_grid_thw.prod(-1)
        return {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_sizes
            ),
            "image_grid_thw": MultiModalFieldConfig.batched("image"),
            "image_num_tiles": MultiModalFieldConfig.batched("image"),
        }

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


class DatNanoVLMVisionTransformer(Siglip2VisionTransformer):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.pixel_shuffle_factors = _resolve_datnano_pixel_shuffle_factors(config=config)

    def forward(
        self,
        packed_seq_patches: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_patches, token_grids = packed_seq_patches
        seq_sizes = torch.prod(token_grids, dim=-1)

        hidden_states = self.embeddings((seq_patches, seq_sizes, token_grids))
        hidden_states = hidden_states.unsqueeze(0)

        cu_seqlens, max_seqlen = create_cumulative_seq_lengths(
            seq_sizes, hidden_states.device
        )
        hidden_states = self.encoder(
            inputs_embeds=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = self.post_layernorm(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                if (
                    name.endswith("embeddings.patch_embedding.weight")
                    and loaded_weight.ndim == 4
                ):
                    loaded_weight = loaded_weight.flatten(1)
                if (
                    name.endswith("embeddings.position_embedding.weight")
                    and loaded_weight.ndim == 2
                    and loaded_weight.shape != param.shape
                ):
                    old_num_patches, hidden_size = loaded_weight.shape
                    new_num_patches, new_hidden_size = param.shape
                    if hidden_size != new_hidden_size:
                        raise ValueError(
                            "Position embedding hidden size mismatch: "
                            f"{hidden_size} vs {new_hidden_size}"
                        )
                    old_grid = int(math.isqrt(old_num_patches))
                    new_grid = int(math.isqrt(new_num_patches))
                    if old_grid * old_grid != old_num_patches:
                        raise ValueError(
                            "Old position embeddings are not square: "
                            f"{old_num_patches}"
                        )
                    if new_grid * new_grid != new_num_patches:
                        raise ValueError(
                            "New position embeddings are not square: "
                            f"{new_num_patches}"
                        )
                    loaded_weight = (
                        F.interpolate(
                            loaded_weight.view(old_grid, old_grid, hidden_size)
                            .permute(2, 0, 1)
                            .unsqueeze(0),
                            size=(new_grid, new_grid),
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze(0)
                        .permute(1, 2, 0)
                        .reshape(new_num_patches, hidden_size)
                        .to(dtype=param.dtype)
                    )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


class DatNanoVLMVisionEmbedding(nn.Module):
    def __init__(
        self,
        vision_cfg,
        hidden_dim: int,
        output_dim: int,
        projector_config: Mapping[str, Any] | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.transformer = DatNanoVLMVisionTransformer(
            vision_cfg,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "0"),
        )
        projector_config = dict(projector_config or {})
        projector_layers = int(projector_config.get("layers", 2))
        projector_hidden_dim = int(projector_config.get("hidden_dim", 4 * hidden_dim))
        projector_activation = str(projector_config.get("activation", "silu")).lower()

        layers: list[nn.Module] = []
        if projector_layers <= 1:
            layers.append(
                ReplicatedLinear(
                    hidden_dim,
                    output_dim,
                    bias=True,
                    return_bias=False,
                )
            )
            layers.append(nn.LayerNorm(output_dim))
        else:
            layers.append(
                ReplicatedLinear(
                    hidden_dim,
                    projector_hidden_dim,
                    bias=True,
                    return_bias=False,
                )
            )
            layers.append(nn.LayerNorm(projector_hidden_dim))
            layers.append(self._get_projector_activation(projector_activation))
            for _ in range(projector_layers - 2):
                layers.append(
                    ReplicatedLinear(
                        projector_hidden_dim,
                        projector_hidden_dim,
                        bias=True,
                        return_bias=False,
                    )
                )
                layers.append(nn.LayerNorm(projector_hidden_dim))
                layers.append(self._get_projector_activation(projector_activation))
            layers.append(
                ReplicatedLinear(
                    projector_hidden_dim,
                    output_dim,
                    bias=True,
                    return_bias=False,
                )
            )
            layers.append(nn.LayerNorm(output_dim))

        self.layers = nn.Sequential(*layers)
        self.linear_fc1 = self.layers[0]
        self.linear_fc2 = self.layers[-2] if len(self.layers) > 1 else self.layers[0]

    @staticmethod
    def _get_projector_activation(activation: str) -> nn.Module:
        if activation == "relu":
            return nn.ReLU()
        if activation == "silu":
            return nn.SiLU()
        if activation == "tanh":
            return nn.Tanh()
        return nn.GELU()

    def forward(
        self, packed_seq_patches: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden_states = self.transformer(packed_seq_patches)
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

        config = self.config
        vision_cfg = config.vision_config
        factor_h, factor_w = _resolve_datnano_pixel_shuffle_factors(config=vision_cfg)
        merge_length = factor_h * factor_w
        hidden_dim = vision_cfg.hidden_size * merge_length

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_embedding = DatNanoVLMVisionEmbedding(
                vision_cfg=vision_cfg,
                hidden_dim=hidden_dim,
                output_dim=config.hidden_size,
                projector_config=getattr(config, "projector_config", None),
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "vision_embedding"),
            )

        image_pad_id = _resolve_vision_token_id(
            vllm_config.model_config, IMAGE_PAD_TOKEN
        )
        self.vision_token_id = image_pad_id
        self.config.image_token_id = image_pad_id

    def iter_mm_grid_hw(
        self, input_tokens: list[int], mm_features: list[MultiModalFeatureSpec]
    ) -> Iterator[tuple[int, int, int]]:
        factor_h, factor_w = _resolve_datnano_pixel_shuffle_factors(
            config=self.config.vision_config
        )
        for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            offset = mm_feature.mm_position.offset
            if mm_feature.modality == "image":
                grid_thw = mm_feature.data["image_grid_thw"].data
                if isinstance(grid_thw, torch.Tensor):
                    if grid_thw.ndim == 1:
                        grid_thw = grid_thw.unsqueeze(0)

                    tile_offset = offset
                    for t, h, w in grid_thw.tolist():
                        assert t == 1, f"Image must have 1 frame, got {t}"
                        llm_h = h // factor_h
                        llm_w = w // factor_w
                        yield tile_offset, llm_h, llm_w
                        tile_offset += llm_h * llm_w
                else:
                    raise TypeError("image_grid_thw must be a tensor")
            else:
                raise ValueError(f"Unsupported modality: {mm_feature.modality}")

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> DatNanoVLMImagePixelInputs | None:
        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")
        image_num_tiles = kwargs.get("image_num_tiles")
        if pixel_values is None or image_grid_thw is None:
            return None

        if image_num_tiles is None:
            image_num_tiles = torch.ones(
                image_grid_thw.shape[0], dtype=torch.int32, device=image_grid_thw.device
            )

        total_tiles = int(image_num_tiles.sum().item())
        if total_tiles != int(image_grid_thw.shape[0]):
            raise ValueError(
                "image_num_tiles must sum to image_grid_thw rows: "
                f"sum(image_num_tiles)={total_tiles}, "
                f"image_grid_thw_rows={int(image_grid_thw.shape[0])}"
            )

        return DatNanoVLMImagePixelInputs(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_num_tiles=image_num_tiles,
        )

    def _process_image_input(
        self,
        image_input: DatNanoVLMImagePixelInputs,
    ) -> tuple[torch.Tensor, ...]:
        pixel_values = image_input["pixel_values"]
        image_grid_thw = image_input["image_grid_thw"]
        image_num_tiles = image_input["image_num_tiles"]
        if pixel_values.numel() == 0:
            return ()

        device = next(self.language_model.parameters()).device
        dtype = self.vision_embedding.linear_fc1.weight.dtype
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        spatial_grids = image_grid_thw[:, 1:3].to(device, dtype=torch.int32)

        vision_embeddings = self.vision_embedding((pixel_values, spatial_grids))
        factor_h, factor_w = _resolve_datnano_pixel_shuffle_factors(
            config=self.config.vision_config
        )
        tile_feature_sizes = (spatial_grids.prod(-1) // (factor_h * factor_w)).tolist()
        tile_embeddings = vision_embeddings.split(tile_feature_sizes)

        grouped_embeddings: list[torch.Tensor] = []
        tile_offset = 0
        for num_tiles in image_num_tiles.tolist():
            num_tiles = int(num_tiles)
            grouped_embeddings.append(
                torch.cat(
                    list(tile_embeddings[tile_offset : tile_offset + num_tiles]), dim=0
                )
            )
            tile_offset += num_tiles

        return tuple(grouped_embeddings)
