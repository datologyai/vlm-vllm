# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Sequence

from transformers import AutoConfig

from vllm.transformers_utils.configs.isaac import IsaacConfig


def _normalize_pixel_shuffle_factors(
    *,
    pixel_shuffle_scale_factor: int | None,
    pixel_shuffle_factors: Sequence[int] | None,
    pixel_shuffle_factor_height: int | None,
    pixel_shuffle_factor_width: int | None,
) -> tuple[int, int]:
    if pixel_shuffle_factors is not None:
        if len(pixel_shuffle_factors) != 2:
            raise ValueError(
                "pixel_shuffle_factors must contain two values: "
                "[height_factor, width_factor]"
            )
        factor_h = int(pixel_shuffle_factors[0])
        factor_w = int(pixel_shuffle_factors[1])
    else:
        factor_h = (
            int(pixel_shuffle_factor_height)
            if pixel_shuffle_factor_height is not None
            else None
        )
        factor_w = (
            int(pixel_shuffle_factor_width)
            if pixel_shuffle_factor_width is not None
            else None
        )

        if factor_h is None and factor_w is None:
            scale = int(pixel_shuffle_scale_factor or 1)
            factor_h = scale
            factor_w = scale
        elif factor_h is None:
            factor_h = int(pixel_shuffle_scale_factor or factor_w or 1)
        elif factor_w is None:
            factor_w = int(pixel_shuffle_scale_factor or factor_h or 1)

    if factor_h < 1 or factor_w < 1:
        raise ValueError(
            "Pixel shuffle factors must be positive, "
            f"got height={factor_h}, width={factor_w}."
        )

    return factor_h, factor_w


class DatNanoVLMConfig(IsaacConfig):
    """Config adapter for datnanovlm SigLIP2/Qwen3 native-res checkpoints."""

    model_type = "datnanovlm"

    def __init__(
        self,
        llm_backbone: str | None = None,
        vision_backbone: str | None = None,
        architecture: str | None = None,
        model_name: str | None = None,
        projector_config: dict | None = None,
        image_tokens_per_image: int | None = None,
        special_token_ids: dict | None = None,
        text_config: dict | None = None,
        vision_config: dict | None = None,
        **kwargs,
    ):
        if text_config is None and llm_backbone:
            text_parent = AutoConfig.from_pretrained(llm_backbone).to_dict()
            text_config = text_parent.get("text_config", text_parent)

        if vision_config is None and vision_backbone:
            vision_parent = AutoConfig.from_pretrained(vision_backbone).to_dict()
            vision_config = vision_parent.get("vision_config", vision_parent)

        if text_config is None:
            text_config = {}
        if vision_config is None:
            vision_config = {}

        if "architectures" not in kwargs:
            kwargs["architectures"] = ["DatNanoVLMForConditionalGeneration"]

        patch_size = vision_config.get("patch_size")
        patch_size_int = int(patch_size) if patch_size is not None else None
        if patch_size_int is not None and "vision_patch_size" not in kwargs:
            kwargs["vision_patch_size"] = patch_size_int

        top_level_factor_h = kwargs.pop("pixel_shuffle_factor_height", None)
        top_level_factor_w = kwargs.pop("pixel_shuffle_factor_width", None)
        top_level_factors = kwargs.pop("pixel_shuffle_factors", None)
        top_level_scale = kwargs.pop("pixel_shuffle_scale", None)

        factor_h, factor_w = _normalize_pixel_shuffle_factors(
            pixel_shuffle_scale_factor=(
                int(top_level_scale)
                if top_level_scale is not None
                else (
                    vision_config.get("pixel_shuffle_scale_factor")
                    or vision_config.get("pixel_shuffle_scale")
                )
            ),
            pixel_shuffle_factors=(
                top_level_factors
                if top_level_factors is not None
                else vision_config.get("pixel_shuffle_factors")
            ),
            pixel_shuffle_factor_height=(
                int(top_level_factor_h)
                if top_level_factor_h is not None
                else vision_config.get("pixel_shuffle_factor_height")
            ),
            pixel_shuffle_factor_width=(
                int(top_level_factor_w)
                if top_level_factor_w is not None
                else vision_config.get("pixel_shuffle_factor_width")
            ),
        )

        dynamic_image_size = bool(kwargs.pop("dynamic_image_size", True))
        tile_size = kwargs.pop("tile_size", vision_config.get("image_size", 384))
        tile_size_int = int(tile_size) if tile_size is not None else 384
        min_num_tiles = int(kwargs.pop("min_num_tiles", 1))
        max_num_tiles = int(kwargs.pop("max_num_tiles", 12))
        use_thumbnail = bool(kwargs.pop("use_thumbnail", True))

        if "vision_max_num_patches" not in kwargs:
            if patch_size_int is not None:
                kwargs["vision_max_num_patches"] = (tile_size_int // patch_size_int) ** 2

        # IsaacConfig only accepts scalar pixel_shuffle_scale.
        kwargs["pixel_shuffle_scale"] = factor_h if factor_h == factor_w else 1

        super().__init__(text_config=text_config, vision_config=vision_config, **kwargs)

        self.llm_backbone = llm_backbone
        self.vision_backbone = vision_backbone
        self.architecture = architecture
        self.model_name = model_name
        self.projector_config = projector_config if projector_config is not None else {}
        self.special_token_ids = special_token_ids if special_token_ids is not None else {}

        self.pixel_shuffle_factor_height = factor_h
        self.pixel_shuffle_factor_width = factor_w
        self.pixel_shuffle_factors = [factor_h, factor_w]
        self.pixel_shuffle_scale = factor_h if factor_h == factor_w else 1
        self.dynamic_image_size = dynamic_image_size
        self.tile_size = tile_size_int
        self.min_num_tiles = min_num_tiles
        self.max_num_tiles = max_num_tiles
        self.use_thumbnail = use_thumbnail

        self.vision_config.pixel_shuffle_factor_height = factor_h
        self.vision_config.pixel_shuffle_factor_width = factor_w
        self.vision_config.pixel_shuffle_factors = [factor_h, factor_w]
        self.vision_config.pixel_shuffle_scale_factor = (
            factor_h if factor_h == factor_w else 1
        )
        self.vision_config.num_patches = getattr(
            self.vision_config,
            "num_patches",
            self.vision_max_num_patches,
        )

        merge_length = factor_h * factor_w
        self.image_tokens_per_image = (
            int(image_tokens_per_image)
            if image_tokens_per_image is not None
            else int(self.vision_max_num_patches) // merge_length
        )
