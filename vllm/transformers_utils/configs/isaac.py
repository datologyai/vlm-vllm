# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Sequence

from transformers import Qwen3Config
from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig


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


class PixelShuffleSiglip2VisionConfig(Siglip2VisionConfig):
    """Vision configuration for Isaac with Pixel Shuffle support.

    Extends Siglip2VisionConfig with additional fields for pixel shuffle.
    """

    model_type = "pixel_shuffle_siglip2"
    base_config_key = "vision_config"

    def __init__(
        self,
        pixel_shuffle_scale_factor: int = 1,
        pixel_shuffle_factors: Sequence[int] | None = None,
        pixel_shuffle_factor_height: int | None = None,
        pixel_shuffle_factor_width: int | None = None,
        num_patches: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)

        factor_h, factor_w = _normalize_pixel_shuffle_factors(
            pixel_shuffle_scale_factor=pixel_shuffle_scale_factor,
            pixel_shuffle_factors=pixel_shuffle_factors,
            pixel_shuffle_factor_height=pixel_shuffle_factor_height,
            pixel_shuffle_factor_width=pixel_shuffle_factor_width,
        )

        # Keep scalar field for backward compatibility with existing checkpoints.
        self.pixel_shuffle_scale_factor = factor_h if factor_h == factor_w else 1
        self.pixel_shuffle_factor_height = factor_h
        self.pixel_shuffle_factor_width = factor_w
        self.pixel_shuffle_factors = [factor_h, factor_w]
        self.num_patches = num_patches


class IsaacConfig(Qwen3Config):
    """Configuration class for Isaac multimodal model."""

    model_type = "isaac"
    sub_configs = {
        "vision_config": PixelShuffleSiglip2VisionConfig,
        "text_config": Qwen3Config,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        vision_patch_size: int = 16,
        vision_max_num_patches: int = 256,
        vision_min_num_patches: int | None = None,
        pixel_shuffle_scale: int = 1,
        pixel_shuffle_factors: Sequence[int] | None = None,
        pixel_shuffle_factor_height: int | None = None,
        pixel_shuffle_factor_width: int | None = None,
        max_sequence_length: int = 16384,
        vision_token: str = "<image>",
        vision_attn_implementation: str | None = None,
        dynamic_image_size: bool = False,
        tile_size: int | None = None,
        min_num_tiles: int = 1,
        max_num_tiles: int = 12,
        use_thumbnail: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(text_config, dict):
            # from HF config
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            # For BC use all kwargs to init text config.
            self.text_config = self.sub_configs["text_config"](**kwargs)
        else:
            # from Qwen3Config
            self.text_config = text_config

        # EventStreamProcessor parameters (for backward compatibility)
        self.video_patch_size = vision_patch_size
        self.vision_max_num_patches = vision_max_num_patches
        self.vision_min_num_patches = vision_min_num_patches
        self.pixel_shuffle_scale = pixel_shuffle_scale
        self.dynamic_image_size = dynamic_image_size
        self.min_num_tiles = min_num_tiles
        self.max_num_tiles = max_num_tiles
        self.use_thumbnail = use_thumbnail

        # Processing parameters
        self.max_sequence_length = max_sequence_length
        self.vision_token = vision_token

        # Handle vision config - PixelShuffleSiglip2VisionConfig instance
        if isinstance(vision_config, dict):
            self.vision_config = PixelShuffleSiglip2VisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = PixelShuffleSiglip2VisionConfig()
        else:
            self.vision_config = vision_config

        factor_h, factor_w = _normalize_pixel_shuffle_factors(
            pixel_shuffle_scale_factor=getattr(
                self.vision_config, "pixel_shuffle_scale_factor", pixel_shuffle_scale
            ),
            pixel_shuffle_factors=getattr(
                self.vision_config, "pixel_shuffle_factors", pixel_shuffle_factors
            ),
            pixel_shuffle_factor_height=getattr(
                self.vision_config,
                "pixel_shuffle_factor_height",
                pixel_shuffle_factor_height,
            ),
            pixel_shuffle_factor_width=getattr(
                self.vision_config,
                "pixel_shuffle_factor_width",
                pixel_shuffle_factor_width,
            ),
        )
        self.pixel_shuffle_factor_height = factor_h
        self.pixel_shuffle_factor_width = factor_w
        self.pixel_shuffle_factors = [factor_h, factor_w]
        self.pixel_shuffle_scale = factor_h if factor_h == factor_w else 1

        # Ensure compatibility with pretrained checkpoints.
        self.vision_config.pixel_shuffle_scale_factor = (
            factor_h if factor_h == factor_w else 1
        )
        self.vision_config.pixel_shuffle_factor_height = factor_h
        self.vision_config.pixel_shuffle_factor_width = factor_w
        self.vision_config.pixel_shuffle_factors = [factor_h, factor_w]
        self.vision_config.num_patches = getattr(
            self.vision_config,
            "num_patches",
            vision_max_num_patches,
        )
        self.tile_size = (
            tile_size
            if tile_size is not None
            else int(getattr(self.vision_config, "image_size", 0) or 0)
        )
        self.vision_attn_implementation = vision_attn_implementation


__all__ = [
    "IsaacConfig",
    "PixelShuffleSiglip2VisionConfig",
]
