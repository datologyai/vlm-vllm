# SPDX-License-Identifier: Apache-2.0

import torch
from PIL import Image

from vllm.model_executor.models.datnanovlm import (
    DatNanoVLMImageProcessor,
    DatNanoVLMProcessor,
    IMAGE_PAD_TOKEN,
    VISION_END_TOKEN,
    VISION_START_TOKEN,
)
from vllm.model_executor.models.internvl import (
    calculate_internvl_targets,
    get_internvl_target_ratios,
)
from vllm.transformers_utils.configs.datnanovlm import DatNanoVLMConfig


class _DummyTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_text": text}

    def apply_chat_template(self, messages, **kwargs):
        return messages


def _make_nativeres_config() -> DatNanoVLMConfig:
    return DatNanoVLMConfig(
        image_tokens_per_image=243,
        vision_config={"patch_size": 14},
        pixel_shuffle_factor_height=3,
        pixel_shuffle_factor_width=1,
        dynamic_image_size=True,
        tile_size=384,
        min_num_tiles=1,
        max_num_tiles=12,
        use_thumbnail=True,
    )


def _make_processor_from_config(config: DatNanoVLMConfig) -> DatNanoVLMProcessor:
    vision_max_num_patches = int(config.vision_max_num_patches)
    image_processor = DatNanoVLMImageProcessor(
        {
            "patch_size": int(config.video_patch_size),
            "vision_max_num_patches": vision_max_num_patches,
            # Keep tile token count stable for strict token math checks.
            "vision_min_num_patches": vision_max_num_patches,
            "pixel_shuffle_factors": (
                int(config.pixel_shuffle_factor_height),
                int(config.pixel_shuffle_factor_width),
            ),
            "dynamic_image_size": bool(config.dynamic_image_size),
            "tile_size": int(config.tile_size),
            "min_num_tiles": int(config.min_num_tiles),
            "max_num_tiles": int(config.max_num_tiles),
            "use_thumbnail": bool(config.use_thumbnail),
        }
    )
    return DatNanoVLMProcessor(image_processor=image_processor, tokenizer=_DummyTokenizer())


def test_datnanovlm_tiling_blocks_adds_thumbnail_for_multi_tile() -> None:
    ratios = get_internvl_target_ratios(min_num=1, max_num=12)
    blocks, target_w, target_h = calculate_internvl_targets(
        orig_width=2048,
        orig_height=1024,
        target_ratios=ratios,
        image_size=448,
        use_thumbnail=True,
    )

    assert blocks > 1
    assert target_w % 448 == 0
    assert target_h % 448 == 0


def test_datnanovlm_tiling_blocks_without_thumbnail_matches_grid() -> None:
    ratios = get_internvl_target_ratios(min_num=1, max_num=12)
    blocks, target_w, target_h = calculate_internvl_targets(
        orig_width=1024,
        orig_height=1024,
        target_ratios=ratios,
        image_size=448,
        use_thumbnail=False,
    )

    assert blocks == (target_w // 448) * (target_h // 448)
    assert blocks >= 1


def test_datnanovlm_config_maps_siglip2_nativeres_fields() -> None:
    config = _make_nativeres_config()

    assert int(config.video_patch_size) == 14
    assert int(config.vision_max_num_patches) == 729
    assert int(config.pixel_shuffle_factor_height) == 3
    assert int(config.pixel_shuffle_factor_width) == 1
    assert int(config.image_tokens_per_image) == 243


def test_datnanovlm_config_defaults_dynamic_image_size_to_true() -> None:
    config = DatNanoVLMConfig(
        image_tokens_per_image=243,
        vision_config={"patch_size": 14},
        pixel_shuffle_factor_height=3,
        pixel_shuffle_factor_width=1,
        tile_size=384,
        min_num_tiles=1,
        max_num_tiles=12,
        use_thumbnail=True,
    )

    assert bool(config.dynamic_image_size) is True


def test_datnanovlm_config_allows_disabling_dynamic_image_size() -> None:
    config = DatNanoVLMConfig(
        image_tokens_per_image=243,
        vision_config={"patch_size": 14},
        pixel_shuffle_factor_height=3,
        pixel_shuffle_factor_width=1,
        dynamic_image_size=False,
        tile_size=384,
        min_num_tiles=1,
        max_num_tiles=12,
        use_thumbnail=True,
    )

    assert bool(config.dynamic_image_size) is False


def test_datnanovlm_processor_single_tile_shape_math() -> None:
    config = _make_nativeres_config()
    processor = _make_processor_from_config(config)

    image = Image.new("RGB", (384, 384), color="white")
    out = processor(text="<image>", images=[image], return_tensors="pt")

    assert tuple(out["pixel_values"].shape) == (729, 588)
    assert out["image_grid_thw"].tolist() == [[1, 27, 27]]
    assert out["image_num_tiles"].tolist() == [1]

    merge_length = int(config.pixel_shuffle_factor_height) * int(
        config.pixel_shuffle_factor_width
    )
    total_patch_tokens = int(out["image_grid_thw"].prod(dim=-1).sum().item())
    expected_image_pad_tokens = total_patch_tokens // merge_length
    assert expected_image_pad_tokens == 243
    prompt = out["input_text"][0]
    assert prompt.count(VISION_START_TOKEN) == 1
    assert prompt.count(VISION_END_TOKEN) == 1
    assert prompt.count(IMAGE_PAD_TOKEN) == expected_image_pad_tokens
    assert "<image>" not in prompt


def test_datnanovlm_processor_dynamic_tiling_shape_math_with_thumbnail() -> None:
    config = _make_nativeres_config()
    processor = _make_processor_from_config(config)

    image = Image.new("RGB", (1536, 512), color="white")
    out = processor(text="<image>", images=[image], return_tensors="pt")

    assert tuple(out["pixel_values"].shape) == (2916, 588)
    assert out["image_num_tiles"].tolist() == [4]
    assert out["image_grid_thw"].tolist() == [[1, 27, 27]] * 4

    merge_length = int(config.pixel_shuffle_factor_height) * int(
        config.pixel_shuffle_factor_width
    )
    total_patch_tokens = int(out["image_grid_thw"].prod(dim=-1).sum().item())
    expected_image_pad_tokens = total_patch_tokens // merge_length
    assert expected_image_pad_tokens == 972
    prompt = out["input_text"][0]
    assert prompt.count(VISION_START_TOKEN) == 1
    assert prompt.count(VISION_END_TOKEN) == 1
    assert prompt.count(IMAGE_PAD_TOKEN) == expected_image_pad_tokens
    assert "<image>" not in prompt
