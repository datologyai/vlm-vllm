# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from PIL import Image

from vllm.model_executor.models.isaac import (
    IsaacImageProcessor,
    IsaacProcessor,
    pixel_shuffle_varlen,
)


class _DummyTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_text": text}

    def apply_chat_template(self, messages, **kwargs):
        return messages


def test_pixel_shuffle_varlen_supports_asymmetric_factors() -> None:
    token_grids = torch.tensor([[3, 4]], dtype=torch.int32)
    x = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3 * 4, 2)

    out = pixel_shuffle_varlen(x=x, token_grids=token_grids, scale_factor=(3, 1))

    assert out.shape == (4, 6)


def test_dynamic_tiling_replaces_single_image_token_with_all_tile_tokens() -> None:
    image_processor = IsaacImageProcessor(
        {
            "patch_size": 14,
            "vision_max_num_patches": 729,
            "vision_min_num_patches": 729,
            "pixel_shuffle_factors": (3, 1),
            "dynamic_image_size": True,
            "tile_size": 384,
            "min_num_tiles": 1,
            "max_num_tiles": 12,
            "use_thumbnail": True,
        }
    )
    processor = IsaacProcessor(
        image_processor=image_processor,
        tokenizer=_DummyTokenizer(),
    )

    image = Image.new("RGB", (1536, 512), color="white")
    out = processor(text="<image>", images=[image], return_tensors="pt")

    image_num_tiles = int(out["image_num_tiles"][0].item())
    assert image_num_tiles > 1
    assert out["image_grid_thw"].shape[0] == image_num_tiles
    assert torch.all(out["image_grid_thw"][:, 1:] == torch.tensor([27, 27]))

    expected_tokens = int(out["image_grid_thw"].prod(dim=-1).sum().item() // 3)
    rendered_prompt = out["input_text"][0]
    assert rendered_prompt.count("<|image_pad|>") == expected_tokens
