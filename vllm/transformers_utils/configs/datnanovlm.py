# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from transformers import AutoConfig

from vllm.transformers_utils.configs.isaac import IsaacConfig


class DatNanoVLMConfig(IsaacConfig):
    """Config adapter for datnanovlm SigLIP2/Qwen3 native-res checkpoints.

    This parses the datnanovlm HF schema directly (model_type=datnanovlm)
    and adapts it to the Isaac execution config expected by vLLM.
    """

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

        patch_size = vision_config.get("patch_size")
        patch_size_int = int(patch_size) if patch_size is not None else None

        if "architectures" not in kwargs:
            kwargs["architectures"] = ["DatNanoVLMForConditionalGeneration"]

        # Ensure the Isaac processing config matches the vision backbone.
        # This is distinct from `vision_config.patch_size`, which is used by
        # the model itself.
        if patch_size_int is not None and "vision_patch_size" not in kwargs:
            kwargs["vision_patch_size"] = patch_size_int

        factor_h = kwargs.get("pixel_shuffle_factor_height")
        factor_w = kwargs.get("pixel_shuffle_factor_width")
        if factor_h is not None:
            factor_h = int(factor_h)
        if factor_w is not None:
            factor_w = int(factor_w)
        if factor_h is not None or factor_w is not None:
            if factor_h is None or factor_w is None:
                raise ValueError(
                    "DatNanoVLMConfig requires both pixel_shuffle_factor_height and "
                    "pixel_shuffle_factor_width when specifying pixel shuffle factors."
                )

            existing_h = vision_config.get("pixel_shuffle_factor_height")
            existing_w = vision_config.get("pixel_shuffle_factor_width")
            if existing_h is not None and int(existing_h) != factor_h:
                raise ValueError(
                    "pixel_shuffle_factor_height mismatch between top-level config "
                    f"({factor_h}) and vision_config ({existing_h})."
                )
            if existing_w is not None and int(existing_w) != factor_w:
                raise ValueError(
                    "pixel_shuffle_factor_width mismatch between top-level config "
                    f"({factor_w}) and vision_config ({existing_w})."
                )
            vision_config["pixel_shuffle_factor_height"] = factor_h
            vision_config["pixel_shuffle_factor_width"] = factor_w

        # `vision_max_num_patches` is a preprocessing constraint on the
        # *pre-shuffle* patch tokens per tile. It should not be set to the
        # post-shuffle `image_tokens_per_image` value.
        if "vision_max_num_patches" not in kwargs:
            tile_size = kwargs.get("tile_size")
            tile_size_int = int(tile_size) if tile_size is not None else None
            if tile_size_int is not None and patch_size_int is not None:
                kwargs["vision_max_num_patches"] = (
                    tile_size_int // patch_size_int
                ) ** 2

        super().__init__(text_config=text_config, vision_config=vision_config, **kwargs)

        self.llm_backbone = llm_backbone
        self.vision_backbone = vision_backbone
        self.architecture = architecture
        self.model_name = model_name
        self.projector_config = projector_config if projector_config is not None else {}
        self.image_tokens_per_image = (
            int(image_tokens_per_image)
            if image_tokens_per_image is not None
            else int(self.vision_max_num_patches)
            // int(self.pixel_shuffle_factor_height * self.pixel_shuffle_factor_width)
        )
        self.special_token_ids = special_token_ids if special_token_ids is not None else {}
