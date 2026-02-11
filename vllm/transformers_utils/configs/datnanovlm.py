# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from transformers import AutoConfig

from vllm.transformers_utils.configs.isaac import IsaacConfig


class DatnaNoVLMConfig(IsaacConfig):
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

        if "architectures" not in kwargs:
            kwargs["architectures"] = ["DatnaNoVLMForConditionalGeneration"]

        if image_tokens_per_image is not None and "vision_max_num_patches" not in kwargs:
            kwargs["vision_max_num_patches"] = int(image_tokens_per_image)

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
        )
        self.special_token_ids = special_token_ids if special_token_ids is not None else {}
