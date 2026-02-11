# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from vllm.multimodal import MULTIMODAL_REGISTRY

from .isaac import (
    IsaacDummyInputsBuilder,
    IsaacForConditionalGeneration,
    IsaacMultiModalProcessor,
    IsaacProcessingInfo,
)


@MULTIMODAL_REGISTRY.register_processor(
    IsaacMultiModalProcessor,
    info=IsaacProcessingInfo,
    dummy_inputs=IsaacDummyInputsBuilder,
)
class DatNanoVLMForConditionalGeneration(IsaacForConditionalGeneration):
    """DatNanoVLM native architecture entrypoint.

    This keeps runtime behavior aligned with Isaac-derived multimodal execution
    while allowing explicit  / architecture routing.
    """

    pass
