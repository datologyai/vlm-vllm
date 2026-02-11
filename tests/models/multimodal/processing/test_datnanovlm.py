# SPDX-License-Identifier: Apache-2.0

from vllm.model_executor.models.internvl import (
    calculate_internvl_targets,
    get_internvl_target_ratios,
)


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
