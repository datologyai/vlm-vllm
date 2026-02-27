# Datology SigLIP2-Qwen3 Native-Res Port Plan

## Goal
Support datnanovlm `siglip2-qwen3-1.7b-nativeres` checkpoints in a custom vLLM `v0.15.1` build with:
- Native-resolution InternVL-style tiling
- Asymmetric pixel shuffle `(3, 1)`
- Standard vLLM multimodal processing and launcher paths

## Upstream References
- Isaac support merge: https://github.com/vllm-project/vllm/pull/31550
- Isaac architecture discussion: https://github.com/vllm-project/vllm/issues/25448
- Related multimodal native-res / processor evolution: https://github.com/vllm-project/vllm/pull/28367

## Reused vLLM Components
- `vllm/model_executor/models/isaac.py`
  - SigLIP2 vision tower
  - packed patch input path
  - multimodal connector + Qwen3 language model wiring
- `vllm/model_executor/models/internvl.py`
  - `get_internvl_target_ratios`
  - `calculate_internvl_targets`
  - `dynamic_preprocess_internvl`

## Custom Delta Implemented
- Extended Isaac config and runtime for asymmetric pixel shuffle factors.
- Added native-res tiling controls on the Isaac processor path:
  - `dynamic_image_size`
  - `tile_size`
  - `min_num_tiles`
  - `max_num_tiles`
  - `use_thumbnail`
- Grouped tile features per source image so one `<image>` placeholder maps to all tiles of that image.

## Files Changed
- `vllm/transformers_utils/configs/isaac.py`
- `vllm/model_executor/models/isaac.py`
- `tests/models/multimodal/processing/test_isaac.py`

## Integration Notes
- Preferred config shape for exported checkpoints:
  - `model_type: "isaac"`
  - `architectures: ["IsaacForConditionalGeneration"]`
  - Include `pixel_shuffle_factors: [3, 1]`
  - Include tiling fields (`dynamic_image_size`, `tile_size`, `min_num_tiles`, `max_num_tiles`, `use_thumbnail`)
- This keeps vLLM model resolution on canonical registry paths without requiring a separate datnanovlm-only runtime.
