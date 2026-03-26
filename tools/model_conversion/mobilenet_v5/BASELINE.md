# MobileNet-v5 Conversion Baseline

## Reference Source
- Upstream repo: https://github.com/huggingface/pytorch-image-models
- Fixed commit: `8d0f79effa3dbc922afbfb431fbadd4648938de7`
- File: `timm/models/mobilenetv5.py`
- Release tag at commit: `Release 1.0.26` (2026-03-23)

## Variants Implemented in PaddleClas (skeleton)
- `MobileNetV5_300M`
- `MobileNetV5_base`

## Current Scope
- Implemented: model structure conversion for forward execution in PaddleClas.
- Implemented: parser for timm `arch_def` block strings (`er`, `uir`, `mqa`).
- Implemented: MSFA (multi-scale fusion adapter) module and classification head.
- Not implemented yet: official pretrained URL and PyTorch->Paddle weight conversion script.

## Weight Mapping Plan (next)
- `conv_stem.*` -> `conv_stem.*`
- `blocks.{i}.*` -> `blocks.{i}.*` (same block order as parsed `arch_def`)
- `msfa.*` -> `msfa.*`
- `classifier.*` -> `classifier.*`

## Precision Validation Plan (after torch env ready)
1. Export PyTorch reference checkpoint from fixed commit.
2. Convert weights with explicit key mapping table.
3. Compare per-layer output and final logits on identical input.
4. Target: max abs diff <= 1e-4 (fp32, eval mode).
