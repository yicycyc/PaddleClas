import argparse
import os
from collections import OrderedDict

import numpy as np
import paddle
try:
    import torch
except Exception:
    torch = None

from ppcls.arch.backbone import MobileNetV5_300M, MobileNetV5_300M_enc, MobileNetV5_base
from ppcls.utils import logger


ARCH_DEF = {
    "mobilenetv5_300m": [3, 5, 37, 39],
    "mobilenetv5_300m_enc": [3, 5, 37, 39],
    "mobilenetv5_base": [3, 5, 16, 15],
}


def _build_index_map(variant: str):
    stage_lens = ARCH_DEF[variant]
    stage_offsets = []
    acc = 0
    for ln in stage_lens:
        stage_offsets.append(acc)
        acc += ln
    return stage_offsets


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
    out = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith("module."):
            k = k[len("module."):]
        out[k] = v
    return out


def _load_source_state(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path, allow_pickle=False)
        out = OrderedDict()
        for k in data.files:
            out[k] = data[k]
        return out

    if torch is None:
        raise RuntimeError(
            "torch is not available. For .pth input install torch, or export to .npz first.")
    ckpt = torch.load(path, map_location="cpu")
    return _extract_state_dict(ckpt)


def _map_block_key(k: str, stage_offsets):
    # blocks.{stage}.{idx}.*  -> blocks.{flat_idx}.*
    parts = k.split(".")
    if len(parts) < 4 or parts[0] != "blocks":
        return k
    try:
        stage_id = int(parts[1])
        inner_id = int(parts[2])
    except ValueError:
        return k

    flat_idx = stage_offsets[stage_id] + inner_id
    suffix = ".".join(parts[3:])

    # EdgeResidual mapping
    suffix = suffix.replace("conv_exp.weight", "conv_exp.conv.weight")
    suffix = suffix.replace("bn1.weight", "conv_exp.bn.weight")
    suffix = suffix.replace("bn1.bias", "conv_exp.bn.bias")
    suffix = suffix.replace("conv_pwl.weight", "conv_pwl.conv.weight")
    suffix = suffix.replace("bn2.weight", "conv_pwl.bn.weight")
    suffix = suffix.replace("bn2.bias", "conv_pwl.bn.bias")

    # UIR mapping
    suffix = suffix.replace("dw_start.bn.weight", "dw_start.bn.weight")
    suffix = suffix.replace("dw_start.bn.bias", "dw_start.bn.bias")
    suffix = suffix.replace("pw_exp.bn.weight", "pw_exp.bn.weight")
    suffix = suffix.replace("pw_exp.bn.bias", "pw_exp.bn.bias")
    suffix = suffix.replace("dw_mid.bn.weight", "dw_mid.bn.weight")
    suffix = suffix.replace("dw_mid.bn.bias", "dw_mid.bn.bias")
    suffix = suffix.replace("pw_proj.bn.weight", "pw_proj.bn.weight")
    suffix = suffix.replace("pw_proj.bn.bias", "pw_proj.bn.bias")

    # MobileAttention mapping
    suffix = suffix.replace("norm.weight", "norm.weight")
    suffix = suffix.replace("norm.bias", "norm.bias")
    suffix = suffix.replace("attn.query.proj.weight", "query_proj.weight")
    suffix = suffix.replace("attn.query.proj.bias", "query_proj.bias")
    suffix = suffix.replace("attn.key.down_conv.weight", "key_down_proj.conv.weight")
    suffix = suffix.replace("attn.key.norm.weight", "key_down_proj.bn.weight")
    suffix = suffix.replace("attn.key.norm.bias", "key_down_proj.bn.bias")
    suffix = suffix.replace("attn.value.down_conv.weight", "value_down_proj.conv.weight")
    suffix = suffix.replace("attn.value.norm.weight", "value_down_proj.bn.weight")
    suffix = suffix.replace("attn.value.norm.bias", "value_down_proj.bn.bias")
    suffix = suffix.replace("attn.key.proj.weight", "key_proj.weight")
    suffix = suffix.replace("attn.key.proj.bias", "key_proj.bias")
    suffix = suffix.replace("attn.value.proj.weight", "value_proj.weight")
    suffix = suffix.replace("attn.value.proj.bias", "value_proj.bias")
    suffix = suffix.replace("attn.output.proj.weight", "proj.weight")
    suffix = suffix.replace("attn.output.proj.bias", "proj.bias")

    return f"blocks.{flat_idx}.{suffix}"


def _map_key(k: str, stage_offsets):
    if "num_batches_tracked" in k:
        return None

    # timm key style for blocks
    if k.startswith("blocks."):
        return _map_block_key(k, stage_offsets)

    # stem
    if k.startswith("conv_stem.bn."):
        return k

    # msfa mapping
    k = k.replace("msfa.ffn.", "msfa.ffn.")

    # common running stats names
    k = k.replace("running_var", "_variance")
    k = k.replace("running_mean", "_mean")
    return k


def _maybe_transpose_linear_weight(new_k, arr, model_state):
    """Transpose 2D weights when Paddle/torch layout is reversed.

    This handles both explicit nn.Linear names and conv-named linear weights.
    """
    if new_k not in model_state:
        return arr, False
    if arr.ndim != 2:
        return arr, False

    target_shape = tuple(model_state[new_k].shape)
    if tuple(arr.shape) == target_shape:
        return arr, False
    if tuple(arr.transpose(1, 0).shape) == target_shape:
        return arr.transpose(1, 0), True
    return arr, False


def torch2paddle(torch_path, paddle_path, variant="mobilenetv5_300m"):
    logger.init_logger()
    if variant not in ARCH_DEF:
        raise ValueError(f"Unsupported variant: {variant}")

    if variant == "mobilenetv5_300m":
        model = MobileNetV5_300M(class_num=0)
    elif variant == "mobilenetv5_300m_enc":
        model = MobileNetV5_300M_enc()
    else:
        model = MobileNetV5_base(class_num=1000)

    model_state = model.state_dict()

    torch_state = _load_source_state(torch_path)
    stage_offsets = _build_index_map(variant)

    paddle_state = OrderedDict()
    converted, skipped, shape_mismatch = 0, 0, 0

    for k, v in torch_state.items():
        new_k = _map_key(k, stage_offsets)
        if new_k is None:
            skipped += 1
            continue

        if torch is not None and torch.is_tensor(v):
            arr = v.detach().cpu().numpy()
        else:
            arr = np.array(v)

        arr, transposed = _maybe_transpose_linear_weight(new_k, arr, model_state)
        if transposed:
            print(f"[transpose] {k} -> {new_k}: {v.shape} -> {arr.shape}")

        if new_k not in model_state:
            skipped += 1
            continue

        if list(arr.shape) != list(model_state[new_k].shape):
            shape_mismatch += 1
            print(
                f"[shape_mismatch] {k} -> {new_k}: torch {arr.shape} vs paddle {list(model_state[new_k].shape)}"
            )
            continue

        paddle_state[new_k] = arr
        converted += 1

    # keep default init params for unmapped keys
    final_state = OrderedDict()
    for k, v in model_state.items():
        if k in paddle_state:
            final_state[k] = paddle_state[k]
        else:
            final_state[k] = v.numpy()

    os.makedirs(os.path.dirname(os.path.abspath(paddle_path)), exist_ok=True)
    paddle.save(final_state, paddle_path)

    print(f"[done] variant={variant}")
    print(f"[done] converted={converted}, skipped={skipped}, shape_mismatch={shape_mismatch}")
    print(f"[done] saved to: {paddle_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert timm MobileNet-v5 weights to Paddle format")
    parser.add_argument("--torch_path", type=str, required=True)
    parser.add_argument("--paddle_path", type=str, required=True)
    parser.add_argument(
        "--variant",
        type=str,
        default="mobilenetv5_300m",
        choices=["mobilenetv5_300m", "mobilenetv5_300m_enc", "mobilenetv5_base"],
    )
    args = parser.parse_args()

    torch2paddle(args.torch_path, args.paddle_path, args.variant)
