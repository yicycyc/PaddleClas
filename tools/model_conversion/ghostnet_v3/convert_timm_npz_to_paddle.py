import argparse
import os
import sys
import types

import numpy as np
import paddle

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../../../")))
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from ppcls.arch.backbone import (  # noqa: E402
    GhostNetV3_x0_5,
    GhostNetV3_x1_0,
    GhostNetV3_x1_3,
    GhostNetV3_x1_6,
)


def build_model(variant, class_num=1000):
    if variant == "ghostnetv3_050":
        return GhostNetV3_x0_5(class_num=class_num)
    if variant == "ghostnetv3_100":
        return GhostNetV3_x1_0(class_num=class_num)
    if variant == "ghostnetv3_130":
        return GhostNetV3_x1_3(class_num=class_num)
    if variant == "ghostnetv3_160":
        return GhostNetV3_x1_6(class_num=class_num)
    raise ValueError(f"Unsupported variant: {variant}")


def map_key(torch_key):
    if torch_key.endswith("running_mean"):
        return torch_key.replace("running_mean", "_mean")
    if torch_key.endswith("running_var"):
        return torch_key.replace("running_var", "_variance")
    # Official GhostNetV3 release uses `bn` in helper Conv-BN blocks while
    # the Paddle implementation reuses PaddleClas naming with `bn1`.
    return torch_key.replace(".bn.", ".bn1.")


def main(variant, state_npz, output, class_num=1000):
    model = build_model(variant, class_num=class_num)
    state = model.state_dict()
    torch_state = np.load(state_npz, allow_pickle=False)

    loaded = 0
    for torch_key in torch_state.files:
        paddle_key = map_key(torch_key)
        if paddle_key not in state:
            continue
        value = torch_state[torch_key]
        if torch_key == "classifier.weight":
            value = value.transpose(1, 0)
        if tuple(state[paddle_key].shape) != tuple(value.shape):
            raise ValueError(
                f"{torch_key} -> {paddle_key} shape mismatch: "
                f"torch {value.shape}, paddle {tuple(state[paddle_key].shape)}"
            )
        state[paddle_key] = paddle.to_tensor(value)
        loaded += 1

    model.set_state_dict(state)
    paddle.save(model.state_dict(), output)
    print(f"saved: {output}")
    print(f"loaded keys: {loaded}")
    print(f"model keys: {len(state)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="ghostnetv3_100")
    parser.add_argument("--state_npz", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--class_num", type=int, default=1000)
    args = parser.parse_args()
    main(args.variant, args.state_npz, args.output, args.class_num)
