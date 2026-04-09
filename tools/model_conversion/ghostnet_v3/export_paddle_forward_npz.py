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


def main(variant, pdparams, input_npz, output, class_num=1000):
    model = build_model(variant, class_num=class_num)
    state = paddle.load(pdparams)
    model.set_state_dict(state)
    model.eval()

    x = np.load(input_npz, allow_pickle=False)["x"].astype("float32")
    x_t = paddle.to_tensor(x)
    with paddle.no_grad():
        feat = model.forward_features(x_t).numpy()
        out = model(x_t).numpy()

    np.savez(output, x=x, feat=feat, out=out, y=out)
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="ghostnetv3_100")
    parser.add_argument("--pdparams", type=str, required=True)
    parser.add_argument("--input_npz", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--class_num", type=int, default=1000)
    args = parser.parse_args()
    main(args.variant, args.pdparams, args.input_npz, args.output, args.class_num)
