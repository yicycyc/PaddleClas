import argparse

import numpy as np
import timm
import torch


def main(variant, output):
    model = timm.create_model(variant, pretrained=False)
    sd = model.state_dict()
    np_sd = {}
    for k, v in sd.items():
        if "num_batches_tracked" in k:
            continue
        np_sd[k] = v.detach().cpu().numpy()
    np.savez(output, **np_sd)
    print(f"saved: {output}")
    print(f"num_keys: {len(np_sd)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="mobilenetv5_300m")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    main(args.variant, args.output)
