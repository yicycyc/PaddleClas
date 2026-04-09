import argparse

import numpy as np
import timm
import torch


def main(variant, output, pretrained=False, seed=2026):
    torch.manual_seed(seed)
    model = timm.create_model(variant, pretrained=pretrained)
    state = model.state_dict()
    arrays = {}
    for key, value in state.items():
        if key.endswith("num_batches_tracked"):
            continue
        arrays[key] = value.detach().cpu().numpy()
    np.savez(output, **arrays)
    print(f"saved: {output}")
    print(f"keys: {len(arrays)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="ghostnetv3_100")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    main(args.variant, args.output, args.pretrained, args.seed)
