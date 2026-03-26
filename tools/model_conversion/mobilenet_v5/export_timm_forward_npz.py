import argparse

import numpy as np
import timm
import torch


def _load_npz_state(model, npz_path):
    data = np.load(npz_path, allow_pickle=False)
    sd = model.state_dict()
    loaded = 0
    for k in sd.keys():
        if k in data.files:
            sd[k] = torch.from_numpy(data[k])
            loaded += 1
    model.load_state_dict(sd, strict=False)
    print(f"loaded from npz: {npz_path}, keys={loaded}")


def main(variant, output, seed=2026, input_size=256, state_npz=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = timm.create_model(variant, pretrained=False)
    if state_npz is not None:
        _load_npz_state(model, state_npz)
    model.eval()

    x = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        y = model(x)

    np.savez(output, x=x.cpu().numpy(), y=y.cpu().numpy())
    print(f"saved: {output}")
    print(f"x shape: {tuple(x.shape)}")
    print(f"y shape: {tuple(y.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="mobilenetv5_300m")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--state_npz", type=str, default=None)
    args = parser.parse_args()

    main(args.variant, args.output, args.seed, args.input_size, args.state_npz)
