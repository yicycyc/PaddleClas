import argparse

import numpy as np
import timm
import torch


def load_npz_state(model, npz_path):
    data = np.load(npz_path, allow_pickle=False)
    state = model.state_dict()
    loaded = 0
    for key in state.keys():
        if key not in data.files:
            continue
        state[key] = torch.from_numpy(data[key])
        loaded += 1
    model.load_state_dict(state, strict=False)
    print(f"loaded from npz: {npz_path}, keys={loaded}")


def main(variant, output, input_npz, state_npz=None, pretrained=False, seed=2026):
    torch.manual_seed(seed)
    model = timm.create_model(variant, pretrained=pretrained)
    if state_npz is not None:
        load_npz_state(model, state_npz)
    model.eval()

    x = np.load(input_npz, allow_pickle=False)["x"].astype("float32")
    x_t = torch.from_numpy(x)
    with torch.no_grad():
        feat = model.forward_features(x_t)
        out = model(x_t)

    np.savez(
        output, x=x, feat=feat.cpu().numpy(), out=out.cpu().numpy(), y=out.cpu().numpy()
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="ghostnetv3_100")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--input_npz", type=str, required=True)
    parser.add_argument("--state_npz", type=str, default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    main(
        args.variant,
        args.output,
        args.input_npz,
        args.state_npz,
        args.pretrained,
        args.seed,
    )
