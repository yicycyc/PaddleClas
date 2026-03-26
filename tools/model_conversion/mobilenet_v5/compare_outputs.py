import argparse

import numpy as np


def main(torch_npz, paddle_npz):
    t = np.load(torch_npz)["y"]
    p = np.load(paddle_npz)["y"]

    if t.shape != p.shape:
        raise ValueError(f"Shape mismatch: torch {t.shape}, paddle {p.shape}")

    diff = np.abs(t - p)
    print(f"shape: {t.shape}")
    print(f"max_abs_diff: {diff.max():.8f}")
    print(f"mean_abs_diff: {diff.mean():.8f}")
    print(f"p99_abs_diff: {np.percentile(diff, 99):.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch_npz", type=str, required=True)
    parser.add_argument("--paddle_npz", type=str, required=True)
    args = parser.parse_args()
    main(args.torch_npz, args.paddle_npz)
