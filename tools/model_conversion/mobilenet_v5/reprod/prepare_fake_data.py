import argparse
import numpy as np


def main(output, seed=2026, input_size=256):
    rng = np.random.RandomState(seed)
    x = rng.randn(1, 3, input_size, input_size).astype('float32')
    np.savez(output, x=x, seed=np.array([seed], dtype='int64'))
    print(f'saved fake data: {output}, x shape={x.shape}, seed={seed}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--input_size', type=int, default=256)
    args = parser.parse_args()
    main(args.output, args.seed, args.input_size)
