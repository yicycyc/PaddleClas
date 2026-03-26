import argparse
from reprod_log import ReprodDiffHelper


def main(torch_file, paddle_file, log_file, atol=1e-5):
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info(torch_file)
    paddle_info = diff_helper.load_info(paddle_file)

    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path=log_file, diff_threshold=atol)
    print(f'saved diff report: {log_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_file', type=str, required=True)
    parser.add_argument('--paddle_file', type=str, required=True)
    parser.add_argument('--log_file', type=str, required=True)
    parser.add_argument('--atol', type=float, default=1e-5)
    args = parser.parse_args()
    main(args.torch_file, args.paddle_file, args.log_file, args.atol)
