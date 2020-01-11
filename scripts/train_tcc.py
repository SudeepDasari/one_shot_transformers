from hem.parse_util import parse_basic_config
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument('experiment_file', type=str, help='path to YAML experiment config file')
    args = parser.parse_args()

    print(args)
