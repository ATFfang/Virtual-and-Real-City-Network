from MultilayerSupraMatrix import caculate_supra_matrix
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="time")
    parser.add_argument('--time', type=int, required=True, help='time(year)')

    return parser.parse_args()

def main(args):
    index_time = args.time
    caculate_supra_matrix(index_time)

if __name__ == '__main__':
    args = parse_args()
    main(args)