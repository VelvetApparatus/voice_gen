import argparse


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-out", "--output", required=True)
    parser.add_argument("-in", "--input", required=True)
    parser.add_argument("-l", "--limit")
    parser.add_argument("-n", "--name")

    return parser.parse_args()
