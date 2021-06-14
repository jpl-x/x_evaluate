import argparse
import os

from x_evaluate.scriptlets import read_evaluation_pickle, write_evaluation_pickle


def main():
    parser = argparse.ArgumentParser(description='Reads evaluation.pickle and plots all summary plots')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--new_name', type=str, required=True)

    args = parser.parse_args()

    output_root = os.path.dirname(args.input)
    filename = os.path.basename(args.input)

    print(F"Reading {args.input}")
    s = read_evaluation_pickle(output_root, filename)
    print(F"Renaming '{s.name}' to '{args.new_name}'")
    s.name = args.new_name
    print("Writing evaluation pickle")
    write_evaluation_pickle(s, output_root, filename)


if __name__ == '__main__':
    main()
