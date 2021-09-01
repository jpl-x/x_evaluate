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

    # Naming quick fix for Ultimate SLAM
    rpg_davis_data = ["Boxes 6DOF", "Boxes Translation", "Dynamic 6DOF", "Dynamic Translation", "HDR Boxes",
                      "HDR Poster", "Poster 6DOF", "Poster Translation", "Shapes 6DOF", "Shapes Translation"]
    small_cap_mapping = {x.lower().replace(' ', '_'): x for x in rpg_davis_data}
    keys = list(s.data.keys()).copy()
    for k in keys:
        if k in small_cap_mapping.keys():
            print(F"Renaming dataset '{k}' --> {small_cap_mapping[k]}")
            s.data[small_cap_mapping[k]] = s.data.pop(k)

    print(F"Renaming '{s.name}' to '{args.new_name}'")
    s.name = args.new_name
    print("Writing evaluation pickle")
    write_evaluation_pickle(s, output_root, filename)


if __name__ == '__main__':
    main()
