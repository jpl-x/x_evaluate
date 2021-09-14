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
    dataset_mapping = {x.lower().replace(' ', '_'): x for x in rpg_davis_data}

    dataset_mapping["Mars Circle Vmax 7.2 Offset 2.5"] = "Mars Circle"
    dataset_mapping["Mars Vertical Circle Vmax 2.4 Offset 2.5"] = "Mars Vertical Circle"
    dataset_mapping["Mars Mellon Vmax 12.4 Offset 10"] = "Mars Mellon"

    # dataset_mapping["Mars Circle Vmax 7.2 Offset 2.5 no bootsrapping"] = "Mars Circle"
    # dataset_mapping["Mars Vertical Circle Vmax 2.4 Offset 2.5 no bootstrapping"] = "Mars Vertical Circle"
    # dataset_mapping["Mars Mellon Vmax 12.4 Offset 10 no bootsrapping"] = "Mars Mellon"

    keys = list(s.data.keys()).copy()
    for k in keys:
        if k in dataset_mapping.keys():
            print(F"Renaming dataset '{k}' --> {dataset_mapping[k]}")
            s.data[dataset_mapping[k]] = s.data.pop(k)

    print(F"Renaming '{s.name}' to '{args.new_name}'")
    s.name = args.new_name
    print("Writing evaluation pickle")
    write_evaluation_pickle(s, output_root, filename)


if __name__ == '__main__':
    main()
