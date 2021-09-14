import argparse
import glob
import os
import pickle

from tqdm import tqdm

from x_evaluate.evaluation_data import FrontEnd
from x_evaluate.scriptlets import find_evaluation_files_recursively, read_evaluation_pickle
from x_evaluate.trajectory_evaluation import create_trajectory_result_table_wrt_traveled_dist, \
    create_trajectory_completion_table


def main():
    parser = argparse.ArgumentParser(description='Hand-made script for getting all result tables')
    parser.add_argument('--root_folder', type=str, required=True)
    parser.add_argument('--path_match', type=str, default=None)

    args = parser.parse_args()

    evaluation_files = find_evaluation_files_recursively(args.root_folder)

    if args.path_match is not None:
        evaluation_files = [e for e in evaluation_files if args.path_match in e]

    print(evaluation_files)

    must_datasets = {"Boxes 6DOF", "Boxes Translation", "Dynamic 6DOF", "Dynamic Translation", "HDR Boxes",
                     "HDR Poster", "Poster 6DOF", "Poster Translation", "Shapes 6DOF", "Shapes Translation"}

    must_datasets = {"Mars Straight Vmax 3.2 Offset 2.5", "Mars Eight Vmax 3.5 Offset 2.5",
                     "Mars Circle Vmax 7.2 Offset 2.5", "Mars Vertical Circle Vmax 2.4 Offset 2.5",
                     "Mars Straight Vmax 3.2 Offset 5", "Mars Eight Vmax 3.5 Offset 5",
                     "Mars Circle Vmax 7.2 Offset 5", "Mars Vertical Circle Vmax 2.4 Offset 5",
                     "Mars Straight Vmax 3.2 Offset 10", "Mars Eight Vmax 3.5 Offset 10",
                     "Mars Circle Vmax 7.2 Offset 10", "Mars Vertical Circle Vmax 2.4 Offset 10"}

    # must_datasets = {"Mars Straight Vmax 3.2 Offset 2.5", "Mars Eight Vmax 3.5 Offset 2.5", "Mars Circle Vmax 7.2 Offset 2.5", "Mars Vertical Circle Vmax 2.4 Offset 2.5", "Mars Circle Vmax 7.2 Offset 10", "Mars Circle Vmax 16.6 Offset 10"}
    must_datasets = {"Mars Straight Vmax 3.2 Offset 2.5", "Mars Circle Vmax 7.2 Offset 2.5", "Mars Vertical Circle Vmax 2.4 Offset 2.5", "Mars Circle Vmax 7.2 Offset 10", "Mars Circle Vmax 16.6 Offset 10"}

    must_datasets = {
        "Mars Vertical Circle Vmax 2.4 Offset 2.5",
        "Mars Circle Vmax 7.2 Offset 2.5",
        "Mars Mellon Vmax 12.4 Offset 10",
    }

    eklt_tables = {}
    haste_tables = {}
    xvio_tables = {}

    for f in tqdm(evaluation_files):
        # if "eklt" not in f:
        #     # print(F"Skipping {f}")
        #     continue
        print(F"Reading {f}")
        s = read_evaluation_pickle(os.path.dirname(f), os.path.basename(f))
        # print(s.data.keys())

        try:
            if len(must_datasets.intersection(s.data.keys())) == len(must_datasets):
                # print("We want you!")
                table = create_trajectory_result_table_wrt_traveled_dist(s)
                completion_table = create_trajectory_completion_table(s)
                table = table.merge(completion_table, left_index=True, right_index=True)
                if s.frontend == FrontEnd.EKLT:
                    eklt_tables[f] = table
                    with open("/tmp/eklt-tables.pickle", 'wb') as file:
                        pickle.dump(eklt_tables, file, pickle.HIGHEST_PROTOCOL)
                elif s.frontend == FrontEnd.HASTE:
                    haste_tables[f] = table
                    with open("/tmp/haste-tables.pickle", 'wb') as file:
                        pickle.dump(haste_tables, file, pickle.HIGHEST_PROTOCOL)
                elif s.frontend == FrontEnd.XVIO:
                    xvio_tables[f] = table
                    with open("/tmp/xvio-tables.pickle", 'wb') as file:
                        pickle.dump(xvio_tables, file, pickle.HIGHEST_PROTOCOL)
                # print(F"Got table and saved it for {s.frontend}")
                # print(table)
            # else:
            #     print("Datasets do not match:")
            #     a = list(s.data.keys())
            #     b = list(must_datasets)
            #     a.sort()
            #     b.sort()
            #     print(a)
            #     print(b)
        except:
            # print(F"Warning: {f} FAILED")
            pass

    print("Saving all tables to pickle")
    with open("/tmp/eklt-tables.pickle", 'wb') as file:
        pickle.dump(eklt_tables, file, pickle.HIGHEST_PROTOCOL)
    with open("/tmp/haste-tables.pickle", 'wb') as file:
        pickle.dump(haste_tables, file, pickle.HIGHEST_PROTOCOL)
    with open("/tmp/xvio-tables.pickle", 'wb') as file:
        pickle.dump(xvio_tables, file, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
