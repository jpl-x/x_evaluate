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

    args = parser.parse_args()

    evaluation_files = find_evaluation_files_recursively(args.root_folder)

    print(evaluation_files)

    must_datasets = {"Boxes 6DOF", "Boxes Translation", "Dynamic 6DOF", "Dynamic Translation", "HDR Boxes",
                     "HDR Poster", "Poster 6DOF", "Poster Translation", "Shapes 6DOF", "Shapes Translation"}

    eklt_tables = {}
    haste_tables = {}
    xvio_tables = {}

    for f in tqdm(evaluation_files):
        # if "eklt" not in f:
        #     # print(F"Skipping {f}")
        #     continue
        # print(F"Reading {f}")
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
