import argparse
import os
import pickle
from enum import Enum

import numpy as np
import pandas as pd


class Dataset(Enum):
    RPG_DAVIS = 'RPG_DAVIS'
    RPG_DAVIS_ROTATION = 'RPG_DAVIS_ROTATION'
    RPG_FPV = 'RPG_FPV'


def main():
    parser = argparse.ArgumentParser(description="Picks the best tables from a filename->result table map")
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--dataset', type=Dataset, choices=list(Dataset), required=True)

    args = parser.parse_args()

    input_file = args.input_file
    with open(input_file, 'rb') as f:
        tables = pickle.load(f)

    # print(tables)

    dataset_sequences = {
        Dataset.RPG_DAVIS: ["Boxes 6DOF", "Boxes Translation", "Dynamic 6DOF", "Dynamic Translation", "HDR Boxes",
                            "HDR Poster", "Poster 6DOF", "Poster Translation", "Shapes 6DOF", "Shapes Translation"],
        Dataset.RPG_DAVIS_ROTATION: ["Boxes Rotation", "Dynamic Rotation", "Poster Rotation", "Shapes Rotation"],
        Dataset.RPG_FPV: ['Indoor 45 Seq 12', 'Indoor 45 Seq 2']
    }

    sequence_lengths = {
        Dataset.RPG_DAVIS: [69.9, 65.2, 39.6, 30.1, 55.1, 55.4, 61.1, 49.3, 47.6, 56.1],
        Dataset.RPG_DAVIS_ROTATION: [14.9, 10.5, 16.9, 15.7],
        Dataset.RPG_FPV: [118.4, 176.2],
    }

    sequences = dataset_sequences[args.dataset]
    expected_lengths = sequence_lengths[args.dataset]

    stats = {
        "max": lambda p: p.max(),
        "mean": lambda p: p.mean(),
        "median": lambda p: p.median()
    }

    best_stats = {k: np.inf for k in stats.keys()}
    best_stats_keys = {k: None for k in stats.keys()}

    best_per_sequence = pd.DataFrame.from_dict({"Sequence": sequences,
                                                "Mean Position Error [%]": [np.inf] * len(sequences),
                                                "Mean Rotation error [deg/m]": [np.inf] * len(sequences),
                                                "Completion rate [%]": [0] * len(sequences),
                                                "File": [""] * len(sequences)})

    best_per_sequence = best_per_sequence.set_index("Sequence")

    pos_error_values = {}
    rot_error_values = {}

    for s in sequences:
        pos_error_values[s] = []
        rot_error_values[s] = []

    for k, v in tables.items():
        table = v.droplevel(0, axis=1)
        if isinstance(table.columns, pd.MultiIndex):
            table = v.droplevel(0, axis=1)
        table = table.loc[sequences, :]

        # only process if GT trajectory length is within 1m of expected length
        if np.all(np.abs(table["GT trajectory length [m]"].to_numpy() - expected_lengths) <= 1) \
                and table["Completion rate [%]"].min() > 99:
            for stat, func in stats.items():
                if func(table["Mean Position Error [%]"]) < best_stats[stat]:
                    best_stats[stat] = func(table["Mean Position Error [%]"])
                    best_stats_keys[stat] = k

        for i, s in enumerate(sequences):
            if np.abs(expected_lengths[i] - table.loc[s, "GT trajectory length [m]"]) > 1 \
                    or table.loc[s, "Completion rate [%]"] < 99:
                continue

            if best_per_sequence.loc[s, "Mean Position Error [%]"] > table.loc[s, "Mean Position Error [%]"]:
                best_per_sequence.loc[s, "Mean Position Error [%]"] = table.loc[s, "Mean Position Error [%]"]
                best_per_sequence.loc[s, "Mean Rotation error [deg/m]"] = table.loc[s, "Mean Rotation error [deg/m]"]
                best_per_sequence.loc[s, "Completion rate [%]"] = table.loc[s, "Completion rate [%]"]
                best_per_sequence.loc[s, "File"] = k

        # print(k)

    print("BEST PER SEQUENCE:")
    print(best_per_sequence[["Mean Position Error [%]", "Mean Rotation error [deg/m]"]])

    # for s in sequences:
    #     index = np.argmin(pos_error_values[s])
    #     print(F"{s}: {pos_error_values[s][index]}  {rot_error_values[s][index]} from '{list(tables.keys())[index]}'")


    for stat, table_key in best_stats_keys.items():
        print(F"\nBest {stat} (={best_stats[stat]} table {table_key}:")
        table = tables[table_key]
        if isinstance(table.columns, pd.MultiIndex):
            table = tables[table_key].droplevel(0, axis=1)
        print(table[["Mean Position Error [%]", "Mean Rotation error [deg/m]"]])

    output_filename = os.path.join(os.path.dirname(input_file), os.path.basename(input_file)[:-7] + "-result.xlsx")
    with pd.ExcelWriter(output_filename) as writer:
        best_per_sequence.to_excel(writer)


    paths_best_per_seq = [os.path.normpath(os.path.join(x, "../../")) for x in best_per_sequence['File'].values]
    paths_best_overall = [os.path.normpath(os.path.join(x, "../../")) for x in best_stats_keys.values()]

    paths = list(set(paths_best_per_seq).union(set(paths_best_overall)))
    paths.sort()

    print()
    print("The following directories contain some best run: ")
    print(paths)

if __name__ == '__main__':
    main()