import argparse
import glob
import os
import pickle
from enum import Enum

import numpy as np
import pandas as pd


class Dataset(Enum):
    RPG_DAVIS = 'RPG_DAVIS'
    RPG_DAVIS_ROTATION = 'RPG_DAVIS_ROTATION'
    RPG_FPV = 'RPG_FPV'
    SIM = 'SIM'


def main():
    parser = argparse.ArgumentParser(description="Picks the best tables from a filename->result table map")
    parser.add_argument('--input_folder', type=str, required=True)

    args = parser.parse_args()

    matches = glob.glob(os.path.join(args.input_folder, "*tables.pickle"))
    matches.sort()

    for input_file in matches:
        if 'haste-ral' not in input_file:
            continue

        print(F"######################################################################################################")
        print(F"# Scanning {input_file} #")
        print(F"######################################################################################################")

        with open(input_file, 'rb') as f:
            tables = pickle.load(f)

        dataset = Dataset.RPG_DAVIS

        if 'rotation' in input_file:
            dataset = Dataset.RPG_DAVIS_ROTATION
        elif 'fpv' in input_file:
            dataset = Dataset.RPG_FPV
        elif 'sim' in input_file:
            dataset = Dataset.SIM

        # print(tables)

        dataset_sequences = {
            Dataset.RPG_DAVIS: ["Boxes 6DOF", "Boxes Translation", "Dynamic 6DOF", "Dynamic Translation", "HDR Boxes",
                                "HDR Poster", "Poster 6DOF", "Poster Translation", "Shapes 6DOF", "Shapes Translation"],
            Dataset.RPG_DAVIS_ROTATION: ["Boxes Rotation", "Dynamic Rotation", "Poster Rotation", "Shapes Rotation"],
            Dataset.RPG_FPV: ['Indoor 45 Seq 12', 'Indoor 45 Seq 2'],
            # Dataset.SIM: ["Mars Straight Vmax 3.2 Offset 2.5", "Mars Eight Vmax 3.5 Offset 2.5",
            #               "Mars Circle Vmax 7.2 Offset 2.5", "Mars Vertical Circle Vmax 2.4 Offset 2.5",
            #               "Mars Straight Vmax 3.2 Offset 5", "Mars Eight Vmax 3.5 Offset 5",
            #               "Mars Circle Vmax 7.2 Offset 5", "Mars Vertical Circle Vmax 2.4 Offset 5",
            #               "Mars Straight Vmax 3.2 Offset 10", "Mars Eight Vmax 3.5 Offset 10",
            #               "Mars Circle Vmax 7.2 Offset 10", "Mars Vertical Circle Vmax 2.4 Offset 10"],
            Dataset.SIM: [
                          # "Mars Straight Vmax 3.2 Offset 2.5",
                          # "Mars Vertical Circle Vmax 2.4 Offset 2.5",
                          # "Mars Circle Vmax 7.2 Offset 2.5",
                          # "Mars Mellon Vmax 12.4 Offset 10",
                            "Mars Vertical Circle",
                            "Mars Circle",
                            "Mars Mellon",
                          # "Mars Circle Vmax 7.2 Offset 10",
                          # "Mars Circle Vmax 16.6 Offset 10"
            ]


        }

        sequence_lengths = {
            Dataset.RPG_DAVIS: [69.9, 65.2, 39.6, 30.1, 55.1, 55.4, 61.1, 49.3, 47.6, 56.1],
            Dataset.RPG_DAVIS_ROTATION: [14.9, 10.5, 16.9, 15.7],
            Dataset.RPG_FPV: [118.4, 176.2],
            # Dataset.SIM: [8.2, 49.5, 141.3, 30.8, 141.3, 397.8]
            # Dataset.SIM: [8.2, 141.3, 30.8, 141.3, 397.8]
            Dataset.SIM: [30.8, 141.3, 409.3]
            # Dataset.SIM: [30.8, 141.3]
            # Dataset.SIM: [8.2, 49.5, 141.3, 30.8, 8.2, 49.5, 141.3, 30.8, 8.2, 49.5, 141.3, 30.8],
        }

        sequences = dataset_sequences[dataset]
        expected_lengths = sequence_lengths[dataset]

        stats = {
            "max": lambda p: p.max(),
            "mean": lambda p: p.mean(),
            "median": lambda p: p.median(),
            "median+max": lambda p: p.median() + 0.25 * p.max(),
        }

        if dataset == Dataset.RPG_DAVIS:
            def get_direct_comparison(results):
                def better_than_uslam(p):
                    return len(results) - np.count_nonzero(p <= results)

                return better_than_uslam

            stats['uslam_global'] = get_direct_comparison([0.68, 1.12, 0.76, 0.63, 1.01, 1.48, 0.59, 0.24, 1.07, 1.36])
            stats['uslam_best'] = get_direct_comparison([0.30, 0.27, 0.19, 0.18, 0.37, 0.31, 0.28, 0.12, 0.10, 0.26])
            # stats['best_global'] = get_direct_comparison([0.68, 0.21, 0.54, 0.45, 0.36, 0.26, 0.33, 0.13, 0.54, 0.39])

            # best out of all except XVIO
            stats['best_global'] = get_direct_comparison([0.68, 0.59, 0.54, 0.45, 0.46, 0.26, 0.33, 0.21, 0.54, 0.39])

            # best out of all except EKLT
            # stats['best_global'] = get_direct_comparison([0.68, 0.21, 0.54, 0.63, 0.36, 0.66, 0.59, 0.13, 1.07, 0.54])

        best_per_sequence_tables = {s: pd.DataFrame(columns=["Mean Position Error [%]", "Mean Rotation error [deg/m]",
                                                             "Completion rate [%]", "File"]) for s in sequences}
        best_overall_stats_table = pd.DataFrame(columns=(list(stats.keys()) + ["File"]))

        #
        #
        # best_stats = {k: np.inf for k in stats.keys()}
        # best_stats_keys = {k: None for k in stats.keys()}
        #
        # best_per_sequence = pd.DataFrame.from_dict({"Sequence": sequences,
        #                                             "Mean Position Error [%]": [np.inf] * len(sequences),
        #                                             "Mean Rotation error [deg/m]": [np.inf] * len(sequences),
        #                                             "Completion rate [%]": [0] * len(sequences),
        #                                             "File": [""] * len(sequences)})
        #
        # best_per_sequence = best_per_sequence.set_index("Sequence")
        #
        # pos_error_values = {}
        # rot_error_values = {}

        # for s in sequences:
        #     pos_error_values[s] = []
        #     rot_error_values[s] = []

        for k, v in tables.items():
            table = v.droplevel(0, axis=1)
            if isinstance(table.columns, pd.MultiIndex):
                table = v.droplevel(0, axis=1)
            table = table.loc[sequences, :]

            # only process if GT trajectory length is within 1m of expected length
            if np.all(np.abs(table["GT trajectory length [m]"].to_numpy() - expected_lengths) <= 1) \
                    and table["Completion rate [%]"].min() > 99:
                new_row = [func(table["Mean Position Error [%]"]) for func in stats.values()]
                new_row += [k]
                best_overall_stats_table.loc[len(best_overall_stats_table)] = new_row

            for i, s in enumerate(sequences):
                if np.abs(expected_lengths[i] - table.loc[s, "GT trajectory length [m]"]) > 1 \
                        or table.loc[s, "Completion rate [%]"] < 99:
                    continue

                best_per_sequence_tables[s].loc[len(best_per_sequence_tables[s])] = \
                    list(table.loc[s, ["Mean Position Error [%]", "Mean Rotation error [deg/m]",
                                       "Completion rate [%]"]]) + [k]

        # os.path.dirname(k)[26:]

        # pd.set_option('max_columns', None)
        # pd.set_option("max_colwidth", None)
        pd.options.display.max_colwidth = None
        pd.options.display.width = 0

        top_n = 7

        print("BEST PER SEQUENCE:")
        for s, t in best_per_sequence_tables.items():
            print(F"Top {top_n} for {s}")
            top = t.sort_values('Mean Position Error [%]').iloc[:top_n]
            top['Folder'] = top['File'].apply(lambda file: os.path.dirname(file)[26:])
            print(top[['Mean Position Error [%]', 'Mean Rotation error [deg/m]', 'Folder']])

        for s in stats.keys():
            print(F"\nTop {top_n} tables for {s}")
            top = best_overall_stats_table.sort_values(s).iloc[:top_n]
            top['Folder'] = top['File'].apply(lambda file: os.path.dirname(file)[26:])
            for index, row in top.iterrows():
                other_stats = {stat: row[stat] for stat in set(stats.keys()).difference({s})}
                print(F"\n{s} (={row[s]} table {row['Folder']}:   {other_stats}")
                table = tables[row['File']]
                if isinstance(table.columns, pd.MultiIndex):
                    table = table.droplevel(0, axis=1)
                print(table.loc[sequences, ["Mean Position Error [%]", "Mean Rotation error [deg/m]"]])

        # print(best_overall_stats_table)
        # print(best_per_sequence[["Mean Position Error [%]", "Mean Rotation error [deg/m]"]])

        # for s in sequences:
        #     index = np.argmin(pos_error_values[s])
        #     print(F"{s}: {pos_error_values[s][index]}  {rot_error_values[s][index]} from '{list(tables.keys())[index]}'")

        # for stat, table_key in best_stats_keys.items():
        #     print(F"\nBest {stat} (={best_stats[stat]} table {table_key}:")
        #     table = tables[table_key]
        #     if isinstance(table.columns, pd.MultiIndex):
        #         table = tables[table_key].droplevel(0, axis=1)
        #     # print(table[["Mean Position Error [%]", "Mean Rotation error [deg/m]", "GT trajectory length [m]"]])
        #     print(table[["Mean Position Error [%]", "Mean Rotation error [deg/m]"]])
        #
        # output_filename = os.path.join(os.path.dirname(input_file), os.path.basename(input_file)[:-7] + "-result.xlsx")
        # with pd.ExcelWriter(output_filename) as writer:
        #     best_per_sequence.to_excel(writer)
        #
        # paths_best_per_seq = [os.path.normpath(os.path.join(x, "../../")) for x in best_per_sequence['File'].values]
        # paths_best_overall = [os.path.normpath(os.path.join(x, "../../")) for x in best_stats_keys.values()]
        #
        # paths = list(set(paths_best_per_seq).union(set(paths_best_overall)))
        # paths.sort()
        #
        # print()
        # print("The following directories contain some best run: ")
        # print(paths)


if __name__ == '__main__':
    main()
