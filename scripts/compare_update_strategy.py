import argparse
import glob
import os

import numpy as np
from matplotlib import pyplot as plt

from x_evaluate.comparisons import create_parameter_changes_table, identify_common_datasets
from x_evaluate.evaluation_data import FrontEnd
from x_evaluate.math_utils import calculate_velocities, moving_average_fixed_n, moving_average
from x_evaluate.plots import PlotContext, time_series_plot
from x_evaluate.scriptlets import find_evaluation_files_recursively, read_evaluation_pickle, cache


def main():
    parser = argparse.ArgumentParser(description="Takes all evaluation files in within the input directory tree and "
                                                 "analyzes ")
    parser.add_argument('--input_folder', type=str, required=True)

    args = parser.parse_args()

    evaluation_files = find_evaluation_files_recursively(args.input_folder)

    print(F"Found {evaluation_files}")

    def read_evaluations():

        evaluations = []

        for f in evaluation_files:
            e = read_evaluation_pickle(os.path.dirname(f), os.path.basename(f))

            if e.frontend is not FrontEnd.EKLT or ('2200 events' not in e.name and '3msec' not in e.name):
                continue

            print(F"Considering evaluation file '{e.name}'")

            evaluations.append(e)
        return evaluations

    print('reading...')
    evaluations = cache("/tmp/evaluations-speedup.pickle", read_evaluations)
    print('done')

    common_datasets = identify_common_datasets(evaluations)
    parameter_change_table = create_parameter_changes_table(evaluations, common_datasets)

    strategies = set(parameter_change_table.T['eklt_ekf_update_strategy'])

    dataset_choice = ["Boxes 6DOF", "Boxes Translation", "Dynamic 6DOF", "Dynamic Translation", "HDR Boxes",
                      "HDR Poster", "Poster 6DOF", "Poster Translation", "Shapes 6DOF", "Shapes Translation"]

    target_datasets = list(common_datasets.intersection(set(dataset_choice)))
    target_datasets.sort()

    PlotContext.FORMATS = [".pdf"]

    with PlotContext("trajectory-dynamics-plot", subplot_cols=2, subplot_rows=len(target_datasets)) as pc:

        for dataset in target_datasets:
            data = evaluations[0].data[dataset]

            t, lin_vel, ang_vel = calculate_velocities(data.trajectory_data.traj_gt)

            t -= t[0]

            lin_vel = np.linalg.norm(lin_vel, axis=1)
            ang_vel = np.linalg.norm(ang_vel, axis=1)

            # ax = pc.get_axis()
            time_windows = [5, 10, 15, 20, 25]
            t_s, lin_vels = zip(*tuple([moving_average(t, lin_vel, time_window) for time_window in time_windows]))
            time_series_plot(pc, list(t_s), lin_vels, [F"{tw}s average" for tw in time_windows],
                             F"Linear velocity norm '{dataset}'", "[m/s]")

            t_s, ang_vels = zip(*tuple([moving_average(t, ang_vel, time_window) for time_window in time_windows]))
            time_series_plot(pc, list(t_s), ang_vels, [F"{tw}s average" for tw in time_windows],
                             F"Angular velocity norm '{dataset}'", "[rad/s]")

    plt.show()

    print(data)


if __name__ == '__main__':
    main()
