#!/usr/bin/env python
import errno
import sys
import argparse
import os

import evo.core.trajectory
import numpy as np
import pandas as pd
from evo.core.trajectory import Trajectory

from x_evaluate import trajectory_evaluation as te
from x_evaluate.plots import time_series_plot, boxplot_compare, ProgressPlotContextManager
from x_evaluate.utils import convert_to_evo_trajectory, n_to_grid_size


def parse_args():
    parser = argparse.ArgumentParser(
        prog='qualitative_analysis.py',
        description='Checks if parameter runs are successful or if they are diverging.')
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str)
    args = parser.parse_args()
    return args


def get_sub_dirs(input_dir):

    directories = os.listdir(input_dir)
    directories.sort()

    eval_dirs = []

    for directory in directories:
        if len(directory) > 3:
            test_str = directory[0:3]
            if test_str.isnumeric():
                eval_dirs.append(directory)

    return eval_dirs


def highlight_if_true(df, color = "green"):

    attr = 'background-color: {}'.format(color)
    df_bool = pd.DataFrame(df.apply(lambda x: [True if v == True else False for v in x],axis=1).apply(pd.Series),
                      index=df.index)
    df_bool.columns =df.columns
    return pd.DataFrame(np.where(df_bool, attr, ""),
                       index= df.index, columns=df.columns)


# Press the green button in the gutter to run the script.
def main():
    args = parse_args()

    output_folder = args.output_folder
    if output_folder is None:
        output_folder = os.path.join(args.input_folder, "results")
        print(F"Using 'output_folder' as output_folder")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.isdir(args.input_folder):
        print("Input directory not found! (" + args.input_folder + ")")
        exit(1)

    class EvaluationRun:
        def __init__(self, dataset_name, eval_run_name, trajectory: Trajectory, imu_bias: pd.DataFrame):
            self.trajectory = trajectory
            self.dataset_name = dataset_name
            self.eval_run_name = eval_run_name
            self.imu_bias = imu_bias

    eval_dirs = get_sub_dirs(args.input_folder)

    # all_trajectories = []
    dataset_dirs = get_sub_dirs(args.input_folder + "/" + eval_dirs[0])

    # contains { dataset: { eval_run: EvaluationRun() }}
    data_dict = dict()

    for directory in eval_dirs:
        current_dir = args.input_folder + "/" + directory
        for dataset_dir in dataset_dirs:
            pose_file = os.path.join(current_dir, dataset_dir, "pose.csv")
            imu_file = os.path.join(current_dir, dataset_dir, "imu_bias.csv")
            df_imu_bias = pd.read_csv(imu_file, delimiter=";")
            #print(pose_file)
            df_poses = pd.read_csv(pose_file, delimiter=";")
            traj_est, _ = convert_to_evo_trajectory(df_poses, prefix="estimated_")
            if dataset_dir not in data_dict.keys():
                data_dict[dataset_dir] = dict()
            data_dict[dataset_dir][directory] = EvaluationRun(dataset_dir, directory, traj_est, df_imu_bias)

    manager = ProgressPlotContextManager()

    # dry run with dummy plot context
    plot_qualitativ_comparison_plots(data_dict, output_folder, manager.dummy_plot_context)

    print()
    print(F"Creating {manager.count} qualitative comparison plots")
    print()

    manager.init_progress_bar()

    # actual run with displaying progress
    plot_qualitativ_comparison_plots(data_dict, output_folder, manager.actual_plot_context)


    # #success = np.zeros((len(dataset_dirs), len(all_pose_files)))
    # dist_max = np.inf * np.ones((len(dataset_dirs), len(all_trajectories)))
    # max_dist = 42.0
    #
    # for i, dataset in enumerate(dataset_dirs):
    #     for j, experiment in enumerate(all_trajectories):
    #         try:
    #             t = experiment[i]['t'].to_numpy()
    #             p_x = experiment[i]['estimated_p_x'].to_numpy()
    #             p_y = experiment[i]['estimated_p_y'].to_numpy()
    #             p_z = experiment[i]['estimated_p_z'].to_numpy()
    #             p_x = p_x[t != -1]
    #             p_y = p_y[t != -1]
    #             p_z = p_z[t != -1]
    #             t = t[t != -1]
    #             max_xx = np.max(p_x**2)
    #             max_yy = np.max(p_y**2)
    #             max_zz = np.max(p_z**2)
    #             dist_max[i, j] = np.sqrt(max_xx + max_yy + max_zz)
    #         except:
    #             dist_max[i, j] = np.inf
    # #print(pd.DataFrame(dist_max, dataset_dirs, eval_dirs))
    # success = dist_max < max_dist
    # results = pd.DataFrame(success, dataset_dirs, eval_dirs)
    # results.style. \
    #     apply(highlight_if_true, axis=None). \
    #     to_excel(args.output_dir + '/' + 'results.xlsx', engine="openpyxl")
    #
    # victory = False
    # for column in results:
    #     victory = all(results[column] == True)
    #     if victory:
    #         print("Run " + column + " was successful on all datasets!!")


def plot_qualitativ_comparison_plots(data_dict, output_folder, PlotContext):
    t_max = 10
    x_max = 30
    y_max = 30
    z_max = 3
    limit = 2
    ## Make xyz-t-plot
    for dataset, eval_runs in data_dict.items():
        eval_run_names = list(eval_runs.keys())
        rows, cols = n_to_grid_size(len(eval_run_names))

        t = [eval_runs[run].trajectory.timestamps - eval_runs[run].trajectory.timestamps[0] for run in eval_run_names]
        p_x = [eval_runs[run].trajectory.positions_xyz[:, 0] for run in eval_run_names]
        p_y = [eval_runs[run].trajectory.positions_xyz[:, 1] for run in eval_run_names]
        p_z = [eval_runs[run].trajectory.positions_xyz[:, 2] for run in eval_run_names]
        distances = [eval_runs[run].trajectory.distances for run in eval_run_names]
        distances_plus_one = [eval_runs[run].trajectory.distances + 1 for run in eval_run_names]

        with PlotContext(os.path.join(output_folder, F"{dataset}_xyz_in_time"), subplot_cols=3) as pc:
            pc.figure.suptitle("Positions in time")
            time_series_plot(pc, t, p_x, eval_run_names, ylabel="x", ylim=[-x_max, x_max])
            time_series_plot(pc, t, p_y, eval_run_names, ylabel="y", ylim=[-y_max, y_max])
            time_series_plot(pc, t, p_z, eval_run_names, ylabel="z", ylim=[-z_max, z_max])

        with PlotContext(os.path.join(output_folder, F"{dataset}_xyz"), subplot_cols=3) as pc:
            time_series_plot(pc, p_x, p_y, eval_run_names, xlabel="x [m]", ylabel="y [m]", axis_equal=True)
            time_series_plot(pc, p_x, p_z, eval_run_names, xlabel="x [m]", ylabel="z [m]", axis_equal=True)
            time_series_plot(pc, p_y, p_z, eval_run_names, xlabel="y [m]", ylabel="z [m]", axis_equal=True)

        with PlotContext(os.path.join(output_folder, F"{dataset}_distance_in_time")) as pc:
            time_series_plot(pc, t, distances, eval_run_names, "Distance over time")

        with PlotContext(os.path.join(output_folder, F"{dataset}_distance_in_time_log")) as pc:
            time_series_plot(pc, t, distances_plus_one, eval_run_names, "Distance over time", use_log=True)

        with PlotContext(os.path.join(output_folder, F"{dataset}_imu_bias"), subplot_rows=rows,
                         subplot_cols=cols) as pc:
            for run in eval_run_names:
                te.plot_imu_bias_in_one(pc, eval_runs[run].imu_bias, dataset, run)

    # Assumes each dataset contains same evaluation runs...
    dataset_labels = list(data_dict.keys())
    eval_run_names = list(data_dict[dataset_labels[0]].keys())
    data = [[data_dict[k][run].trajectory.distances for k in dataset_labels] for run in eval_run_names]

    scaled_with_runs = max(10 * len(eval_run_names) / 6, 10)

    with PlotContext(os.path.join(output_folder, F"boxplot_distances"), base_width_inch=scaled_with_runs*2.0) as pc:
        boxplot_compare(pc.get_axis(), dataset_labels, data, eval_run_names, ylabel="Distances",
                        title="Distances")

    with PlotContext(os.path.join(output_folder, F"boxplot_distances_log"), base_width_inch=scaled_with_runs*2.0) as pc:
        boxplot_compare(pc.get_axis(), dataset_labels, data, eval_run_names, ylabel="Distances",
                        title="Distances", use_log=True)


if __name__ == '__main__':
    main()
