#!/usr/bin/env python
import errno
import sys
import argparse
import os
from typing import List

import evo.core.trajectory
import numpy as np
import pandas as pd
from evo.core.trajectory import Trajectory
from matplotlib import pyplot as plt

from x_evaluate import trajectory_evaluation as te
from x_evaluate.plots import time_series_plot, boxplot_compare, ProgressPlotContextManager, PlotContext, DEFAULT_COLORS
import x_evaluate.plots
from x_evaluate.utils import convert_to_evo_trajectory, n_to_grid_size, timestamp_to_rosbag_time_zero, \
    name_to_identifier


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


class QualitativeEvaluationRun:
    def __init__(self, dataset_name, eval_run_name, trajectory: Trajectory, imu_bias: pd.DataFrame,
                 features: pd.DataFrame, realtime: pd.DataFrame):
        self.trajectory = trajectory
        self.dataset_name = dataset_name
        self.eval_run_name = eval_run_name
        self.imu_bias = imu_bias
        self.features = features
        self.realtime = realtime

        # print("UEHILA")

        if np.all(self.realtime['ts_real'] == 0):
            # HACKY fix of missing realtime timestamp (from missing easyprofiler library)
            ts_lowest = self.features['ts'].min()
            ts_highest = self.features['ts'].max()

            sim_lowest = self.realtime['t_sim'].min()
            sim_highest = self.realtime['t_sim'].max()

            approximate_timestamps = np.interp(self.realtime['t_sim'], [sim_lowest, sim_highest],
                                               [ts_lowest, ts_highest])

            self.realtime['ts_real'] = approximate_timestamps


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
            features_file = os.path.join(current_dir, dataset_dir, "features.csv")
            realtime_file = os.path.join(current_dir, dataset_dir, "realtime.csv")
            df_imu_bias = pd.read_csv(imu_file, delimiter=";")
            df_features = pd.read_csv(features_file, delimiter=";")
            df_realtime = pd.read_csv(realtime_file, delimiter=";")
            #print(pose_file)
            df_poses = pd.read_csv(pose_file, delimiter=";")
            traj_est, _ = convert_to_evo_trajectory(df_poses, prefix="estimated_")
            if dataset_dir not in data_dict.keys():
                data_dict[dataset_dir] = dict()
            data_dict[dataset_dir][directory] = QualitativeEvaluationRun(dataset_dir, directory, traj_est,
                                                              df_imu_bias, df_features, df_realtime)

    ####################################################################################################################
    ################################################### PAPER PLOTS ####################################################
    ####################################################################################################################

    try:
        klt_vio_run: QualitativeEvaluationRun = data_dict['001_wells_test_5']['000-xvio-framebased']
        eklt_vio_run: QualitativeEvaluationRun = data_dict['001_wells_test_5']['001-eklt-baseline']

        klt_vio_run.eval_run_name = "KLT-VIO"
        eklt_vio_run.eval_run_name = "EKLT-VIO"
        klt_vio_run.dataset_name = "Wells Cave Exit"
        eklt_vio_run.dataset_name = "Wells Cave Exit"

        plot_paper_comparison_plots_wells([klt_vio_run, eklt_vio_run], output_folder)

        return
    except KeyError:
        print("Tried doing paper plots, but didn't get the right keys")

    try:
        # klt_vio_run: QualitativeEvaluationRun = data_dict['001_mars_yard']['009-p-33']
        eklt_vio_run: QualitativeEvaluationRun = data_dict['001_mars_yard']['009-p-33']

        # klt_vio_run.eval_run_name = "KLT-VIO"
        eklt_vio_run.eval_run_name = "EKLT-VIO"
        # klt_vio_run.dataset_name = "Mars Yard"
        eklt_vio_run.dataset_name = "Mars Yard"

        # plot_paper_comparison_plots_wells([klt_vio_run, eklt_vio_run], output_folder)
        plot_paper_comparison_mars_yard([eklt_vio_run], output_folder)

        return
    except KeyError:
        print("Tried doing paper plots, but didn't get the right keys")

    ####################################################################################################################
    ################################################# OVERVIEW PLOTS  ##################################################
    ####################################################################################################################

    # dry run with dummy plot context
    manager = ProgressPlotContextManager()
    plot_qualitative_comparison_plots(data_dict, output_folder, manager.dummy_plot_context)

    print()
    print(F"Creating {manager.count} qualitative comparison plots")
    print()

    manager.init_progress_bar()

    # actual run with displaying progress
    plot_qualitative_comparison_plots(data_dict, output_folder, manager.actual_plot_context)


def plot_paper_comparison_plots_wells(eval_runs: List[QualitativeEvaluationRun], output_folder):
    # slice to desired region
    lim_t = [0, 25]

    x_evaluate.plots.use_paper_style_plots = True
    dataset, distances_from_origin, distances_from_origin_plus_one, num_slam_features,\
        p_x, p_y, p_z, distances, distances_plus_one, run_names, t, t_f = extract_and_slice_data(eval_runs, lim_t)

    lim_y = [-1, 5]
    lim_z = [-1, 5]

    with PlotContext(os.path.join(output_folder, F"{name_to_identifier(dataset)}_yz_plot")) as pc:
        time_series_plot(pc, p_y, p_z, run_names, xlabel="y [m]", ylabel="z [m]", xlim=lim_y, ylim=lim_z,
                         axis_equal=True)

    with PlotContext(os.path.join(output_folder, F"{name_to_identifier(dataset)}_distance")) as pc:
        time_series_plot(pc, t, distances_from_origin, run_names, ylabel="Distance from origin [m]", ylim=[0, 5],
                         xlim=lim_t)

    with PlotContext(os.path.join(output_folder, F"{name_to_identifier(dataset)}_distance_log")) as pc:
        ax = time_series_plot(pc, t, distances_from_origin_plus_one, run_names, ylabel="Distance from origin [m]",
                         use_log=True, xlim=lim_t)
        ax.axvspan(0, 15, alpha=0.1, facecolor="gray")
        ax.text(11, 2.2, "inside", style='italic')
        ax.text(10.2, 1.5, "(low-light)", style='italic')
        ax.text(15.5, 2.2, "entrance", style='italic')
        ax.text(15.9, 1.5, "(HDR)", style='italic')

    with PlotContext(os.path.join(output_folder, F"{name_to_identifier(dataset)}_num_features")) as pc:
        ax = time_series_plot(pc, t_f, num_slam_features, run_names, ylabel="\\# of SLAM features", xlim=lim_t,
                              ylim=[0, 16])
        ax.axvspan(0, 15, alpha=0.1, facecolor="gray")
        ax.text(11, 4.5, "inside", style='italic')
        ax.text(10.2, 3, "(low-light)", style='italic')
        ax.text(15.5, 4.5, "entrance", style='italic')
        ax.text(15.9, 3, "(HDR)", style='italic')
    # with PlotContext(os.path.join(output_folder, F"{name_to_identifier(dataset)}_yz_plot")) as pc:
    #     time_series_plot(pc, p_y, p_z, run_names, xlabel="y [m]", ylabel="z [m]", xlim=[-5, 5], ylim=[-5, 5])
    #
    # with PlotContext(os.path.join(output_folder, F"{name_to_identifier(dataset)}_features_plot")) as pc:
    #     time_series_plot(pc, p_y, p_z, run_names, xlabel="y [m]", ylabel="z [m]", xlim=[-5, 5], ylim=[-5, 5])


def plot_paper_comparison_mars_yard(eval_runs: List[QualitativeEvaluationRun], output_folder):
    # slice to desired region
    lim_t = [0, 20]

    x_evaluate.plots.use_paper_style_plots = True
    dataset, distances_from_origin, distances_from_origin_plus_one, num_slam_features,\
        p_x, p_y, p_z, distances, distances_plus_one, run_names, t, t_f = extract_and_slice_data(eval_runs, lim_t)

    lim_x = [-5, 5]
    lim_y = [-5, 5]

    with PlotContext(os.path.join(output_folder, F"{name_to_identifier(dataset)}_xy_plot")) as pc:
        time_series_plot(pc, p_x, p_y, run_names, xlabel="x [m]", ylabel="y [m]", xlim=lim_x, ylim=lim_y,
                         axis_equal=True)


def extract_and_slice_data(eval_runs, lim_t):
    run_names = [e.eval_run_name for e in eval_runs]
    dataset_names = [e.dataset_name for e in eval_runs]
    assert len(set(dataset_names)) == 1, "Tried comparing runs on different sequences, not supported"
    dataset = dataset_names[0]
    t = [e.trajectory.timestamps - e.trajectory.timestamps[0] for e in eval_runs]
    p_x = [np.array(e.trajectory.positions_xyz[:, 0]) for e in eval_runs]
    p_y = [e.trajectory.positions_xyz[:, 1] for e in eval_runs]
    p_z = [e.trajectory.positions_xyz[:, 2] for e in eval_runs]
    distances = [e.trajectory.distances for e in eval_runs]
    distances_from_origin = [np.linalg.norm(e.trajectory.positions_xyz, axis=1) for e in eval_runs]
    distances_from_origin_plus_one = [np.linalg.norm(e.trajectory.positions_xyz, axis=1) + 1 for e in eval_runs]
    distances_plus_one = [e.trajectory.distances + 1 for e in eval_runs]
    num_slam_features = [e.features['num_slam_features'].to_numpy() for e in eval_runs]
    t_f = [timestamp_to_rosbag_time_zero(e.features['ts'].to_numpy(), e.realtime) for e in eval_runs]

    # SLICE
    all_ids = [np.where(np.logical_and(t_i >= lim_t[0], t_i <= lim_t[1]))[0] for t_i in t]
    all_ids_f = [np.where(np.logical_and(t_i >= lim_t[0], t_i <= lim_t[1]))[0] for t_i in t_f]
    t = [t[i][ids] for i, ids in enumerate(all_ids)]
    t_f = [t_f[i][ids] for i, ids in enumerate(all_ids_f)]
    p_x = [p_x[i][ids] for i, ids in enumerate(all_ids)]
    p_y = [p_y[i][ids] for i, ids in enumerate(all_ids)]
    p_z = [p_z[i][ids] for i, ids in enumerate(all_ids)]
    distances = [distances[i][ids] for i, ids in enumerate(all_ids)]
    distances_from_origin = [distances_from_origin[i][ids] for i, ids in enumerate(all_ids)]
    distances_from_origin_plus_one = [distances_from_origin_plus_one[i][ids] for i, ids in enumerate(all_ids)]
    num_slam_features = [num_slam_features[i][ids] for i, ids in enumerate(all_ids_f)]
    return dataset, distances_from_origin, distances_from_origin_plus_one, num_slam_features, p_x, p_y, p_z, \
           distances, distances_plus_one, run_names, t, t_f


def plot_qualitative_comparison_plots(data_dict, output_folder, PlotContext):
    t_max = 10
    x_max = 10
    y_max = 10
    z_max = 10
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
        t_f = [timestamp_to_rosbag_time_zero(eval_runs[run].features['ts'].to_numpy(), eval_runs[run].realtime)
               for run in eval_run_names]

        num_slam_features = [eval_runs[run].features['num_slam_features'].to_numpy() for run in eval_run_names]
        num_msckf_features = [eval_runs[run].features['num_msckf_features'].to_numpy() for run in eval_run_names]
        num_potential_features = [eval_runs[run].features['num_potential_features'].to_numpy() for run in
                                  eval_run_names]

        with PlotContext(os.path.join(output_folder, F"{dataset}_xyz_in_time"), subplot_cols=3) as pc:
            pc.figure.suptitle("Positions in time")
            time_series_plot(pc, t, p_x, eval_run_names, ylabel="x", ylim=[-x_max, x_max])
            time_series_plot(pc, t, p_y, eval_run_names, ylabel="y", ylim=[-y_max, y_max])
            time_series_plot(pc, t, p_z, eval_run_names, ylabel="z", ylim=[-z_max, z_max])

        with PlotContext(os.path.join(output_folder, F"{dataset}_xyz"), subplot_cols=3) as pc:
            time_series_plot(pc, p_x, p_y, eval_run_names, xlabel="x [m]", ylabel="y [m]", axis_equal=True)
            time_series_plot(pc, p_x, p_z, eval_run_names, xlabel="x [m]", ylabel="z [m]", axis_equal=True)
            time_series_plot(pc, p_y, p_z, eval_run_names, xlabel="y [m]", ylabel="z [m]", axis_equal=True)

        with PlotContext(os.path.join(output_folder, F"{dataset}_xyz_lim"), subplot_cols=3) as pc:
            time_series_plot(pc, p_x, p_y, eval_run_names, xlabel="x [m]", ylabel="y [m]", axis_equal=True,
                             xlim=[-x_max, x_max], ylim=[-y_max, y_max])
            time_series_plot(pc, p_x, p_z, eval_run_names, xlabel="x [m]", ylabel="z [m]", axis_equal=True,
                             xlim=[-x_max, x_max], ylim=[-z_max, z_max])
            time_series_plot(pc, p_y, p_z, eval_run_names, xlabel="y [m]", ylabel="z [m]", axis_equal=True,
                             xlim=[-y_max, y_max], ylim=[-z_max, z_max])

        with PlotContext(os.path.join(output_folder, F"{dataset}_distance_in_time")) as pc:
            time_series_plot(pc, t, distances, eval_run_names, "Distance over time")

        with PlotContext(os.path.join(output_folder, F"{dataset}_num_features"), subplot_cols=3) as pc:
            time_series_plot(pc, t_f, num_slam_features, eval_run_names, ylabel="SLAM features")
            time_series_plot(pc, t_f, num_msckf_features, eval_run_names, ylabel="MSCKF features")
            time_series_plot(pc, t_f, num_potential_features, eval_run_names, ylabel="Potential features")

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
