import copy
import os
from typing import Collection, List

from evo.core.filters import FilterException
from evo.core.metrics import APE, RPE, PoseRelation

from x_evaluate.rpg_trajectory_evaluation import get_split_distances_on_equal_parts
from x_evaluate.utils import convert_to_evo_trajectory, rms, merge_tables, n_to_grid_size, get_nans_in_trajectory
from x_evaluate.plots import boxplot, time_series_plot, PlotType, PlotContext, boxplot_compare, barplot_compare, \
    DEFAULT_COLORS, align_yaxis, plot_evo_trajectory_with_euler_angles
from evo.core import sync
from evo.core import metrics
from evo.tools import plot
import pandas as pd
import numpy as np
import x_evaluate.rpg_trajectory_evaluation as rpg

from x_evaluate.evaluation_data import TrajectoryData, EvaluationDataSummary, EvaluationData, AlignmentType

POSE_RELATIONS = [metrics.PoseRelation.translation_part, metrics.PoseRelation.rotation_angle_deg]

APE_METRICS = [APE(p) for p in POSE_RELATIONS]

METRICS = APE_METRICS


def evaluate_trajectory(df_poses: pd.DataFrame, df_groundtruth: pd.DataFrame, df_imu_bias=None) -> \
        TrajectoryData:
    d = TrajectoryData()
    # filter invalid states
    if df_imu_bias is not None:
        d.imu_bias = df_imu_bias[df_imu_bias['t'] != -1]
    traj_est, d.raw_est_t_xyz_wxyz = convert_to_evo_trajectory(df_poses, prefix="estimated_")
    d.traj_gt, _ = convert_to_evo_trajectory(df_groundtruth)

    max_diff = 0.01
    d.traj_gt_synced, d.traj_est_synced = sync.associate_trajectories(d.traj_gt, traj_est, max_diff)

    d.traj_est_aligned = copy.deepcopy(d.traj_est_synced)

    d.alignment_type = AlignmentType.PosYaw
    d.alignment_frames = 5  # only temporary value meaning [3, 8] seconds
    rpg.rpg_align(d.traj_gt_synced, d.traj_est_aligned, d.alignment_type, use_subtrajectory=True)

    split_distances = get_split_distances_on_equal_parts(d.traj_gt, num_parts=5)
    # split_distances = np.hstack((split_distances, get_split_distances_equispaced(d.traj_gt, step_size=10)))

    d.rpe_error_r, d.rpe_error_t = calculate_rpe_errors_for_pairs_at_different_distances(split_distances,
                                                                                         d.traj_gt_synced,
                                                                                         d.traj_est_aligned)

    for m in METRICS:
        m.process_data((d.traj_gt_synced, d.traj_est_aligned))
        result = m.get_result()
        d.ate_errors[str(m)] = result.np_arrays['error_array']

    return d


def calculate_rpe_errors_for_pairs_at_different_distances(distances, trajectory_1, trajectory_2):
    translational_errors = {}
    rotational_errors = {}
    for d in distances:
        try:
            m_t = RPE(PoseRelation.translation_part, d, metrics.Unit.meters, all_pairs=True)
            m_r = RPE(PoseRelation.rotation_angle_deg, d, metrics.Unit.meters, all_pairs=True)

            m_t.process_data((trajectory_1, trajectory_2))
            m_r.process_data((trajectory_1, trajectory_2))

            translational_errors[d] = m_t.get_result().np_arrays['error_array']
            rotational_errors[d] = m_r.get_result().np_arrays['error_array']

        except FilterException:
            translational_errors[d] = np.ndarray([])
            rotational_errors[d] = np.ndarray([])
    return translational_errors, rotational_errors


def get_relative_errors_wrt_traveled_dist(s: EvaluationDataSummary):
    pos_errors = dict()
    rot_errors = dict()
    yaw_errors = dict()
    for k, d in s.data.items():
        if d.trajectory_data is None:
            continue
        position_metric = metrics.APE(metrics.PoseRelation.translation_part)
        orientation_metric = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
        # yaw_metric = metrics.APE(metrics.PoseRelation.yaw_rotation_deg)

        # yaw_metric.process_data((d.trajectory_data.traj_gt_synced, d.trajectory_data.traj_est_aligned))
        # result = yaw_metric.get_result()

        pos_error = np.mean(d.trajectory_data.ate_errors[str(position_metric)]) / d.trajectory_data.traj_gt.distances[-1]
        deg_error = np.mean(d.trajectory_data.ate_errors[str(orientation_metric)]) / d.trajectory_data.traj_gt.distances[-1]
        # yaw_error = np.mean(result.np_arrays['error_array']) / d.trajectory_data.traj_gt.distances[-1]

        pos_error = np.round(np.mean(pos_error*100), 2)  # in percent
        deg_error = np.round(np.mean(deg_error), 2)
        # yaw_error = np.round(np.mean(yaw_error), 2)
        pos_errors[k] = pos_error
        rot_errors[k] = deg_error
        # yaw_errors[k] = yaw_error
        yaw_errors[k] = -1.0

    return pos_errors, rot_errors, yaw_errors


def create_trajectory_result_table_wrt_traveled_dist(s: EvaluationDataSummary):
    pos_errors, rot_errors, yaw_errors = get_relative_errors_wrt_traveled_dist(s)

    data = np.empty((len(pos_errors), 3), dtype=np.float)
    i = 0
    for k, v in pos_errors.items():
        data[i, :] = [v, rot_errors[k], yaw_errors[k]]
        i += 1

    index_columns = [(s.name, "Mean Position Error [%]"), (s.name, "Mean Rotation error [deg/m]"),
                     (s.name, "Mean Yaw error [deg/m]")]
    index = pd.MultiIndex.from_tuples(index_columns, names=["Evaluation Run", "Metric"])
    result_table = pd.DataFrame(data, index=pos_errors.keys(), columns=index)

    return result_table


def compare_trajectory_performance_wrt_traveled_dist(summaries: List[EvaluationDataSummary]) -> pd.DataFrame:
    tables = [create_trajectory_result_table_wrt_traveled_dist(s) for s in summaries]
    return merge_tables(tables)


def create_trajectory_completion_table(s: EvaluationDataSummary):
    index_columns = [(s.name, "Completion rate [%]"), (s.name, "NaNs in estimate [%]"), (s.name, "GT trajectory "
                                                                                                 "length [m]")]
    index = pd.MultiIndex.from_tuples(index_columns, names=["Evaluation Run", "Metric"])

    data = np.empty((len(s.data.items()), 3), dtype=np.float)
    i = 0
    for k, v in s.data.items():
        gt_length = v.trajectory_data.traj_gt.timestamps[-1] - v.trajectory_data.traj_gt.timestamps[0]
        traveled_distance = v.trajectory_data.traj_gt.distances[-1]
        estimation_length = v.trajectory_data.traj_est_synced.timestamps[-1] - \
                            v.trajectory_data.traj_est_synced.timestamps[0]
        completion_percentage = estimation_length / gt_length * 100
        nans_percentage, _ = get_nans_in_trajectory(v.trajectory_data.raw_est_t_xyz_wxyz)
        data[i, :] = np.round([completion_percentage, nans_percentage, traveled_distance], 1)
        i += 1

    df = pd.DataFrame(data, index=s.data.keys(), columns=index)
    return df


def compare_trajectory_completion_rates(summaries: List[EvaluationDataSummary]) -> pd.DataFrame:
    tables = [create_trajectory_completion_table(s) for s in summaries]
    return merge_tables(tables)


def plot_trajectory_comparison_overview(pc: PlotContext, summary_table: pd.DataFrame, use_log=False):
    pose_column = 'Mean Position Error [%]'
    rot_column = 'Mean Rotation error [deg/m]'
    pos_table = summary_table.xs(pose_column, axis=1, level=1, drop_level=True)
    rot_table = summary_table.xs(rot_column, axis=1, level=1, drop_level=True)

    ax = pc.get_axis()
    ax.set_title("Average absolute pose errors normalized by traveled distance")
    ax.set_xlabel(pos_table.columns.name)
    ax.set_ylabel(F"Errors w.r.t traveled distance [\\%, deg/m]")
    if use_log:
        ax.set_yscale('log')
    evaluation_run_names = pos_table.columns.values
    labels = ['Mean Position Error [\\%]', 'Mean Rotation error [deg/m]']
    # do boxplots

    data = []

    if len(summary_table) < 5:
        # simply plot averages using barplots
        data.append([np.mean(pos_table[c].to_numpy()) for c in pos_table.columns])
        data.append([np.mean(rot_table[c].to_numpy()) for c in rot_table.columns])
        barplot_compare(ax, evaluation_run_names, data, labels)
    else:
        data.append([pos_table[c].to_numpy() for c in pos_table.columns])
        data.append([rot_table[c].to_numpy() for c in rot_table.columns])
        # for c in pos_table.columns:
        #     data.append([pos_table[c].to_numpy(), rot_table[c].to_numpy()])

        boxplot_compare(ax, evaluation_run_names, data, labels)


def create_absolute_trajectory_result_table(s: EvaluationDataSummary) -> pd.DataFrame:
    stats = {'RMS': lambda x: rms(x)}
    stats = {'MAX': lambda x: np.max(x)}
    columns = ["Name"] + [F"{str(m)} [{k}]" for m in METRICS for k in stats.keys()]

    result_table = pd.DataFrame(columns=columns)

    def add_result_row(name, errors):
        results = []
        for m in METRICS:
            for l in stats.values():
                array = errors[str(m)]
                results += [l(array)]

        result_row = [name] + results
        result_table.loc[len(result_table)] = result_row

    for d in s.data.values():
        if d.trajectory_data is None:
            continue
        add_result_row(d.name, d.trajectory_data.ate_errors)

    overall_errors = dict()

    is_empty = False

    for m in METRICS:
        overall_errors[str(m)] = combine_error(s.data.values(), str(m))
        if len(overall_errors[str(m)]) <= 0:
            is_empty = True
            break

    if not is_empty:
        add_result_row("OVERALL", overall_errors)

    return result_table


def combine_error(evaluations: Collection[EvaluationData], error_key) -> np.ndarray:
    arrays = []
    for d in evaluations:
        if d.trajectory_data is not None:
            arrays.append(d.trajectory_data.ate_errors[error_key])
    if len(arrays) > 0:
        return np.hstack(tuple(arrays))
    return []


def plot_rpg_error_arrays(pc: PlotContext, trajectories: List[TrajectoryData], labels, use_log=False,
                          realtive_to_trav_dist=False, desired_distances=None):
    errors_rot_deg = []
    errors_trans_m = []
    if len(trajectories) < 1:
        return
    gt_trajectory = trajectories[0].traj_gt

    distances = []
    if desired_distances is None:
        distances = get_split_distances_on_equal_parts(gt_trajectory, 5)
    else:
        distances = np.array(desired_distances)

    for t in trajectories:
        errors_rot_deg.append(t.rpe_error_r)
        errors_trans_m.append(t.rpe_error_t)

        missing_distances = [d for d in distances if d not in t.rpe_error_t.keys() or d not in t.rpe_error_r.keys()]

        if len(missing_distances) > 0:
            print(F"WARNING: For the following pose-pair distances the RPE error is missing: {missing_distances}")
            print("          Please use 'customize_rpe_error_arrays' script to calculate them")

    is_degenerate = (len(errors_rot_deg) <= 0) or (len(errors_trans_m) <= 0)
    # this happens e.g. on calibration, where there are no pairs over 10m and you ask for 10, 20, 30, 40
    is_degenerate = is_degenerate or np.any([np.all([np.size(e_m[d]) <= 1 for d in distances]) for e_m in
                                            errors_trans_m])
    is_degenerate = is_degenerate or np.any([np.all([np.size(e_rot[d]) <= 1 for d in distances]) for e_rot in
                                             errors_rot_deg])

    if is_degenerate:
        return

    if realtive_to_trav_dist:
        data_trans_m = [[e[k]/(0.01*k) for k in distances] for e in errors_trans_m]
    else:
        data_trans_m = [[e[k] for k in distances] for e in errors_trans_m]

    ax = pc.get_axis()
    ax.set_xlabel("Distance traveled [m]")
    if realtive_to_trav_dist:
        ax.set_ylabel(F"Translation error [\\%]")
    else:
        ax.set_ylabel(F"Translation error [m]")
    if use_log:
        ax.set_yscale('log')
    boxplot_compare(ax, distances, data_trans_m, labels, showfliers=False, legend=False)  # legend only on rotation plot

    ax = pc.get_axis()

    ax.set_xlabel("Distance traveled [m]")
    if realtive_to_trav_dist:
        ax.set_ylabel(F"Rotation error [deg/m]")
    else:
        ax.set_ylabel(F"Rotation error [deg]")
    if use_log:
        ax.set_yscale('log')
    if realtive_to_trav_dist:
        data_rot_deg = [[e[k]/k for k in distances] for e in errors_rot_deg]
    else:
        data_rot_deg = [[e[k] for k in distances] for e in errors_rot_deg]
    boxplot_compare(ax, distances, data_rot_deg, labels, showfliers=False)


def plot_imu_bias(pc: PlotContext, trajectory_data: TrajectoryData, name):
    df = trajectory_data.imu_bias
    # filter invalid states
    df = df[df['t'] != -1]
    t = df['t'].to_numpy()
    t = t - t[0]

    if "sigma_b_w_x" in df.columns.values:
        labels = ["b_a_x", "b_a_y", "b_a_z"]
        b_a_xyz = df[labels].to_numpy().T
        sigma_b_a = df[["sigma_b_a_x", "sigma_b_a_y", "sigma_b_a_z"]].to_numpy().T
        sigma_b_a_lower = b_a_xyz - np.sqrt(sigma_b_a)
        sigma_b_a_upper = b_a_xyz + np.sqrt(sigma_b_a)
        time_series_plot(pc, t, list(b_a_xyz), labels, F"Accelerometer bias on '{name}'", "m/s^2",
                         shaded_area_lower=list(sigma_b_a_lower), shaded_area_upper=list(sigma_b_a_upper))
        labels = ["b_w_x", "b_w_y", "b_w_z"]
        b_w_xyz = df[labels].to_numpy().T
        sigma_b_w = df[["sigma_b_w_x", "sigma_b_w_y", "sigma_b_w_z"]].to_numpy().T
        sigma_b_w_lower = b_w_xyz - np.sqrt(sigma_b_w)
        sigma_b_w_upper = b_w_xyz + np.sqrt(sigma_b_w)
        time_series_plot(pc, t, list(b_w_xyz), labels, F"Gyroscope bias on '{name}'", "rad/s",
                         shaded_area_lower=list(sigma_b_w_lower), shaded_area_upper=list(sigma_b_w_upper))
    else:
        labels = ["b_a_x", "b_a_y", "b_a_z"]
        b_a_xyz = list(df[labels].to_numpy().T)
        time_series_plot(pc, t, b_a_xyz, labels, F"Accelerometer bias on '{name}'", "m/s^2")
        labels = ["b_w_x", "b_w_y", "b_w_z"]
        b_w_xyz = list(df[labels].to_numpy().T)
        time_series_plot(pc, t, b_w_xyz, labels, F"Gyroscope bias on '{name}'", "rad/s")


def plot_imu_bias_in_one(pc: PlotContext, eval_data: EvaluationData, eval_name):
    df = eval_data.trajectory_data.imu_bias
    df = df[df['t'] != -1]
    t = df['t'].to_numpy()
    t = t - t[0]

    labels = ["$b_{a_x}$", "$b_{a_y}$", "$b_{a_z}$", "$b_{w_x}$", "$b_{w_y}$", "$b_{w_z}$"]
    data = list(df[["b_a_x", "b_a_y", "b_a_z", "b_w_x", "b_w_y", "b_w_z"]].to_numpy().T)

    ax = pc.get_axis()
    ax_right = ax.twinx()

    lines = None
    for i in range(len(data)):
        if i < 3:
            line = ax.plot(t, data[i], label=labels[i], color=DEFAULT_COLORS[i])
        else:
            line = ax_right.plot(t, data[i], label=labels[i], color=DEFAULT_COLORS[i], linestyle='--')
        if not lines:
            lines = line
        else:
            lines = lines + line

    align_yaxis(ax_right, ax)

    # https://stackoverflow.com/a/5487005
    ax_right.legend(lines, labels)
    ax.set_title(F"Gyroscope and accelerometer bias on '{eval_data.name}' ({eval_name})")
    ax.set_xlabel("$t [s]$")
    ax.set_ylabel("$m/s^2$")
    ax_right.set_ylabel("$rad/s$")


def plot_trajectory_plots(trajectory_data: TrajectoryData, name, output_folder):

    if trajectory_data is None:
        return

    if hasattr(trajectory_data, 'imu_bias') and trajectory_data.imu_bias is not None:
        with PlotContext(os.path.join(output_folder, "imu_bias"), subplot_cols=2) as pc:
            plot_imu_bias(pc, trajectory_data, name)

    with PlotContext(os.path.join(output_folder, "xy_plot")) as pc:
        plot_trajectory(pc, [trajectory_data], [name], use_aligned=False)

    with PlotContext(os.path.join(output_folder, "xy_plot_aligned")) as pc:
        plot_trajectory(pc, [trajectory_data], [name], use_aligned=True)

    with PlotContext(os.path.join(output_folder, "trajectory_plot"), subplot_rows=3, subplot_cols=3) as pc:
        plot_trajectory_with_gt_and_euler_angles(pc, trajectory_data, name, use_aligned=False)

    with PlotContext(os.path.join(output_folder, "trajectory_plot_aligned"), subplot_rows=3, subplot_cols=3) as pc:
        plot_trajectory_with_gt_and_euler_angles(pc, trajectory_data, name, use_aligned=True)

    with PlotContext(os.path.join(output_folder, "rpg_subtrajectory_errors"), subplot_cols=2) as pc:
        plot_rpg_error_arrays(pc, [trajectory_data], [name], realtive_to_trav_dist=True)


def create_summary_info(summary: EvaluationDataSummary, output_folder):
    summary.trajectory_summary_table = create_absolute_trajectory_result_table(summary)

    table = create_trajectory_result_table_wrt_traveled_dist(summary)
    with pd.ExcelWriter(os.path.join(output_folder, "trajectory_tracking_summary.xlsx")) as writer:
        table.to_excel(writer)


def plot_trajectory(pc: PlotContext, trajectories: Collection[TrajectoryData], labels, use_aligned=True):
    traj_by_label = dict()

    if len(trajectories) <= 0:
        return

    first = True

    for i, t in enumerate(trajectories):
        if first:
            first = False
            traj_by_label[F"{labels[i]} reference"] = t.traj_gt_synced
        if use_aligned:
            traj_by_label[F"{labels[i]} estimate"] = t.traj_est_aligned
        else:
            traj_by_label[F"{labels[i]} estimate"] = t.traj_est_synced

    plot.trajectories(pc.figure, traj_by_label, plot.PlotMode.xy)


def plot_trajectory_with_gt_and_euler_angles(pc: PlotContext, trajectory: TrajectoryData, label, use_aligned=True):
    if use_aligned:
        plot_evo_trajectory_with_euler_angles(pc, trajectory.traj_est_aligned, label, trajectory.traj_gt_synced)
    else:
        plot_evo_trajectory_with_euler_angles(pc, trajectory.traj_est_synced, label, trajectory.traj_gt_synced)


def plot_summary_plots(summary: EvaluationDataSummary, output_folder):
    with PlotContext(os.path.join(output_folder, "ate_boxplot"), subplot_cols=2) as pc:
        plot_ape_error_comparison(pc, summary.data.values(), PlotType.BOXPLOT)
    with PlotContext(os.path.join(output_folder, "ate_in_time"), subplot_cols=2) as pc:
        plot_ape_error_comparison(pc, summary.data.values(), PlotType.TIME_SERIES)

    has_imu_bias = [hasattr(e.trajectory_data, 'imu_bias') and e.trajectory_data.imu_bias is not None for e in
                    summary.data.values()]
    has_imu_bias = np.all(has_imu_bias)

    rows, cols = n_to_grid_size(len(summary.data.values()))

    if has_imu_bias:
        with PlotContext(os.path.join(output_folder, F"imu_bias"), subplot_rows=rows, subplot_cols=cols) as pc:
            for e in summary.data.values():
                plot_imu_bias_in_one(pc, e, summary.name)


def plot_ape_error_comparison(pc: PlotContext, evaluations: Collection[EvaluationData],
                              plot_type: PlotType = PlotType.BOXPLOT, labels=None, use_log=False):
    translation_metric = metrics.APE(PoseRelation.translation_part)
    rotation_metric = metrics.APE(PoseRelation.rotation_angle_deg)

    auto_labels = []
    rotation_data = []
    translation_data = []
    time_arrays = []
    for e in evaluations:
        if e.trajectory_data is not None:
            t = e.trajectory_data.traj_est_synced.timestamps.flatten()

            translation_data.append(e.trajectory_data.ate_errors[str(translation_metric)])
            rotation_data.append(e.trajectory_data.ate_errors[str(rotation_metric)])
            time_arrays.append(t - t[0])
            auto_labels.append(e.name)

    if len(auto_labels) <= 0:
        return

    if labels is None:
        labels = auto_labels

    if plot_type == PlotType.BOXPLOT:
        boxplot(pc, translation_data, labels, "ATE w.r.t. translation [m]", use_log=use_log)
        boxplot(pc, rotation_data, labels, "ATE w.r.t. rotation [deg]", use_log=use_log)
    elif plot_type == PlotType.TIME_SERIES:
        time_series_plot(pc, time_arrays, translation_data, labels, title="ATE w.r.t. translation in time",
                         ylabel="translation error [m]", use_log=use_log)
        time_series_plot(pc, time_arrays, rotation_data, labels, title="ATE w.r.t. rotation in time",
                         ylabel="rotation error [deg]", use_log=use_log)
    else:
        raise ValueError(F"Invalid plot type '{plot_type}'")
