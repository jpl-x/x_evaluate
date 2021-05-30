import copy
import os
from typing import Collection

from evo.core.filters import FilterException
from evo.core.metrics import APE, RPE, PoseRelation, PE

from x_evaluate.rpg_trajectory_evaluation import get_split_distances_on_equal_parts, \
    get_split_distances_equispaced
from x_evaluate.utils import convert_to_evo_trajectory, rms, name_to_identifier
from x_evaluate.plots import boxplot, time_series_plot, PlotType, PlotContext, boxplot_compare
from evo.core import sync
from evo.core import metrics
from evo.tools import plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import x_evaluate.rpg_trajectory_evaluation as rpg

from x_evaluate.evaluation_data import TrajectoryData, EvaluationDataSummary, EvaluationData, AlignmentType, \
    DistributionSummary

POSE_RELATIONS = [metrics.PoseRelation.translation_part, metrics.PoseRelation.rotation_angle_deg]

# POSE_RELATIONS = [metrics.PoseRelation.full_transformation]

APE_METRICS = [APE(p) for p in POSE_RELATIONS]

METRICS = APE_METRICS


def evaluate_trajectory(df_poses: pd.DataFrame, df_groundtruth: pd.DataFrame) -> TrajectoryData:
    d = TrajectoryData()
    traj_est, d.raw_est_t_xyz_wxyz = convert_to_evo_trajectory(df_poses, prefix="estimated_")
    d.traj_gt, _ = convert_to_evo_trajectory(df_groundtruth)

    max_diff = 0.01
    d.traj_gt_synced, d.traj_est_synced = sync.associate_trajectories(d.traj_gt, traj_est, max_diff)

    d.traj_est_aligned = copy.deepcopy(d.traj_est_synced)

    d.alignment_type = AlignmentType.PosYaw
    d.alignment_frames = -1
    rpg.rpg_align(d.traj_gt_synced, d.traj_est_aligned, d.alignment_type)

    split_distances = get_split_distances_on_equal_parts(d.traj_gt, num_parts=5)
    split_distances = np.hstack((split_distances, get_split_distances_equispaced(d.traj_gt, step_size=10)))

    for s in split_distances:
        try:
            m_t = RPE(PoseRelation.translation_part, s, metrics.Unit.meters, all_pairs=True)
            m_r = RPE(PoseRelation.rotation_angle_deg, s, metrics.Unit.meters, all_pairs=True)
            m_t.process_data((d.traj_gt_synced, d.traj_est_aligned))
            m_r.process_data((d.traj_gt_synced, d.traj_est_aligned))

            d.rpe_error_t[s] = m_t.get_result().np_arrays['error_array']
            d.rpe_error_r[s] = m_r.get_result().np_arrays['error_array']

        except FilterException:
            d.rpe_error_t[s] = np.ndarray([])
            d.rpe_error_r[s] = np.ndarray([])

    for m in METRICS:
        m.process_data((d.traj_gt_synced, d.traj_est_aligned))
        result = m.get_result()
        d.ate_errors[str(m)] = result.np_arrays['error_array']

    return d


def create_trajectory_result_table_wrt_traveled_dist(s: EvaluationDataSummary) -> pd.DataFrame:
    columns = ["Dataset", F"Mean Position Error ({s.name}) [%]", F"Mean Rotation error ({s.name}) [deg/m]"]
    result_table = pd.DataFrame(columns=columns)

    for d in s.data.values():
        if d.trajectory_data is None:
            continue
        position_metric = metrics.APE(metrics.PoseRelation.translation_part)
        orientation_metric = metrics.APE(metrics.PoseRelation.rotation_angle_deg)

        mask = d.trajectory_data.traj_gt_synced.distances > 0

        pos_error = d.trajectory_data.ate_errors[str(position_metric)][mask] / d.trajectory_data.traj_gt_synced.distances[mask]
        deg_error = d.trajectory_data.ate_errors[str(orientation_metric)][mask] / d.trajectory_data.traj_gt_synced.distances[mask]

        pos_error = np.round(np.mean(pos_error*100), 2)  # in percent
        deg_error = np.round(np.mean(deg_error), 2)

        result_table.loc[len(result_table)] = [d.name, pos_error, deg_error]

    return result_table


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
    return np.hstack(tuple(arrays))


def plot_rpg_error_arrays(pc: PlotContext, trajectories: Collection[EvaluationData], labels=None, use_log=False):
    auto_labels = []
    errors_rot_deg = []
    errors_trans_m = []
    gt_trajectory = None
    for t in trajectories:
        if t.trajectory_data is not None:
            errors_rot_deg.append(t.trajectory_data.rpe_error_r)
            errors_trans_m.append(t.trajectory_data.rpe_error_t)
            auto_labels.append(t.name)
            if gt_trajectory is None:
                gt_trajectory = t.trajectory_data.traj_gt

    if len(errors_rot_deg) <= 0:
        return

    if labels is None:
        labels = auto_labels

    distances = get_split_distances_on_equal_parts(gt_trajectory, 5)

    if use_log:
        data_trans_m = [[np.log1p(e[k]) / np.log(10) for k in distances] for e in errors_trans_m]
    else:
        data_trans_m = [[e[k] for k in distances] for e in errors_trans_m]

    ax = pc.get_axis()
    ax.set_xlabel("Distance traveled [m]")
    if use_log:
        ax.set_ylabel(F"Translation error [log m]")
    else:
        ax.set_ylabel(F"Translation error [m]")
    boxplot_compare(ax, distances, data_trans_m, labels)

    ax = pc.get_axis()
    ax.set_xlabel("Distance traveled [m]")
    ax.set_ylabel(F"Rotation error [deg]")
    data_rot_deg = [[e[k] for k in distances] for e in errors_rot_deg]
    boxplot_compare(ax, distances, data_rot_deg, labels)


def plot_trajectory_plots(eval_data: EvaluationData, output_folder):
    with PlotContext(os.path.join(output_folder, "xy_plot.svg")) as pc:
        plot_trajectory(pc, [eval_data])

    # with PlotContext(None) as pc:
    with PlotContext(os.path.join(output_folder, "rpg_subtrajectory_errors.svg"), subplot_cols=2) as pc:
        plot_rpg_error_arrays(pc, [eval_data])


def create_summary_info(summary: EvaluationDataSummary):
    summary.trajectory_summary_table = create_absolute_trajectory_result_table(summary)


def plot_trajectory(pc: PlotContext, trajectories: Collection[EvaluationData]):
    traj_by_label = dict()

    if len(trajectories) <= 0:
        return

    first = True

    for t in trajectories:
        if t.trajectory_data is not None:
            if first:
                first = False
                traj_by_label[F"{t.name} reference"] = t.trajectory_data.traj_gt_synced
            traj_by_label[F"{t.name} estimate"] = t.trajectory_data.traj_est_synced

    plot.trajectories(pc.figure, traj_by_label, plot.PlotMode.xy)


def plot_summary_plots(summary: EvaluationDataSummary, output_folder):
    for m in METRICS:
        with PlotContext(os.path.join(output_folder, name_to_identifier(str(m)) + "_boxplot.svg")) as pc:
            plot_error_comparison(pc, summary.data.values(), str(m), PlotType.BOXPLOT)
        with PlotContext(os.path.join(output_folder, name_to_identifier(str(m)) + "_in_time.svg")) as pc:
            plot_error_comparison(pc, summary.data.values(), str(m), PlotType.TIME_SERIES)


def plot_error_comparison(pc: PlotContext, evaluations: Collection[EvaluationData], error_key: str,
                          plot_type: PlotType = PlotType.BOXPLOT, labels=None):
    auto_labels = []
    data = []
    time_arrays = []
    for e in evaluations:
        if e.trajectory_data is not None:
            t = e.trajectory_data.traj_est_synced.timestamps.flatten()

            # data.append(np.log1p(e.trajectory_data.errors[error_key]))
            data.append(e.trajectory_data.ate_errors[error_key])

            if len(t)-1 == len(e.trajectory_data.ate_errors[error_key]):
                # this can happen for RPE errors, when they are calculated between two poses (aka Zaunproblem: |-|-|-|)
                t = t[:-1]

            time_arrays.append(t - t[0])
            auto_labels.append(e.name)

    if len(auto_labels) <= 0:
        return

    if labels is None:
        labels = auto_labels

    if plot_type == PlotType.BOXPLOT:
        boxplot(pc, data, labels, error_key)
    elif plot_type == PlotType.TIME_SERIES:
        # time_series_plot(filename, time_arrays, data, labels, F"APE w.r.t. {kind.value}", ylabel="log error")
        time_series_plot(pc, time_arrays, data, labels, error_key)
    else:
        raise ValueError(F"Invalid plot type '{plot_type}'")
