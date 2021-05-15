import os
from typing import Collection

from evo.core.metrics import APE, RPE, PoseRelation, PE

from x_evaluate.rpg_trajectory_evaluation import split_trajectory_into_equal_parts, \
    split_trajectory_on_traveled_distance_grid
from x_evaluate.utils import convert_to_evo_trajectory, rms, name_to_identifier
from x_evaluate.plots import boxplot, time_series_plot, PlotType, PlotContext, boxplot_compare
from evo.core import sync
from evo.core import metrics
from evo.tools import plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import x_evaluate.rpg_trajectory_evaluation as rpg

from x_evaluate.evaluation_data import TrajectoryData, EvaluationDataSummary, EvaluationData, AlignmentType

POSE_RELATIONS = [metrics.PoseRelation.translation_part, metrics.PoseRelation.rotation_angle_deg]

# POSE_RELATIONS = [metrics.PoseRelation.full_transformation]

APE_METRICS = [APE(p) for p in POSE_RELATIONS]
RPE_METRICS = [RPE(p) for p in POSE_RELATIONS]

METRICS = APE_METRICS + RPE_METRICS


def evaluate_trajectory(df_poses: pd.DataFrame, df_groundtruth: pd.DataFrame) -> TrajectoryData:
    d = TrajectoryData()
    d.traj_est, d.raw_estimate_t_xyz_wxyz = convert_to_evo_trajectory(df_poses, prefix="estimated_")
    d.traj_ref, _ = convert_to_evo_trajectory(df_groundtruth)

    max_diff = 0.01
    d.traj_ref, d.traj_est = sync.associate_trajectories(d.traj_ref, d.traj_est, max_diff)

    split_distances = split_trajectory_into_equal_parts(d.traj_ref, num_parts=5)
    # fancy_split_distances = split_trajectory_on_traveled_distance_grid(d.traj_ref, step_size=5)

    sub_trajectories = rpg.rpg_sub_trajectories(d.traj_ref, d.traj_est, split_distances)

    for (gt, est, split_distance) in sub_trajectories:

        # aligns est to gt
        rpg.rpg_align(gt, est, AlignmentType.PosYaw)

        d.sub_traj_errors[split_distance] = dict()

        for m in APE_METRICS:
            m.process_data((gt, est))
            d.sub_traj_errors[split_distance][str(m)] = m.get_result().np_arrays['error_array']

    for m in METRICS:
        m.process_data((d.traj_ref, d.traj_est))
        result = m.get_result()
        d.errors[str(m)] = result.np_arrays['error_array']
    return d


def create_trajectory_result_table(s: EvaluationDataSummary) -> pd.DataFrame:
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
        add_result_row(d.name, d.trajectory_data.errors)

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
            arrays.append(d.trajectory_data.errors[error_key])
    return np.hstack(tuple(arrays))


def plot_rpg_error_arrays(pc: PlotContext, trajectories: Collection[EvaluationData], labels=None, use_log=False):
    auto_labels = []
    errors = []
    for t in trajectories:
        if t.trajectory_data is not None:
            errors.append(t.trajectory_data.sub_traj_errors)
            auto_labels.append(t.name)

    if len(errors) <= 0:
        return

    if labels is None:
        labels = auto_labels

    labels = [l[1:] if l.startswith('_') else l for l in labels]

    distances = list(errors[0].keys())

    for m in APE_METRICS:
        if use_log:
            data = [[np.log1p(e[k][str(m)]) / np.log(10) for k in distances] for e in errors]
        else:
            data = [[e[k][str(m)] for k in distances] for e in errors]

        ax = pc.get_axis()
        ax.set_xlabel("Distance traveled [m]")
        if use_log:
            ax.set_ylabel(F"APE (log {m.unit.name})")
        else:
            ax.set_ylabel(F"APE ({m.unit.name})")
        boxplot_compare(ax, distances, data, labels)


def plot_trajectory_plots(eval_data: EvaluationData, output_folder):
    with PlotContext(os.path.join(output_folder, "xy_plot.svg")) as pc:
        plot_trajectory(pc, [eval_data])

    # with PlotContext(None) as pc:
    with PlotContext(os.path.join(output_folder, "rpg_subtrajectory_errors.svg")) as pc:
        plot_rpg_error_arrays(pc, [eval_data])


def create_summary_info(summary: EvaluationDataSummary):
    summary.trajectory_summary_table = create_trajectory_result_table(summary)


def plot_trajectory(pc: PlotContext, trajectories: Collection[EvaluationData]):
    traj_by_label = dict()

    if len(trajectories) <= 0:
        return

    first = True

    for t in trajectories:
        if t.trajectory_data is not None:
            if first:
                first = False
                traj_by_label[F"{t.name} reference"] = t.trajectory_data.traj_ref
            traj_by_label[F"{t.name} estimate"] = t.trajectory_data.traj_est

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
            t = e.trajectory_data.traj_est.timestamps.flatten()

            # data.append(np.log1p(e.trajectory_data.errors[error_key]))
            data.append(e.trajectory_data.errors[error_key])

            if len(t)-1 == len(e.trajectory_data.errors[error_key]):
                # this happens for RPE errors, since they are calculated between two poses (aka Zaunproblem |-|-|-|-|)
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
