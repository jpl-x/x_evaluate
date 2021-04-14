import os
from typing import Collection

from evo.core.metrics import PoseRelation
from natsort import os_sorted

from x_evaluate.conversions import convert_to_evo_trajectory
from evo.core import sync
from evo.core import metrics
from evo.tools import plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

from x_evaluate.evaluation_data import TrajectoryData, EvaluationDataSummary, EvaluationData

# POSE_RELATIONS = [metrics.PoseRelation.full_transformation, metrics.PoseRelation.rotation_angle_deg,
#                   metrics.PoseRelation.translation_part]

POSE_RELATIONS = [metrics.PoseRelation.full_transformation]


def evaluate_trajectory(df_poses: pd.DataFrame, df_groundtruth: pd.DataFrame) -> TrajectoryData:
    d = TrajectoryData()
    d.traj_est = convert_to_evo_trajectory(df_poses, prefix="estimated_")
    d.traj_ref = convert_to_evo_trajectory(df_groundtruth)

    max_diff = 0.01
    d.traj_ref, d.traj_est = sync.associate_trajectories(d.traj_ref, d.traj_est, max_diff)

    for r in POSE_RELATIONS:
        ape_metric = metrics.APE(r)
        ape_metric.process_data((d.traj_ref, d.traj_est))
        ape_metric.pose_relation = metrics.PoseRelation.rotation_angle_deg

        rpe_metric = metrics.RPE(r)
        rpe_metric.process_data((d.traj_ref, d.traj_est))

        rpe_error_array = rpe_metric.get_result().np_arrays['error_array']
        ape_error_array = ape_metric.get_result().np_arrays['error_array']

        d.ape_error_arrays[r] = ape_error_array
        d.rpe_error_arrays[r] = rpe_error_array
    return d


def create_trajectory_result_table(s: EvaluationDataSummary) -> pd.DataFrame:
    columns = ["Name"] + ["RPE " + r.value + " [RMS]" for r in POSE_RELATIONS] + \
              ["APE " + r.value + " [RMS]" for r in POSE_RELATIONS]

    result_table = pd.DataFrame(columns=columns)

    for d in s.data.values():
        ape_results = []
        rpe_results = []

        for r in POSE_RELATIONS:
            ape_array = d.trajectory_data.ape_error_arrays[r]
            ape_rmse = np.linalg.norm(ape_array) / np.sqrt(len(ape_array))
            rpe_array = d.trajectory_data.rpe_error_arrays[r]
            rpe_rmse = np.linalg.norm(rpe_array) / np.sqrt(len(rpe_array))
            ape_rmse += [ape_rmse]
            rpe_results += [rpe_rmse]

        result_row = [d.name] + ape_results + rpe_results
        result_table.loc[len(result_table)] = result_row
    return result_table


def combine_rpe_error(evaluations: Collection[EvaluationData], pose_relation: metrics.PoseRelation) -> np.ndarray:
    arrays = []
    for d in evaluations:
        if d.trajectory_data is not None:
            arrays.append(d.trajectory_data.rpe_error_arrays[pose_relation])
    return np.hstack(tuple(arrays))


def combine_ape_error(evaluations: Collection[EvaluationData], pose_relation: metrics.PoseRelation) -> np.ndarray:
    arrays = []
    for d in evaluations:
        if d.trajectory_data is not None:
            arrays.append(d.trajectory_data.ape_error_arrays[pose_relation])
    return np.hstack(tuple(arrays))


def plot_trajectory_plots(eval_data: EvaluationData, output_folder):
    plot_trajectory(os.path.join(output_folder, "xy_plot.svg"), [eval_data])


def print_trajectory_summary(summary: EvaluationDataSummary):
    ape_array = combine_ape_error(summary.data.values(), metrics.PoseRelation.full_transformation)
    if len(ape_array) <= 0:
        return
    rpe_array = combine_rpe_error(summary.data.values(), metrics.PoseRelation.full_transformation)

    rpe_rms = np.linalg.norm(rpe_array) / np.sqrt(len(rpe_array))
    ape_rms = np.linalg.norm(ape_array) / np.sqrt(len(rpe_array))

    print(F"Overall [RPE] [APE]: {rpe_rms:>15.2f} {ape_rms:>15.2f}")


def plot_trajectory(filename, trajectories: Collection[EvaluationData]):
    traj_by_label = dict()

    if len(trajectories):
        return

    first = True

    for t in trajectories:
        if t.trajectory_data is not None:
            if first:
                first = False
                traj_by_label[F"{t.name} reference"] = t.trajectory_data.traj_ref
            traj_by_label[F"{t.name} estimate"] = t.trajectory_data.traj_est

    fig = plt.figure()
    plot.trajectories(fig, traj_by_label, plot.PlotMode.xy)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.clf()


def plot_summary_boxplot(summary: EvaluationDataSummary, output_folder):
    for r in POSE_RELATIONS:
        filename = os.path.join(output_folder, "ape_boxplot_" + r.name + ".svg")
        plot_ape_comparison(summary.data.values(), filename, r)
        filename = os.path.join(output_folder, "rpe_boxplot_" + r.name + ".svg")
        plot_rpe_comparison(summary.data.values(), filename, r)


def plot_ape_comparison(evaluations: Collection[EvaluationData], filename, kind: metrics.PoseRelation):
    labels = []
    data = []
    for e in evaluations:
        if e.trajectory_data is not None:
            data.append(e.trajectory_data.ape_error_arrays[kind])
            labels.append(e.name)

    if len(labels) <= 0:
        return

    boxplot(filename, data, labels, F"APE w.r.t. {kind.value} comparison")


def plot_rpe_comparison(evaluations: Collection[EvaluationData], filename, kind: metrics.PoseRelation):
    labels = []
    data = []
    for e in evaluations:
        if e.trajectory_data is not None:
            data.append(e.trajectory_data.rpe_error_arrays[kind])
            labels.append(e.name)

    if len(labels) <= 0:
        return

    boxplot(filename, data, labels, F"RPE w.r.t. {kind.value} comparison")


def boxplot(filename, data, labels, title=""):
    plt.figure()
    plt.boxplot(data, vert=True, labels=labels)
    plt.title(title)

    if filename is None or len(filename) == 0:
        plt.show()
    else:
        plt.savefig(filename)
    plt.clf()
