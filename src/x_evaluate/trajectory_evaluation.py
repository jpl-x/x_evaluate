import os

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


class TrajectoryEvaluator:

    # POSE_RELATIONS = [metrics.PoseRelation.full_transformation, metrics.PoseRelation.rotation_angle_deg,
    #                   metrics.PoseRelation.translation_part]
    POSE_RELATIONS = [metrics.PoseRelation.full_transformation]


    def __init__(self):

        columns = ["Name"] + ["RPE " + r.value + " [RMS]" for r in self.POSE_RELATIONS] + \
                  ["APE " + r.value + " [RMS]" for r in self.POSE_RELATIONS]

        self._result_table = pd.DataFrame(columns=columns)
        self._ape_error_arrays = {
            '': {k: np.array([]) for k in self.POSE_RELATIONS}
        }
        self._rpe_error_arrays = copy.deepcopy(self._ape_error_arrays)
        self._ref_trajectories = dict()
        self._est_trajectories = dict()

    def evaluate(self, name, df_poses: pd.DataFrame, df_groundtruth: pd.DataFrame, output_folder):

        traj_est = convert_to_evo_trajectory(df_poses, prefix="estimated_")
        traj_ref = convert_to_evo_trajectory(df_groundtruth)

        self._ref_trajectories[name] = traj_ref
        self._est_trajectories[name] = traj_est

        self.plot_trajectory(os.path.join(output_folder, "xy_plot.svg"), name, name)

        max_diff = 0.01
        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff)
        # traj_est.align(traj_ref, correct_scale=False, correct_only_scale=False)

        ape_results = []
        rpe_results = []

        for r in self.POSE_RELATIONS:
            ape_metric = metrics.APE(r)
            ape_metric.process_data((traj_ref, traj_est))
            ape_metric.pose_relation = metrics.PoseRelation.rotation_angle_deg
            ape_results += [ape_metric.get_statistic(metrics.StatisticsType.rmse)]

            rpe_metric = metrics.RPE(r)
            rpe_metric.process_data((traj_ref, traj_est))
            rpe_results += [rpe_metric.get_statistic(metrics.StatisticsType.rmse)]

            rpe_error_array = rpe_metric.get_result().np_arrays['error_array']
            ape_error_array = ape_metric.get_result().np_arrays['error_array']

            self._ape_error_arrays[''][r] = np.hstack((self._ape_error_arrays[''][r], ape_error_array))
            self._rpe_error_arrays[''][r] = np.hstack((self._rpe_error_arrays[''][r], rpe_error_array))

        result_row = [name] + ape_results + rpe_results
        self._result_table.loc[len(self._result_table)] = result_row

    def print_summary(self):
        rpe_array = self._rpe_error_arrays[''][metrics.PoseRelation.full_transformation]
        ape_array = self._ape_error_arrays[''][metrics.PoseRelation.full_transformation]
        rpe_rms = np.linalg.norm(rpe_array) / np.sqrt(len(rpe_array))
        ape_rms = np.linalg.norm(ape_array) / np.sqrt(len(rpe_array))

        print(F"Overall [RPE] [APE]: {rpe_rms:>15.2f} {ape_rms:>15.2f}")

    def plot_trajectory(self, filename, reference_name, *estimate_names):

        fig = plt.figure()
        traj_by_label = {F"{x} estimate": self._est_trajectories[x] for x in estimate_names}
        traj_by_label[F"{reference_name} reference"] = self._ref_trajectories[reference_name]

        plot.trajectories(fig, traj_by_label, plot.PlotMode.xy)

        if filename is None or len(filename) == 0:
            plt.show()
        else:
            plt.savefig(filename)

