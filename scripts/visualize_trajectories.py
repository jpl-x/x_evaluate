import argparse
import copy
from datetime import datetime
import glob
import os
from typing import List, Callable, Optional, Dict

import numpy as np
import time

import pandas as pd
import x_evaluate.plots
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import plot
from klampt.math import se3
from klampt.model.trajectory import SE3Trajectory, SE3HermiteTrajectory
from scipy.spatial.transform import Rotation as R

from x_evaluate.plots import PlotContext, plot_evo_trajectory_with_euler_angles, DEFAULT_COLORS
import matplotlib.pyplot as plt

from x_evaluate.utils import read_neurobem_trajectory, read_esim_trajectory_csv, read_x_evaluate_gt_csv


class EvoTrajectoryVisualizer:
    def __init__(self, pc: PlotContext, files: List[str], filename_to_evo_trajectory: Callable[[str], PoseTrajectory3D],
                 special_key_actions=Optional[Dict[str, Callable[[PoseTrajectory3D], PoseTrajectory3D]]]):
        assert len(files) > 0, "provide at least one trajectory"

        self._pc = pc
        self._files = files
        self._special_key_actions: Optional[Dict[str, Callable[[PoseTrajectory3D], None]]] = special_key_actions
        self._filename_to_evo_trajectory = filename_to_evo_trajectory
        self._digits = None
        self._digits_last_ts = None

        self._current_idx = 0
        self._current_trajectory = self._filename_to_evo_trajectory(self._files[self._current_idx])

        pc.figure.canvas.mpl_connect('key_press_event', self.press)

        self._blocked = False

    def plot_current_trajectory(self):
        plt.clf()
        self._plot_evo_trajectory()

        self._pc.figure.canvas.draw()

        if not self._blocked:
            self._blocked = True
            self.block()

    def _plot_evo_trajectory(self):
        label = self._files[self._current_idx]
        trajectory = self._current_trajectory
        plot_context = self._pc
        # plot_evo_trajectory_with_euler_angles(plot_context, trajectory, os.path.basename(label))
        plot_evo_trajectory_with_euler_angles(plot_context, trajectory, label)

        stats = trajectory.get_statistics()

        v_avg = np.round(stats['v_avg (m/s)'], 1)
        v_max = np.round(stats['v_max (m/s)'], 1)
        t = np.round(trajectory.timestamps[-1] - trajectory.timestamps[0], 1)
        d = np.round(trajectory.distances[-1], 1)
        hz = np.round(1/np.mean(trajectory.timestamps[1:]-trajectory.timestamps[:-1]) , 1)
        ts = os.path.getmtime(self._files[self._current_idx])
        plot_context.figure.suptitle(F"[{self._current_idx}/{len(self._files)}] n = {len(trajectory.timestamps)} "
                                 F"poses, {hz}Hz, "
                                 F" {d}m,"
                                 F" {t}s, v = {v_avg} m/s (max: {v_max} m/s) [{datetime.fromtimestamp(ts).ctime()}]")

    def block(self):
        plt.show()

    def press(self, event):
        if event.key in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            if self._digits and self._digits_last_ts and (time.time() - self._digits_last_ts < 1):
                self._digits = F"{self._digits}{event.key}"
            else:
                self._digits = event.key
            self._digits_last_ts = time.time()
        elif event.key == 'escape':
            plt.close(self._pc.figure)
        elif event.key == 'enter' and self._digits and self._digits_last_ts and (time.time() - self._digits_last_ts <1):
            new_idx = int(self._digits)
            if 0 <= new_idx < len(self._files)-1:
                self._current_idx = new_idx
                self.load_trajectory_from_current_index()
            self._digits = None
        elif event.key == 'left':
            if self._current_idx > 0:
                self._current_idx -= 1
                self.load_trajectory_from_current_index()
        elif event.key == 'right':
            if self._current_idx < len(self._files) - 1:
                self._current_idx += 1
                self.load_trajectory_from_current_index()
        elif self._special_key_actions and event.key in self._special_key_actions.keys():
            self._current_trajectory = self._special_key_actions[event.key](self._current_trajectory)
            self.plot_current_trajectory()

    def load_trajectory_from_current_index(self):
        self._current_trajectory = self._filename_to_evo_trajectory(self._files[self._current_idx])
        self.plot_current_trajectory()


def convert_to_esim_trajectory(output_filename, input_trajectory: PoseTrajectory3D):
    output_trajectory = copy.deepcopy(input_trajectory)
    print(F"Converting NeuroBEM to ESIM trajectory, and saving to '{output_filename}'")
    headers = ["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]

    # # # let the camera look downwards
    tf = np.eye(4)
    tf[1, 1] = -1
    tf[2, 2] = -1
    output_trajectory.transform(tf, True)

    # normalize quaternions: ESIM checks for this, and seems NeuroBEM trajectories are not perfect in this regard
    rotations = R.from_quat(output_trajectory.orientations_quat_wxyz[:, [1, 2, 3, 0]])

    data = np.hstack(((output_trajectory.timestamps[:, np.newaxis] - output_trajectory.timestamps[0]) * 1e9,
                      output_trajectory.positions_xyz, rotations.as_quat())).T
    # output_traj = pd.concat(data, axis=1, keys=headers)
    output_traj = pd.DataFrame.from_dict(dict(zip(headers, data)))

    t = output_traj['timestamp'].to_numpy()

    print(output_traj)
    with open(output_filename, 'w') as f:
        # ESIM checks for this header comment
        f.write("# timestamp, x, y, z, qx, qy, qz, qw\n")
        output_traj.to_csv(f, header=False, index=False)
    return input_trajectory  # return unchanged trajectory


def downsample(trajectory: PoseTrajectory3D) -> PoseTrajectory3D:
    t = trajectory.timestamps[::2]
    xyz = trajectory.positions_xyz[::2, :]
    wxyz = trajectory.orientations_quat_wxyz[::2, :]
    return PoseTrajectory3D(xyz, wxyz, t)

# output_traj = output_traj.iloc[::8, :]
# t = output_traj['timestamp'].to_numpy()
# print(F"Average pose freq after down-sampling: {1/np.round(np.mean(t[1:]-t[:-1]) / 1e9, 6)}Hz")


def add_bootstrapping_sequence(trajectory: PoseTrajectory3D) -> PoseTrajectory3D:
    t = trajectory.timestamps

    t_zero = trajectory.timestamps[0]

    first_n = 50
    target_poses = [(rot.as_matrix().flatten().tolist(), pos.flatten().tolist()) for rot, pos in \
            zip(R.from_quat(trajectory.orientations_quat_wxyz[:first_n, [1, 2, 3, 0]]), trajectory.positions_xyz[:first_n])]

    target_chunk = SE3Trajectory(t[:first_n], target_poses)

    target_chunk_spline = SE3HermiteTrajectory()
    target_chunk_spline.makeSpline(target_chunk)
    v0_spline = np.array(target_chunk_spline.deriv(0.0)[1])

    initial_rot = R.from_quat(trajectory.orientations_quat_wxyz[0, [1, 2, 3, 0]])
    rot_z, rot_y, rot_x = initial_rot.as_euler("ZYX")

    if np.max(np.abs(np.rad2deg([rot_y, rot_x]))) > 15:
        print("WARNING: initial orientation is tilted more than 15° in x or y, interpolation to bootstrapping "
              "sequence might be non-smooth")

    initial_pos = np.array(trajectory.positions_xyz[0, :])
    bootstrap_pos = initial_pos - 0.5*v0_spline
    bootstrap_rot = R.from_euler("ZYX", [rot_z, 0, 0])
    bootstrap_pose = (bootstrap_rot.as_matrix().flatten(), bootstrap_pos.flatten())

    origin = se3.from_translation([0, 0, 0])
    left = se3.from_translation([0, 2, 0])
    front = se3.from_translation([2, 0, 0])

    sequence = [origin, origin, left, origin, front, origin, origin]
    sequence_t = [ -10,   -8.5,   -7,   -5.5,    -4,   -2.5,     -1]

    bootstrap_poses = [se3.mul(bootstrap_pose, s) for s in sequence]
    bootstrap_times = [i + t_zero for i in sequence_t]
    bootstrapping_chunk = SE3Trajectory(bootstrap_times, bootstrap_poses)

    bootstrapping = SE3HermiteTrajectory()
    concatenated = bootstrapping_chunk.concat(target_chunk)
    bootstrapping.makeSpline(concatenated)

    dt = np.mean(trajectory.timestamps[1:] - trajectory.timestamps[:-1])
    prefix_trajectory = bootstrapping.discretize(dt)
    prefix_trajectory = prefix_trajectory.before(-dt + t_zero)  # before(..) logic means '<=' ---> do not include zero

    xyz = np.array(prefix_trajectory.getPositionTrajectory().milestones)
    xyzw = R.from_matrix(np.array(prefix_trajectory.getRotationTrajectory().milestones).reshape((-1, 3, 3))).as_quat()
    # xyzw[:, :] = initial_rot.as_quat()
    wxyz = xyzw[:, [3, 0, 1, 2]]
    t = np.array(prefix_trajectory.times)
    xyz = np.concatenate((xyz, trajectory.positions_xyz))
    wxyz = np.concatenate((wxyz, trajectory.orientations_quat_wxyz))
    t = np.concatenate((t, trajectory.timestamps))
    new_trajectory = PoseTrajectory3D(xyz, wxyz, t)
    return new_trajectory


def main():
    parser = argparse.ArgumentParser(description='Converting NeuroBEM drone trajectories to ESIM compatible csv')

    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    matcher = os.path.join(os.path.dirname(args.input), "*.csv")
    matches = glob.glob(matcher)
    matches.sort()

    # selected_indices = [1]
    # matches = [matches[i] for i in selected_indices]

    # esim_csvs = ["/storage/data/projects/nasa-eve/eklt/src/rpg_esim/event_camera_simulator/esim_ros/cfg/traj"
    #              "/neuro_bem_esim.csv",  # "/storage/data/projects/nasa-eve/eklt/src/rpg_esim/event_camera_simulator"
    #                                      # "/esim_ros/cfg/traj/neuro_bem_esim_eight.slow",
    #              "/tmp/esim_spline_dump.csv",
    #              "/storage/data/projects/nasa-eve/friedrich_python_helpers/trajectory_gen/test.csv"]

    # esim_matcher = os.path.join("/storage/data/projects/nasa-eve/eklt/src/rpg_esim/event_camera_simulator/esim_ros"
    #                             "/cfg/traj/vmax/", "*.csv")
    # esim_matches = glob.glob(esim_matcher)
    # esim_matches.sort()
    gt_matcher = os.path.join("/storage/data/projects/nasa-eve/experiments/sim_uslam/gt/*/gt.csv")
    gt_matches = glob.glob(gt_matcher)
    gt_matches.sort()

    # with PlotContext(subplot_rows=2, subplot_cols=2) as pc:
    #     v = EvoTrajectoryVisualizer(pc, gt_matches, read_x_evaluate_gt_csv,
    #     # v = EvoTrajectoryVisualizer(pc, matches, read_neurobem_trajectory,
    #                                 special_key_actions={
    #                                     'enter': lambda t: convert_to_esim_trajectory(args.output, t),
    #                                     'b': add_bootstrapping_sequence,
    #                                     'd': downsample
    #                                 })
    #     v.plot_current_trajectory()

    x_evaluate.plots.use_paper_style_plots = True

    with PlotContext("/home/flo/cloud/school/2020-2021/nasa/figures/for_paper/mellon") as pc:
        traj = read_x_evaluate_gt_csv(gt_matches[7])
        # traj_by_label = {
        #     "Mars Mellon": traj
        # }

        ax = pc.get_axis(projection="3d")
        x = traj.positions_xyz[:, 0]
        y = traj.positions_xyz[:, 1]
        z = traj.positions_xyz[:, 2]
        ax.plot(x, y, z, color=DEFAULT_COLORS[
            0], label="Mars Mellon")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")

        ax.w_xaxis.pane.fill = False
        ax.w_yaxis.pane.fill = False
        ax.w_zaxis.pane.fill = False
        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))

        # plot.trajectories(pc.figure, traj_by_label, plot.PlotMode.xyz, subplot_arg=211)

    # plt.show()


if __name__ == '__main__':
    main()
