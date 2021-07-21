import argparse
import glob
import os
from typing import List, Callable

import numpy as np

import pandas as pd
from evo.core.trajectory import PoseTrajectory3D
from klampt.math import se3
from klampt.model.trajectory import SE3Trajectory, SE3HermiteTrajectory
from scipy.spatial.transform import Rotation as R
from evo.tools import plot

from x_evaluate.plots import PlotContext, time_series_plot
import matplotlib.pyplot as plt


class EvoTrajectoryVisualizer:
    def __init__(self, pc: PlotContext, files: List[str], filename_to_evo_trajectory: Callable[[str], PoseTrajectory3D],
                 enter_action=None):
        assert len(files) > 0, "provide at least one trajectory"

        self._pc = pc
        self._files = files
        self._enter_action = enter_action
        self._filename_to_evo_trajectory = filename_to_evo_trajectory

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
        traj_by_label = {
            str(os.path.basename(self._files[self._current_idx])): self._current_trajectory
        }
        plot.trajectories(self._pc.figure, traj_by_label, plot.PlotMode.xyz, subplot_arg=211)
        time_series_plot(self._pc, self._current_trajectory.timestamps, self._current_trajectory.positions_xyz.T,
                         ["x", "y", "z"], subplot_arg=223)
        rotations = R.from_quat(self._current_trajectory.orientations_quat_wxyz[:, [1, 2, 3, 0]])

        time_series_plot(self._pc, self._current_trajectory.timestamps, np.rad2deg(rotations.as_euler("ZYX")).T,
                         ["euler_z", "euler_y", "euler_x"], subplot_arg=224)
        # plot.trajectories(self._pc.figure, traj_by_label, plot.PlotMode.xy, subplot_arg=121)
        # plot.trajectories(self._pc.figure, traj_by_label, plot.PlotMode.xz, subplot_arg=122)

        stats = self._current_trajectory.get_statistics()

        v_avg = np.round(stats['v_avg (m/s)'], 1)
        v_max = np.round(stats['v_max (m/s)'], 1)
        t = np.round(self._current_trajectory.timestamps[-1] - self._current_trajectory.timestamps[0], 1)
        d = np.round(self._current_trajectory.distances[-1], 1)
        self._pc.figure.suptitle(F"n = {len(self._current_trajectory.timestamps)} poses, {d}m, {t}s, v = {v_avg} m/s "
                                 F"(max: {v_max} m/s)")

    def block(self):
        plt.show()

    def press(self, event):
        if event.key == 'escape':
            plt.close(self._pc.figure)
        elif event.key == 'enter':
            if self._enter_action:
                self._enter_action(self._current_trajectory)
        elif event.key == 'left':
            if self._current_idx > 0:
                self._current_idx -= 1
                self.update_current_trajectory()
        elif event.key == 'right':
            if self._current_idx < len(self._files) - 1:
                self._current_idx += 1
                self.update_current_trajectory()
        elif event.key == 'b':
            self._current_trajectory = add_bootstrapping(self._current_trajectory)
            self.plot_current_trajectory()

    def update_current_trajectory(self):
        self._current_trajectory = self._filename_to_evo_trajectory(self._files[self._current_idx])
        self.plot_current_trajectory()


def main():
    parser = argparse.ArgumentParser(description='Converting NeuroBEM drone trajectories to ESIM compatible csv')

    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    # matches = glob.glob(os.path.join(args.input_folder, "*"))

    matcher = os.path.join(os.path.dirname(args.input), "*.csv")
    matches = glob.glob(matcher)
    matches.sort()

    with PlotContext(subplot_rows=2, subplot_cols=2) as pc:
        v = EvoTrajectoryVisualizer(pc, matches, read_neurobem_trajectory,
                                    enter_action=lambda t: convert_to_esim_trajectory(args.output, t))
        v.plot_current_trajectory()

        # input("Let's try this")


def convert_to_esim_trajectory(output_filename, trajectory: PoseTrajectory3D):
    print(F"Converting NeuroBEM to ESIM trajectory, and saving to '{output_filename}'")
    headers = ["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]

    # normalize quaternions: ESIM checks for this, and seems NeuroBEM trajectories are not perfect in this regard
    rotations = R.from_quat(trajectory.orientations_quat_wxyz[:, [1, 2, 3, 0]])

    data = np.hstack(((trajectory.timestamps[:, np.newaxis] - trajectory.timestamps[0]) * 1e9,
                      trajectory.positions_xyz, rotations.as_quat())).T
    # output_traj = pd.concat(data, axis=1, keys=headers)
    output_traj = pd.DataFrame.from_dict(dict(zip(headers, data)))
    print(output_traj)
    with open(output_filename, 'w') as f:
        # ESIM checks for this header comment
        f.write("# timestamp, x, y, z, qx, qy, qz, qw\n")
        output_traj.to_csv(f, header=False, index=False)


def read_neurobem_trajectory(filename):
    input_traj = pd.read_csv(filename)
    # zero align
    input_traj['t'] -= input_traj['t'][0]
    t_xyz_wxyz = input_traj[["t", "pos x", "pos y", "pos z", "quat w", "quat x", "quat y", "quat z"]].to_numpy()
    trajectory = PoseTrajectory3D(t_xyz_wxyz[:, 1:4], t_xyz_wxyz[:, 4:8], t_xyz_wxyz[:, 0])
    return trajectory


def add_bootstrapping(trajectory: PoseTrajectory3D):
    xyz = trajectory.positions_xyz
    t = trajectory.timestamps

    t_zero = trajectory.timestamps[0]

    v0 = (xyz[1, :] - xyz[0, :])/(t[1] - t[0])

    print(F"v_0 = {np.round(v0, 2)}")
    print(F"||v_0|| = {np.round(np.linalg.norm(v0), 2)} m/s")

    first_n = 50

    target_poses = [se3.from_homogeneous(t) for t in trajectory.poses_se3[:first_n]]

    target_chunk = SE3Trajectory(t[:first_n], target_poses)

    target_chunk_spline = SE3HermiteTrajectory()
    target_chunk_spline.makeSpline(target_chunk)
    v0_spline = np.array(target_chunk_spline.deriv(0.0)[1])

    initial_rot = R.from_quat(trajectory.orientations_quat_wxyz[0, [1, 2, 3, 0]])
    # rot_z, rot_y, rot_x = initial_rot.as_euler("ZYX")
    # print(F'Rotation around z: {np.rad2deg(rot_z)}')
    #
    # if np.max(np.abs(np.rad2deg([rot_y, rot_x]))) > 15:
    #     print("WARNING: initial orientation is tilted more than 15° in x or y")
    #
    initial_pos = np.array(trajectory.positions_xyz[0, :])
    bootstrap_pos = initial_pos - 0.5*v0_spline
    # bootstrap_rot = R.from_euler("ZYX", [rot_z, 0, 0])
    bootstrap_rot = initial_rot
    bootstrap_pose = (bootstrap_rot.as_matrix().flatten(), bootstrap_pos.flatten())

    print(F'Rotation around z: {np.rad2deg(bootstrap_rot.as_euler("ZYX")[0])}')

    origin = se3.from_translation([0, 0, 0])
    left = se3.from_translation([0, 1, 0])
    front = se3.from_translation([1, 0, 0])

    sequence = [origin, origin, left, origin, front, origin]

    bootstrap_poses = [se3.mul(bootstrap_pose, s) for s in sequence]
    bootstrap_times = [i + t_zero for i in range(-len(sequence), 0)]
    bootstrapping_chunk = SE3Trajectory(bootstrap_times, bootstrap_poses)

    bootstrapping = SE3HermiteTrajectory()
    concatenated = bootstrapping_chunk.concat(target_chunk)
    bootstrapping.makeSpline(concatenated)

    v0_final = np.array(bootstrapping.deriv(t_zero)[1])

    print(F"v_0 ~ {np.round(v0_final, 2)}")
    print(F"||v_0|| ~ {np.round(np.linalg.norm(v0_final), 2)} m/s")

    dt = trajectory.timestamps[1] - trajectory.timestamps[0]
    prefix_trajectory = bootstrapping.discretize(dt)
    prefix_trajectory = prefix_trajectory.before(-dt + t_zero)  # before(..) logic means '<=' ---> do not include zero

    xyz = np.array(prefix_trajectory.getPositionTrajectory().milestones)
    xyzw = R.from_matrix(np.array(prefix_trajectory.getRotationTrajectory().milestones).reshape((-1, 3, 3))).as_quat()
    xyzw[:, :] = initial_rot.as_quat()
    wxyz = xyzw[:, [3, 0, 1, 2]]
    t = np.array(prefix_trajectory.times)
    xyz = np.concatenate((xyz, trajectory.positions_xyz))
    wxyz = np.concatenate((wxyz, trajectory.orientations_quat_wxyz))
    t = np.concatenate((t, trajectory.timestamps))
    new_trajectory = PoseTrajectory3D(xyz, wxyz, t)

    return new_trajectory


if __name__ == '__main__':
    main()
