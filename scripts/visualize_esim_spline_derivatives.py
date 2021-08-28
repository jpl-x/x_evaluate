import argparse

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R

from x_evaluate.plots import plot_evo_trajectory_with_euler_angles, PlotContext, time_series_plot
from x_evaluate.trajectory_evaluation import plot_trajectory_with_gt_and_euler_angles
from x_evaluate.utils import read_esim_trajectory_csv, convert_t_xyz_wxyz_to_evo_trajectory


def vec_to_skew_mat(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


class DoubleIntegrator:
    def __init__(self, p_start, v_start):
        self.p = p_start
        self.v = v_start

    def propagate(self, a, delta_t):
        self.p += self.v * delta_t + 1/2 * delta_t**2 * a
        self.v += a * delta_t


def main():
    parser = argparse.ArgumentParser(description='Visualizing ESIM simulated IMU')

    parser.add_argument('--esim_spline_derivative', type=str, default="/tmp/esim_spline_derivative.csv")

    args = parser.parse_args()

    spline_derivatives = pd.read_csv(args.esim_spline_derivative)
    spline_derivatives[['qx_est', 'qy_est', 'qz_est', 'qw_est']] = [0, 0, 0, 1]
    spline_derivatives[['x_est', 'y_est', 'z_est']] = [0, 0, 0]

    p_start = spline_derivatives.loc[0, ['x', 'y', 'z']].to_numpy()
    delta_t = spline_derivatives.loc[1, ['t']].to_numpy() - spline_derivatives.loc[0, ['t']].to_numpy()
    v_start = (spline_derivatives.loc[1, ['x', 'y', 'z']].to_numpy() - p_start) / delta_t

    integrator = DoubleIntegrator(p_start, v_start)
    #
    # # This solves the issue
    # # spline_derivatives[['wx']] = -spline_derivatives[['wx']]
    #
    # spline_derivatives[['wx', 'wy', 'wz']] = -spline_derivatives[['wx', 'wy', 'wz']]

    current_rotation = R.from_quat(spline_derivatives.loc[0, ['qx', 'qy', 'qz', 'qw']])
    spline_derivatives.loc[0, ['qx_est', 'qy_est', 'qz_est', 'qw_est']] = current_rotation.as_quat()
    spline_derivatives.loc[0, ['x_est', 'y_est', 'z_est']] = p_start

    for i in tqdm.tqdm(range(len(spline_derivatives)-1)):
        delta_t = spline_derivatives.loc[i+1, ['t']].to_numpy() - spline_derivatives.loc[i, ['t']].to_numpy()
        a = spline_derivatives.loc[i, ['ax', 'ay', 'az']].to_numpy()
        integrator.propagate(np.matmul(current_rotation.as_matrix().T, a), delta_t)
        omega = spline_derivatives.loc[i, ['wx', 'wy', 'wz']].to_numpy()
        omega_cross = vec_to_skew_mat(omega)
        rot_mat = expm(omega_cross * delta_t)
        current_rotation = R.from_matrix(np.matmul(current_rotation.as_matrix(), rot_mat))
        spline_derivatives.loc[i + 1, ['qx_est', 'qy_est', 'qz_est', 'qw_est']] = current_rotation.as_quat()
        spline_derivatives.loc[i + 1, ['x_est', 'y_est', 'z_est']] = integrator.p

    t_xyz_wxyz = spline_derivatives[['t', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']].to_numpy()
    reference_trajectory = convert_t_xyz_wxyz_to_evo_trajectory(t_xyz_wxyz)

    t_xyz_wxyz_est = spline_derivatives[['t', 'x_est', 'y_est', 'z_est', 'qw_est', 'qx_est', 'qy_est',
                                         'qz_est']].to_numpy()
    estimated_trajectory = convert_t_xyz_wxyz_to_evo_trajectory(t_xyz_wxyz_est)

    with PlotContext(subplot_cols=3, subplot_rows=3) as pc:
        plot_evo_trajectory_with_euler_angles(pc, estimated_trajectory, "ESIM spline", reference_trajectory)

    # imu = pd.read_csv(args.esim_imu_dump)
    #
    # with PlotContext(subplot_rows=2, subplot_cols=2) as pc:
    #     t = imu['t']
    #     fields = ['acc_actual_x', 'acc_actual_y', 'acc_actual_z']
    #     time_series_plot(pc, t, imu[fields].to_numpy().T, fields)
    #     fields = ['acc_corrupted_x', 'acc_corrupted_y', 'acc_corrupted_z']
    #     time_series_plot(pc, t, imu[fields].to_numpy().T, fields)
    #     fields = ['ang_vel_actual_x', 'ang_vel_actual_y', 'ang_vel_actual_z']
    #     time_series_plot(pc, t, imu[fields].to_numpy().T, fields)
    #     fields = ['ang_vel_corrupted_x', 'ang_vel_corrupted_y', 'ang_vel_corrupted_z']
    #     time_series_plot(pc, t, imu[fields].to_numpy().T, fields)
    #
    # plt.show()
    print('Done')
    plt.show()


if __name__ == '__main__':
    main()
