import argparse

import pandas as pd
import matplotlib.pyplot as plt

from x_evaluate.plots import plot_evo_trajectory_with_euler_angles, PlotContext, time_series_plot
from x_evaluate.utils import read_esim_trajectory_csv


def main():
    parser = argparse.ArgumentParser(description='Visualizing ESIM simulated IMU')

    parser.add_argument('--esim_imu_dump', type=str, default="/tmp/esim_imu_dump.csv")
    parser.add_argument('--esim_spline_dump', type=str, default="/tmp/esim_spline_dump.csv")
    args = parser.parse_args()

    trajectory = read_esim_trajectory_csv(args.esim_spline_dump)

    with PlotContext(subplot_cols=3, subplot_rows=3) as pc:
        plot_evo_trajectory_with_euler_angles(pc, trajectory, "ESIM Spline")

    imu = pd.read_csv(args.esim_imu_dump)

    with PlotContext(subplot_rows=2, subplot_cols=2) as pc:
        t = imu['t']
        fields = ['acc_actual_x', 'acc_actual_y', 'acc_actual_z']
        time_series_plot(pc, t, imu[fields].to_numpy().T, fields)
        fields = ['acc_corrupted_x', 'acc_corrupted_y', 'acc_corrupted_z']
        time_series_plot(pc, t, imu[fields].to_numpy().T, fields)
        fields = ['ang_vel_actual_x', 'ang_vel_actual_y', 'ang_vel_actual_z']
        time_series_plot(pc, t, imu[fields].to_numpy().T, fields)
        fields = ['ang_vel_corrupted_x', 'ang_vel_corrupted_y', 'ang_vel_corrupted_z']
        time_series_plot(pc, t, imu[fields].to_numpy().T, fields)

    plt.show()

if __name__ == '__main__':
    main()
