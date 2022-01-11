import argparse
import logging
import os

import numpy as np
from matplotlib import pyplot as plt

from rosbag import Bag

from x_evaluate.math_utils import calculate_velocities
from x_evaluate.plots import PlotContext, DEFAULT_COLORS, \
    plot_moving_boxplot_in_time_from_stats
from x_evaluate.scriptlets import cache
from x_evaluate.utils import get_ros_topic_name_from_msg_type, read_all_ros_msgs_from_topic_into_dict, \
    convert_t_xyz_wxyz_to_evo_trajectory, n_to_grid_size, get_quantized_statistics_along_axis


def main():
    parser = argparse.ArgumentParser(description="Plots event rate from BAG")
    parser.add_argument('--inputs', nargs='+', required=True, type=str)

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')

    args = parser.parse_args()

    def get_trajectories_from_file():
        trajectories = {}
        for input in args.inputs:
            input_bag = Bag(input, 'r')
            try:
                trajectory = read_gt_trajectory_from_ros_bag(input, input_bag)
            except LookupError as e:
                logging.warning("No GT topic found in file '%s' (%s)", input, e)
                continue
            trajectories[input] = trajectory
        return trajectories

    def get_imu_measurements_from_file():
        imu_measurements = {}
        for input in args.inputs:
            input_bag = Bag(input, 'r')
            try:
                imu_measurement = read_imu_measurements_from_file(input, input_bag)
            except LookupError as e:
                logging.warning("No IMU topic found in file '%s' (%s)", input, e)
                continue
            imu_measurements[input] = imu_measurement
        return imu_measurements

    trajectories = cache("tmp_trajectory_cache.pickle", get_trajectories_from_file)
    imu_measurements = cache("tmp_imu_data_cache.pickle", get_imu_measurements_from_file)
    # trajectories = get_trajectories_from_file()
    # imu_measurements = get_imu_measurements_from_file()

    # # temporary filter
    # trajectories = {k: v for k, v in trajectories.items() if 'poster_translation' in k}
    # imu_measurements = {k: v for k, v in imu_measurements.items() if 'poster_translation' in k}
    #
    # plot_trajectory_dynamics(trajectories, imu_measurements)
    plot_trajectory_dynamics({k: v for k, v in trajectories.items() if 'translation' in k},
                             {k: v for k, v in imu_measurements.items() if 'translation' in k})
    plot_trajectory_dynamics({k: v for k, v in trajectories.items() if 'rotation' in k},
                             {k: v for k, v in imu_measurements.items() if 'rotation' in k})
    plot_trajectory_dynamics({k: v for k, v in trajectories.items() if 'hdr' in k},
                             {k: v for k, v in imu_measurements.items() if 'hdr' in k})

    # with PlotContext(subplot_rows=rows, subplot_cols=cols) as pc:
    #
    #     for file, v in velocities.items():
    #         t = v[0]
    #         lin_vel = v[1]
    #         ang_vel = v[2]
    #
    #         t_quantized, stats = get_quantized_statistics_along_axis(t, ang_vel, resolution=0.1)
    #         plot_moving_boxplot_in_time_from_stats(pc, t_quantized, stats, F"Angular velocity norms of '"
    #                                                                        F"{os.path.basename(file)}'", "rad/s",
    #                                                DEFAULT_COLORS[1])

            # ax = pc.get_axis()
            # ax_right = ax.twinx()
            #
            # lin_vel_label = "linear velocity"
            # lin_vel_line = ax.plot(t, lin_vel, label=lin_vel_label, color=DEFAULT_COLORS[0])
            # ang_vel_label = "rotational velocity"
            # ang_vel_line = ax_right.plot(t, ang_vel, label=ang_vel_label, color=DEFAULT_COLORS[1])
            #
            # align_yaxis(ax_right, ax)
            #
            # # https://stackoverflow.com/a/5487005
            # ax_right.legend(lin_vel_line + ang_vel_line, [lin_vel_label, ang_vel_label])
            # ax.set_title(F"Linear and rotational velocity norms of '{os.path.basename(file)}'")
            # ax.set_xlabel("$t [s]$")
            # ax.set_ylabel("$m/s$")
            # ax_right.set_ylabel("$rad/s$")

            # time_series_plot(pc, t, [lin_vel, ang_vel], ["Linear velocity norm (m/s)", "Angular velocity norm (rad/s)"])

        # times = [v[0] for v in velocities]
        # vels = [v[1] for v in velocities]
        # time_series_plot(pc, times, vels, [os.path.basename(f) for f in trajectories.keys()])

    plt.show()
    #
    # with PlotContext(subplot_rows=2, subplot_cols=2) as pc:
    #     def read_from_dict(file: str):
    #         return trajectories[file]
    #     v = EvoTrajectoryVisualizer(pc, list(trajectories.keys()), read_from_dict)
    #     v.plot_current_trajectory()

    # event_times = np.array([e.ts.to_sec() for ea in gt_msgs.values() for e in ea.events])
    #
    # start = input_bag.get_start_time()
    # end = input_bag.get_end_time()
    #
    # event_times -= start
    #
    # bins = np.arange(start, end, 1.0)
    # events_per_sec, t = np.histogram(event_times, bins=bins)
    #
    # with PlotContext() as pc:
    #     ax = pc.get_axis()
    #     ax.set_title(F"Events per second")
    #     ax.plot(t[1:], events_per_sec)
    #     ax.set_xlabel("time")
    #     ax.set_ylabel("events/s")
    #
    # # block for visualization
    # plt.show()


def plot_trajectory_dynamics(trajectories, imu_measurements):
    rows, cols = n_to_grid_size(len(trajectories) * 3)
    velocities = {file: calculate_velocities(t) for file, t in trajectories.items()}
    with PlotContext(subplot_rows=rows, subplot_cols=cols) as pc:
        for file, v in velocities.items():
            t = v[0]
            lin_vel = np.linalg.norm(v[1], axis=1)
            ang_vel = np.linalg.norm(v[2], axis=1)

            imu_data = imu_measurements[file]
            t_imu = imu_data[:, 0]
            omega_imu = np.linalg.norm(imu_data[:, 4:], axis=1)

            t_quantized, stats = get_quantized_statistics_along_axis(t, lin_vel, resolution=0.1)
            plot_moving_boxplot_in_time_from_stats(pc, t_quantized, stats, F"Linear velocity norms of '"
                                                                           F"{os.path.basename(file)}'", "m/s",
                                                   DEFAULT_COLORS[0])

            # t_quantized, stats = get_quantized_statistics_along_axis(t[::5], ang_vel, resolution=0.1)
            t_quantized, stats = get_quantized_statistics_along_axis(t, ang_vel, resolution=0.1)
            plot_moving_boxplot_in_time_from_stats(pc, t_quantized, stats, F"Angular velocity norms of '"
                                                                           F"{os.path.basename(file)}'", "rad/s",
                                                   DEFAULT_COLORS[1], plot_min_max_shade=False)

            t_quantized, stats = get_quantized_statistics_along_axis(t_imu, omega_imu, resolution=0.1)
            plot_moving_boxplot_in_time_from_stats(pc, t_quantized, stats, F"Angular velocity norms from IMU "
                                                                           F"measurements '"
                                                                           F"{os.path.basename(file)}'", "rad/s",
                                                   DEFAULT_COLORS[2])


def read_gt_trajectory_from_ros_bag(input, input_bag):
    t_0 = input_bag.get_start_time()
    gt_topic = get_ros_topic_name_from_msg_type(input_bag, 'geometry_msgs/PoseStamped')
    gt_msgs = read_all_ros_msgs_from_topic_into_dict(gt_topic, input_bag)
    t_xyz_wxyz = np.empty((len(gt_msgs), 8))
    logging.info("Read %d messages from topic '%s' in file '%s'", len(gt_msgs), gt_topic, input)
    t_xyz_wxyz[:, 0] = [m.header.stamp.to_sec() for m in gt_msgs.values()]
    t_xyz_wxyz[:, 0] -= t_0
    t_xyz_wxyz[:, 1] = [m.pose.position.x for m in gt_msgs.values()]
    t_xyz_wxyz[:, 2] = [m.pose.position.y for m in gt_msgs.values()]
    t_xyz_wxyz[:, 3] = [m.pose.position.z for m in gt_msgs.values()]
    t_xyz_wxyz[:, 4] = [m.pose.orientation.w for m in gt_msgs.values()]
    t_xyz_wxyz[:, 5] = [m.pose.orientation.x for m in gt_msgs.values()]
    t_xyz_wxyz[:, 6] = [m.pose.orientation.y for m in gt_msgs.values()]
    t_xyz_wxyz[:, 7] = [m.pose.orientation.z for m in gt_msgs.values()]
    trajectory = convert_t_xyz_wxyz_to_evo_trajectory(t_xyz_wxyz)
    return trajectory


def read_imu_measurements_from_file(input, input_bag):
    t_0 = input_bag.get_start_time()
    imu_topic = get_ros_topic_name_from_msg_type(input_bag, 'sensor_msgs/Imu')
    imu_msgs = read_all_ros_msgs_from_topic_into_dict(imu_topic, input_bag)
    t_linacc_angvel = np.empty((len(imu_msgs), 7))
    logging.info("Read %d messages from topic '%s' in file '%s'", len(imu_msgs), imu_topic, input)
    t_linacc_angvel[:, 0] = [t for t in imu_msgs.keys()]
    t_linacc_angvel[:, 0] -= t_0
    t_linacc_angvel[:, 1] = [m.linear_acceleration.x for m in imu_msgs.values()]
    t_linacc_angvel[:, 2] = [m.linear_acceleration.y for m in imu_msgs.values()]
    t_linacc_angvel[:, 3] = [m.linear_acceleration.z for m in imu_msgs.values()]
    t_linacc_angvel[:, 4] = [m.angular_velocity.x for m in imu_msgs.values()]
    t_linacc_angvel[:, 5] = [m.angular_velocity.y for m in imu_msgs.values()]
    t_linacc_angvel[:, 6] = [m.angular_velocity.z for m in imu_msgs.values()]
    # trajectory = convert_t_xyz_wxyz_to_evo_trajectory(t_linacc_angvel)
    return t_linacc_angvel


if __name__ == '__main__':
    main()
