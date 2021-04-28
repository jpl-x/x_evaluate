import os
import pickle

import numpy as np
from evo.core.trajectory import PoseTrajectory3D
from matplotlib import pyplot as plt

from x_evaluate.evaluation_data import EvaluationDataSummary


def convert_to_evo_trajectory(df_poses, prefix="") -> PoseTrajectory3D:
    xyz_est = df_poses[[prefix + 'p_x', prefix + 'p_y', prefix + 'p_z']].to_numpy()
    wxyz_est = df_poses[[prefix + 'q_w', prefix + 'q_x', prefix + 'q_y', prefix + 'q_z']].to_numpy()
    return PoseTrajectory3D(xyz_est, wxyz_est, df_poses[['t']].to_numpy())


def boxplot(filename, data, labels, title="", outlier_params=1.5):
    f = plt.figure()
    f.set_size_inches(10, 7)
    # WHIS explanation see https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.boxplot.html, it's worth it
    plt.boxplot(data, vert=True, labels=labels, whis=outlier_params)
    plt.title(title)

    if filename is None or len(filename) == 0:
        plt.show()
    else:
        plt.savefig(filename)
    plt.clf()
    plt.close(f)


def time_series_plot(filename, time, data, labels, title="", ylabel=None):
    f = plt.figure()
    for i in range(len(data)):
        if isinstance(time, list):
            plt.plot(time[i], data[i], label=labels[i])
        else:
            plt.plot(time, data[i], label=labels[i])

    plt.legend()
    plt.title(title)
    plt.xlabel("Time [s]")

    if ylabel is not None:
        plt.ylabel(ylabel)

    if filename is None or len(filename) == 0:
        plt.show()
    else:
        plt.savefig(filename)
    plt.clf()
    plt.close(f)


def read_evaluation_pickle(input_folder, filename) -> EvaluationDataSummary:
    file = os.path.join(input_folder, filename)
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


def run_evaluate_cpp(executable, rosbag, image_topic, pose_topic, imu_topic, events_topic, output_folder, params_file,
                     use_eklt):
    if pose_topic is None:
        pose_topic = "\"\""
    if events_topic is None:
        events_topic = "\"\""

    command = F"{executable}" \
              F" --input_bag {rosbag}" \
              F" --image_topic {image_topic}" \
              F" --pose_topic {pose_topic}" \
              F" --imu_topic {imu_topic}" \
              F" --events_topic {events_topic}" \
              F" --params_file {params_file}" \
              F" --output_folder {output_folder}"
    if use_eklt:
        command = command + " --use_eklt"
    # when running from console this was necessary
    command = command.replace('\n', ' ')
    print(F"Running {command}")
    stream = os.popen(command)
    out = stream.read()  # waits for process to finish, captures stdout
    print("################### <STDOUT> ################")
    print(out)
    print("################### </STDOUT> ################")


def timestamp_to_rosbag_time(timestamps, df_rt):
    return np.interp(timestamps, df_rt['ts_real'], df_rt['t_sim'])


def timestamp_to_rosbag_time_zero(timestamps, df_rt):
    return timestamp_to_rosbag_time(timestamps, df_rt) - df_rt['t_sim'][0]


def timestamp_to_real_time(timestamps, df_rt):
    return np.interp(timestamps, df_rt['ts_real'], df_rt['t_real'])
