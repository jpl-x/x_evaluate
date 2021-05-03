import os
import pickle
from typing import Dict

import git
import numpy as np
from envyaml import EnvYAML
from evo.core.trajectory import PoseTrajectory3D
from matplotlib import pyplot as plt

from x_evaluate.evaluation_data import EvaluationDataSummary, GitInfo


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
    f.set_size_inches(10, 7)

    for i in range(len(data)):

        # this causes issues, quick fix:
        label = labels[i]
        if label.startswith('_'):
            label = label[1:]

        if isinstance(time, list):
            plt.plot(time[i], data[i], label=label)
        else:
            plt.plot(time, data[i], label=label)

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


def read_evaluation_pickle(input_folder, filename="evaluation.pickle") -> EvaluationDataSummary:
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

    return command


def timestamp_to_rosbag_time(timestamps, df_rt):
    return np.interp(timestamps, df_rt['ts_real'], df_rt['t_sim'])


def timestamp_to_rosbag_time_zero(timestamps, df_rt):
    return timestamp_to_rosbag_time(timestamps, df_rt) - df_rt['t_sim'][0]


def timestamp_to_real_time(timestamps, df_rt):
    return np.interp(timestamps, df_rt['ts_real'], df_rt['t_real'])


def envyaml_to_archive_dict(eval_config: EnvYAML) -> Dict:
    conf = eval_config.export()
    # remove environment keys to protect privacy
    for k in os.environ.keys():
        conf.pop(k)
    return conf


def get_git_info(path) -> GitInfo:
    x = git.Repo(path)
    return GitInfo(branch=x.active_branch.name,
                   last_commit=x.head.object.hexsha,
                   files_changed=len(x.index.diff(None)) > 0)


def name_to_identifier(name):
    return name.lower().replace(' ', '_')
