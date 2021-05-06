import os
import pickle

import git

from x_evaluate.evaluation_data import EvaluationDataSummary, GitInfo


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


def read_evaluation_pickle(file) -> EvaluationDataSummary:
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


def get_git_info(path) -> GitInfo:
    x = git.Repo(path)
    return GitInfo(branch=x.active_branch.name,
                   last_commit=x.head.object.hexsha,
                   files_changed=len(x.index.diff(None)) > 0)