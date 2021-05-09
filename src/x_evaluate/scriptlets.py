import os
import pickle

import git
import yaml

from x_evaluate import trajectory_evaluation as te, performance_evaluation as pe, tracking_evaluation as fe
from x_evaluate.evaluation_data import EvaluationDataSummary, GitInfo, FrontEnd, EvaluationData
from x_evaluate.utils import read_output_files, read_eklt_output_files


def run_evaluate_cpp(executable, rosbag, image_topic, pose_topic, imu_topic, events_topic, output_folder, params_file,
                     frontend):
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
              F" --output_folder {output_folder}" \
              F" --frontend {frontend}"
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


def process_dataset(executable, dataset, output_folder, tmp_yaml_filename, yaml_file, cmdline_override_params,
                    frontend: FrontEnd) -> EvaluationData:

    d = EvaluationData()
    d.name = dataset['name']

    d.params = create_temporary_params_yaml(dataset, yaml_file['common_params'], tmp_yaml_filename, cmdline_override_params)
    d.command = run_evaluate_cpp(executable, dataset['rosbag'], dataset['image_topic'], dataset['pose_topic'],
                                 dataset['imu_topic'], dataset['events_topic'], output_folder, tmp_yaml_filename,
                                 frontend)

    print(F"Running dataset completed, analyzing outputs now...")

    gt_available = dataset['pose_topic'] is not None

    df_groundtruth, df_poses, df_realtime, df_features, df_resources = read_output_files(output_folder, gt_available)

    if df_groundtruth is not None:
        d.trajectory_data = te.evaluate_trajectory(df_poses, df_groundtruth)

    d.performance_data = pe.evaluate_computational_performance(df_realtime, df_resources)

    df_tracks = None

    if frontend == FrontEnd.EKLT:
        df_events, df_optimize, df_tracks = read_eklt_output_files(output_folder)
        d.eklt_performance_data = pe.evaluate_ektl_performance(d.performance_data, df_events, df_optimize)

    d.feature_data = fe.evaluate_feature_tracking(d.performance_data, df_features, df_tracks)
    return d


def create_temporary_params_yaml(dataset, common_params, tmp_yaml_filename, cmdline_override_params):
    base_params_filename = dataset['params']
    with open(base_params_filename) as base_params_file:
        params = yaml.full_load(base_params_file)
        for k, c in common_params.items():
            if c != params[k]:
                print(F"Overwriting '{k}': '{params[k]}' --> '{c}'")
                params[k] = c
        if 'override_params' in dataset.keys():
            for k, c in dataset['override_params'].items():
                if c != params[k]:
                    print(F"Overwriting '{k}': '{params[k]}' --> '{c}'")
                    params[k] = c
        for k, c in cmdline_override_params.items():
            if c != params[k]:
                print(F"Overwriting '{k}': '{params[k]}' --> '{c}'")
                params[k] = c
        with open(tmp_yaml_filename, 'w') as tmp_yaml_file:
            yaml.dump(params, tmp_yaml_file)
    return params