import copy
import os
import pickle

import git
import yaml

from x_evaluate import performance_evaluation as pe, tracking_evaluation as fe, trajectory_evaluation as te
from x_evaluate.evaluation_data import EvaluationDataSummary, GitInfo, FrontEnd, EvaluationData
from x_evaluate.rpg_tracking_analysis.evaluate_tracks import rpg_evaluate_tracks
from x_evaluate.utils import read_output_files, read_eklt_output_files, DynamicAttributes, \
    convert_eklt_to_rpg_tracks, convert_xvio_to_rpg_tracks


def run_evaluate_cpp(executable, rosbag, image_topic, pose_topic, imu_topic, events_topic, output_folder, params_file,
                     frontend, from_t=None, to_t=None):
    if pose_topic is None:
        pose_topic = "\"\""
    if events_topic is None:
        events_topic = "\"\""

    # command = F"gdbserver :1106 {executable}" \
    command = F"{executable}" \
              F" --input_bag {rosbag}" \
              F" --image_topic {image_topic}" \
              F" --pose_topic {pose_topic}" \
              F" --imu_topic {imu_topic}" \
              F" --events_topic {events_topic}" \
              F" --params_file {params_file}" \
              F" --output_folder {output_folder}" \
              F" --frontend {frontend}"

    if from_t:
        command += F" --from {from_t}"

    if to_t:
        command += F" --to {to_t}"
    # when running from console this was necessary
    command = command.replace('\n', ' ')
    print(F"Running {command}")
    stream = os.popen(command)
    out = stream.read()  # waits for process to finish, captures stdout
    print("################### <STDOUT> ################")
    print(out)
    print("################### </STDOUT> ################")

    return command


def read_evaluation_pickle(folder, filename="evaluation.pickle") -> EvaluationDataSummary:
    file = os.path.join(folder, filename)
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


def get_git_info(path) -> GitInfo:
    x = git.Repo(path)
    return GitInfo(branch=x.active_branch.name,
                   last_commit=x.head.object.hexsha,
                   files_changed=len(x.index.diff(None)) > 0)


def does_key_exist(dataset, key):
    return (key in dataset.keys()) and dataset[key] is not None


def get_param_if_exists(dataset, key):
    if does_key_exist(dataset, key):
        return dataset[key]
    return None


def process_dataset(executable, dataset, output_folder, tmp_yaml_filename, yaml_file, cmdline_override_params,
                    frontend: FrontEnd) -> EvaluationData:

    d = EvaluationData()
    d.name = dataset['name']

    d.params = create_temporary_params_yaml(dataset, yaml_file['common_params'], tmp_yaml_filename, cmdline_override_params)
    d.command = run_evaluate_cpp(executable, dataset['rosbag'], dataset['image_topic'], dataset['pose_topic'],
                                 dataset['imu_topic'], dataset['events_topic'], output_folder, tmp_yaml_filename,
                                 frontend, get_param_if_exists(dataset, 'from'), get_param_if_exists(dataset, 'to'))

    print(F"Running dataset completed, analyzing outputs now...")

    gt_available = does_key_exist(dataset, 'pose_topic')

    df_groundtruth, df_poses, df_realtime, df_features,\
    df_resources, df_xvio_tracks, df_imu_bias = read_output_files(output_folder, gt_available)

    if df_groundtruth is not None:
        d.trajectory_data = te.evaluate_trajectory(df_poses, df_groundtruth, df_imu_bias)

    d.performance_data = pe.evaluate_computational_performance(df_realtime, df_resources)

    df_eklt_tracks = None

    if frontend == FrontEnd.EKLT:
        df_events, df_optimize, df_eklt_tracks = read_eklt_output_files(output_folder)
        d.eklt_performance_data = pe.evaluate_ektl_performance(d.performance_data, df_events, df_optimize)

        d.feature_data = fe.evaluate_feature_tracking(d.performance_data, df_features, df_eklt_tracks)

        # track_file = os.path.join(output_folder, "eklt_tracks.txt")
        # convert_eklt_to_rpg_tracks(df_eklt_tracks, track_file)
        # gt_tracks, error_data, tracker_config = call_rpg_feature_tracking_evaluation(dataset, track_file)
        # d.feature_data.eklt_tracks_gt = gt_tracks
        # d.feature_data.eklt_tracks_error = error_data
        # d.feature_data.eklt_tracking_evaluation_config = tracker_config
    else:
        d.feature_data = fe.evaluate_feature_tracking(d.performance_data, df_features, None)

    track_file = os.path.join(output_folder, "xvio_tracks.txt")
    convert_xvio_to_rpg_tracks(df_xvio_tracks, track_file)
    gt_tracks, error_data, tracker_config = call_rpg_feature_tracking_evaluation(dataset, track_file)
    d.feature_data.df_xvio_tracks = df_xvio_tracks
    d.feature_data.xvio_tracks_gt = gt_tracks
    d.feature_data.xvio_tracks_error = error_data
    d.feature_data.xvio_tracking_evaluation_config = tracker_config

    d.configuration = copy.deepcopy(dataset)
    return d


def call_rpg_feature_tracking_evaluation(dataset, track_file):
    args = DynamicAttributes()
    root_path = os.path.dirname(dataset["rosbag"])
    rosbag_name = os.path.basename(dataset["rosbag"])
    args.root = root_path  # Directory where datasets are found
    # args.dataset = None  # Params yaml-file for dataset
    args.file = track_file  # Tracks file for ground truth computation
    args.error_threshold = 10
    args.plot_3d = False
    args.plot_errors = False
    args.video_preview = False

    pose_topic = get_param_if_exists(dataset, 'pose_topic')
    depth_map_topic = get_param_if_exists(dataset, 'depth_map_topic')
    camera_info_topic = get_param_if_exists(dataset, 'camera_info_topic')

    if pose_topic and depth_map_topic and camera_info_topic:
        tracker_config = {
            "type": "reprojection"
        }
        dataset_config = {
            "type": "bag",
            "name": rosbag_name,
            "image_topic": dataset['image_topic'],
            "depth_map_topic": depth_map_topic,
            "camera_info_topic": camera_info_topic,
            "pose_topic": pose_topic,
        }
    else:
        tracker_config = {
            "type": "KLT",  # ['KLT', 'reprojection'] type of algorithm used.
            # "window_size": 21,  # window size of tracked patch
            # "num_pyramidal_layers": 1,  # number of layers in pyramidal search
            "window_size": 31,
            "num_pyramidal_layers": 2,
        }

        dataset_config = {
            "type": "bag",
            "name": rosbag_name,
            "image_topic": dataset['image_topic']
        }
    tracked_features, error_data = rpg_evaluate_tracks(args, dataset_config, tracker_config)
    return tracked_features, error_data, tracker_config


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


def write_evaluation_pickle(summary: EvaluationDataSummary, output_folder, filename="evaluation.pickle"):
    full_name = os.path.join(output_folder, filename)
    with open(full_name, 'wb') as f:
        pickle.dump(summary, f, pickle.HIGHEST_PROTOCOL)
