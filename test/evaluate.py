import os
import pickle
import sys

import argparse
import pandas as pd

from envyaml import EnvYAML
import yaml
# import orjson
from x_evaluate.evaluation_data import EvaluationDataSummary, EvaluationData
import x_evaluate.performance_evaluation as pe
import x_evaluate.trajectory_evaluation as te
import x_evaluate.tracking_evaluation as fe
from x_evaluate.utils import run_evaluate_cpp, envyaml_to_archive_dict, get_git_info, name_to_identifier


def main():
    print("Python evaluation script for X-library")
    print()
    print(F"Python executable: {sys.executable}")
    script_dir = os.path.dirname(os.path.realpath(__file__))
    print(F"Script located here: {script_dir}")
    print()

    parser = argparse.ArgumentParser(description='Automatic evaluation of X library according to evaluate.yaml')
    parser.add_argument('--evaluate', type=str, default="", help='location of c++ evaluate executable')
    parser.add_argument('--configuration', type=str, default="", help="YAML file specifying what to run")
    parser.add_argument('--dataset_dir', type=str, default="", help="substitutes XVIO_DATASET_DIR in yaml file")
    parser.add_argument('--output_folder', type=str, required=True)

    args = parser.parse_args()

    if len(args.evaluate) == 0:
        try:
            print("Calling catkin_find x_vio_ros evaluate")
            stream = os.popen('catkin_find x_vio_ros evaluate')
            args.evaluate = stream.read()
        finally:
            if len(args.evaluate) == 0:
                print("Error: could not find 'evaluate' executable")
                return

    if len(args.configuration) == 0:
        # default to evaluate.yaml in same folder as this script
        args.configuration = os.path.join(script_dir, "evaluate.yaml")

    if len(args.dataset_dir) > 0:
        os.environ['XVIO_DATASET_DIR'] = args.dataset_dir

    if 'XVIO_SRC_ROOT' not in os.environ:
        src_root = os.path.normpath(os.path.join(script_dir, '..'))
        print(F"Assuming '{src_root}' to be the xVIO source root")
        os.environ['XVIO_SRC_ROOT'] = src_root

    print(F"Reading '{args.configuration}'")

    eval_config = EnvYAML(args.configuration)
    tmp_yaml_filename = os.path.join(args.output_folder, 'tmp.yaml')

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    print(F"Using the following 'evaluate' executable: {args.evaluate}")
    print(F"Processing the following datasets: {str.join(', ', (d['name'] for d in eval_config['datasets']))}")
    print()

    N = len(eval_config['datasets'])

    summary = EvaluationDataSummary()

    conf = envyaml_to_archive_dict(eval_config)

    summary.configuration = conf

    try:
        for i, dataset in enumerate(eval_config['datasets']):
            output_folder = F"{i+1:>03}_{name_to_identifier(dataset['name'])}"
            print(F"Processing dataset {i+1} of {N}, writing to {output_folder}")
            output_folder = os.path.join(args.output_folder, output_folder)

            d = process_dataset(args.evaluate, dataset, output_folder, tmp_yaml_filename, eval_config)

            pe.plot_performance_plots(d, output_folder)
            te.plot_trajectory_plots(d, output_folder)
            fe.plot_feature_plots(d, output_folder)

            summary.data[dataset['name']] = d

            print(F"Analysis of output {i+1} of {N} completed")

        te.plot_summary_plots(summary, args.output_folder)
        te.print_trajectory_summary(summary)
        pe.plot_summary_plots(summary, args.output_folder)
        pe.print_realtime_factor_summary(summary)
        fe.plot_summary_plots(summary, args.output_folder)

        x_vio_ros_root = os.environ['XVIO_SRC_ROOT']
        x_root = os.path.normpath(os.path.join(x_vio_ros_root, "../x"))

        print(F"Assuming '{x_root}' to be the x source root")

        summary.x_git_info = get_git_info(x_root)
        summary.x_vio_ros_git_info = get_git_info(x_vio_ros_root)

        filename = os.path.join(args.output_folder, 'evaluation.pickle')

        print(F"Dumping evaluation results to '{filename}'")

        with open(filename, 'wb') as f:
            pickle.dump(summary, f, pickle.HIGHEST_PROTOCOL)

    finally:
        if os.path.exists(tmp_yaml_filename):
            os.remove(tmp_yaml_filename)


def process_dataset(executable, dataset, output_folder, tmp_yaml_filename, yaml_file) -> EvaluationData:

    d = EvaluationData()
    d.name = dataset['name']

    d.params = create_temporary_params_yaml(dataset, yaml_file['common_params'], tmp_yaml_filename)
    d.command = run_evaluate_cpp(executable, dataset['rosbag'], dataset['image_topic'], dataset['pose_topic'],
                                 dataset['imu_topic'], dataset['events_topic'], output_folder, tmp_yaml_filename,
                                 dataset['use_eklt'])

    print(F"Running dataset completed, analyzing outputs now...")

    gt_available = dataset['pose_topic'] is not None

    df_groundtruth, df_poses, df_realtime, df_features, df_resources = read_output_files(output_folder, gt_available)

    if df_groundtruth is not None:
        d.trajectory_data = te.evaluate_trajectory(df_poses, df_groundtruth)

    d.performance_data = pe.evaluate_computational_performance(df_realtime, df_resources)

    df_tracks = None

    if dataset['use_eklt']:
        df_events, df_optimize, df_tracks = read_eklt_output_files(output_folder)
        d.eklt_performance_data = pe.evaluate_ektl_performance(d.performance_data, df_events, df_optimize)

    d.feature_data = fe.evaluate_feature_tracking(d.performance_data, df_features, df_tracks)
    return d


def read_output_files(output_folder, gt_available):
    df_poses = pd.read_csv(os.path.join(output_folder, "pose.csv"), delimiter=";")
    df_features = pd.read_csv(os.path.join(output_folder, "features.csv"), delimiter=";")
    df_resources = pd.read_csv(os.path.join(output_folder, "resource.csv"), delimiter=";")
    df_groundtruth = None
    if gt_available:
        df_groundtruth = pd.read_csv(os.path.join(output_folder, "gt.csv"), delimiter=";")
    df_realtime = pd.read_csv(os.path.join(output_folder, "realtime.csv"), delimiter=";")

    # profiling_json = read_json_file(output_folder)
    return df_groundtruth, df_poses, df_realtime, df_features, df_resources


def read_eklt_output_files(output_folder):
    df_events = pd.read_csv(os.path.join(output_folder, "events.csv"), delimiter=";")
    df_optimizations = pd.read_csv(os.path.join(output_folder, "optimizations.csv"), delimiter=";")
    df_tracks = pd.read_csv(os.path.join(output_folder, "tracks.csv"), delimiter=";")
    return df_events, df_optimizations, df_tracks


# def read_json_file(output_folder):
#     profile_json_filename = os.path.join(output_folder, "profiling.json")
#     if os.path.exists(profile_json_filename):
#         with open(profile_json_filename, "rb") as f:
#             profiling_json = orjson.loads(f.read())
#     else:
#         profiling_json = None
#     return profiling_json


def create_temporary_params_yaml(dataset, common_params, tmp_yaml_filename):
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
        with open(tmp_yaml_filename, 'w') as tmp_yaml_file:
            yaml.dump(params, tmp_yaml_file)
    return params


if __name__ == '__main__':
    main()

