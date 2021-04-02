import os
import sys

import argparse
import pandas as pd

from envyaml import EnvYAML
import yaml
import orjson

from x_evaluate.performance_evaluation import PerformanceEvaluator
from x_evaluate.trajectory_evaluation import TrajectoryEvaluator


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

    print(F"Using the following 'evaluate' executable: {args.evaluate}")
    print(F"Processing the following datasets: {str.join(', ', (d['name'] for d in eval_config['datasets']))}")
    print()

    N = len(eval_config['datasets'])

    perf_evaluator = PerformanceEvaluator()
    traj_evaluator = TrajectoryEvaluator()

    try:
        for i, dataset in enumerate(eval_config['datasets']):
            output_folder = F"{i+1:>03}_{dataset['name'].lower().replace(' ', '_')}"
            print(F"Processing dataset {i+1} of {N}, writing to {output_folder}")
            output_folder = os.path.join(args.output_folder, output_folder)

            process_dataset(args.evaluate, dataset, output_folder, perf_evaluator, tmp_yaml_filename, traj_evaluator,
                            eval_config)

            print(F"Analysis of output {i+1} of {N} completed")

        # traj_evaluator.plot_trajectory("45 Deg Carpet", "45 Deg Carpet")

        traj_evaluator.plot_summary_boxplot(args.output_folder)

        traj_evaluator.print_summary()
        perf_evaluator.print_summary()

    finally:
        os.remove(tmp_yaml_filename)


def process_dataset(executable, dataset, output_folder, perf_evaluator, tmp_yaml_filename, traj_evaluator, yaml_file):

    create_temporary_params_yaml(dataset['params'], yaml_file['common_params'], tmp_yaml_filename)

    run_evaluate_cpp(executable, dataset['rosbag'], dataset['image_topic'], dataset['pose_topic'],
                     dataset['imu_topic'], dataset['events_topic'], output_folder, tmp_yaml_filename,
                     dataset['use_eklt'])

    print(F"Running dataset completed, analyzing outputs now...")

    gt_available = dataset['pose_topic'] is not None

    df_groundtruth, df_poses, df_realtime, profiling_json = read_output_files(output_folder, gt_available)

    if df_groundtruth is not None:
        traj_evaluator.evaluate(dataset['name'], df_poses, df_groundtruth, output_folder)
    perf_evaluator.evaluate(df_realtime, profiling_json)


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
    stream.read()  # waits for process to finish, captures stdout


def read_output_files(output_folder, gt_available):
    df_poses = pd.read_csv(os.path.join(output_folder, "pose.csv"), delimiter=";")
    df_groundtruth = None
    if gt_available:
        df_groundtruth = pd.read_csv(os.path.join(output_folder, "gt.csv"), delimiter=";")
    df_realtime = pd.read_csv(os.path.join(output_folder, "realtime.csv"), delimiter=";")
    with open(os.path.join(output_folder, "profiling.json"), "rb") as f:
        profiling_json = orjson.loads(f.read())
    return df_groundtruth, df_poses, df_realtime, profiling_json


def create_temporary_params_yaml(base_params_filename, common_params, tmp_yaml_filename):
    with open(base_params_filename) as base_params_file:
        params = yaml.full_load(base_params_file)
        for k, c in common_params.items():
            if c != params[k]:
                print(F"Overwriting '{k}': '{params[k]}' --> '{c}'")
                params[k] = c
        with open(tmp_yaml_filename, 'w') as tmp_yaml_file:
            yaml.dump(params, tmp_yaml_file)


if __name__ == '__main__':
    main()

