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

    yaml_file = EnvYAML(args.configuration)
    tmp_yaml_filename = os.path.join(args.output_folder, 'tmp.yaml')

    print(F"Using the following 'evaluate' executable: {args.evaluate}")
    print(F"Processing the following datasets: {str.join(', ', (d['name'] for d in yaml_file['datasets']))}")
    print()

    N = len(yaml_file['datasets'])

    perf_evaluator = PerformanceEvaluator()
    traj_evaluator = TrajectoryEvaluator()

    try:
        for i, dataset in enumerate(yaml_file['datasets']):
            output_folder = F"{i:<03}_{dataset['name'].lower().replace(' ', '_')}"
            print(F"Processing dataset {i+1} of {N}, writing to {output_folder}")
            output_folder = os.path.join(args.output_folder, output_folder)

            create_temporary_params_yaml(dataset['params'], yaml_file['common_params'], tmp_yaml_filename)

            #  F" --events_topic {dataset['events_topic']}" \ # ADD ME LATER
            command = F"{args.evaluate}" \
                      F" --input_bag {dataset['rosbag']}" \
                      F" --image_topic {dataset['image_topic']}" \
                      F" --pose_topic {dataset['pose_topic']}" \
                      F" --imu_topic {dataset['imu_topic']}" \
                      F" --params_file {tmp_yaml_filename}" \
                      F" --output_folder {output_folder}"

            # when running from console this was necessary
            command = command.replace('\n', ' ')

            print(F"Running {command}")

            stream = os.popen(command)
            stream.read()  # waits for process to finish, captures stdout

            print(F"Running dataset {i+1} of {N} completed, analyzing outputs now...")

            df_poses = pd.read_csv(os.path.join(output_folder, "pose.csv"), delimiter=";")
            df_groundtruth = pd.read_csv(os.path.join(output_folder, "gt.csv"), delimiter=";")
            df_realtime = pd.read_csv(os.path.join(output_folder, "realtime.csv"), delimiter=";")
            with open(os.path.join(output_folder, "profiling.json"), "rb") as f:
                profiling_json = orjson.loads(f.read())

            traj_evaluator.evaluate(dataset['name'], df_poses, df_groundtruth)
            perf_evaluator.evaluate(df_realtime, profiling_json)

            print(F"Analysis of output {i+1} of {N} completed")

        traj_evaluator.print_summary()
        perf_evaluator.print_summary()

    finally:
        os.remove(tmp_yaml_filename)


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








