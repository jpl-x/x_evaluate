import os
import sys

import argparse

from envyaml import EnvYAML
# import orjson
from x_evaluate.evaluation_data import EvaluationDataSummary, FrontEnd
import x_evaluate.performance_evaluation as pe
import x_evaluate.trajectory_evaluation as te
import x_evaluate.tracking_evaluation as fe
from x_evaluate.utils import envyaml_to_archive_dict, name_to_identifier, \
    ArgparseKeyValueAction
from x_evaluate.scriptlets import get_git_info, process_dataset, write_evaluation_pickle


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
    parser.add_argument('--name', type=str, help="optional name, if not the output folder name is used")
    parser.add_argument('--dataset_dir', type=str, default="", help="substitutes XVIO_DATASET_DIR in yaml file")
    parser.add_argument('--skip_feature_tracking', help="Whether to do a 3d plot.", action="store_true",  default=False)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--frontend', type=FrontEnd, choices=list(FrontEnd), required=True)
    parser.add_argument('--overrides', nargs='*', action=ArgparseKeyValueAction)

    args = parser.parse_args()

    cmdline_override_params = dict()
    if args.overrides is not None:
        cmdline_override_params = args.overrides

    if len(args.evaluate) == 0:
        try:
            print("Calling catkin_find x_evaluate evaluate")
            stream = os.popen('catkin_find x_evaluate evaluate')
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

    if 'X_EVALUATE_SRC_ROOT' not in os.environ:
        src_root = os.path.normpath(os.path.join(script_dir, '..'))
        print(F"Assuming '{src_root}' to be the x_evaluate source root")
        os.environ['X_EVALUATE_SRC_ROOT'] = src_root

    print(F"Reading '{args.configuration}'")

    eval_config = EnvYAML(args.configuration)
    tmp_yaml_filename = os.path.join(args.output_folder, 'tmp.yaml')

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    print(F"Using the following 'evaluate' executable: {args.evaluate}")
    print(F"Processing the following datasets: {str.join(', ', (d['name'] for d in eval_config['datasets']))}")
    print()

    n = len(eval_config['datasets'])

    summary = EvaluationDataSummary()

    conf = envyaml_to_archive_dict(eval_config)

    summary.configuration = conf
    summary.frontend = args.frontend
    summary.name = args.name
    if summary.name is None:
        summary.name = os.path.basename(os.path.normpath(args.output_folder))

    try:
        for i, dataset in enumerate(eval_config['datasets']):
            output_folder = F"{i+1:>03}_{name_to_identifier(dataset['name'])}"
            print(F"Processing dataset {i+1} of {n}, writing to {output_folder}")
            output_folder = os.path.join(args.output_folder, output_folder)

            d = process_dataset(args.evaluate, dataset, output_folder, tmp_yaml_filename, eval_config,
                                cmdline_override_params, args.frontend, args.skip_feature_tracking)

            pe.plot_performance_plots(d, output_folder)
            te.plot_trajectory_plots(d.trajectory_data, d.name, output_folder)
            fe.plot_feature_plots(d, output_folder)

            summary.data[dataset['name']] = d

            print(F"Analysis of output {i+1} of {n} completed")

        te.plot_summary_plots(summary, args.output_folder)
        te.create_summary_info(summary, args.output_folder)
        pe.plot_summary_plots(summary, args.output_folder)
        pe.print_realtime_factor_summary(summary)
        fe.plot_summary_plots(summary, args.output_folder)

        x_evaluate_root = os.environ['X_EVALUATE_SRC_ROOT']
        x_root = os.path.normpath(os.path.join(x_evaluate_root, "../x"))

        print(F"Assuming '{x_root}' to be the x source root")

        summary.x_git_info = get_git_info(x_root)
        summary.x_vio_ros_git_info = get_git_info(x_evaluate_root)

    finally:
        if summary is not None:
            write_evaluation_pickle(summary, args.output_folder)

        # if os.path.exists(tmp_yaml_filename):
        #     os.remove(tmp_yaml_filename)


if __name__ == '__main__':
    main()

