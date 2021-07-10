import argparse
import os

import yaml

from x_evaluate.utils import name_to_identifier
from x_evaluate.scriptlets import read_evaluation_pickle

import x_evaluate.tracking_evaluation as fe
import x_evaluate.performance_evaluation as pe
import x_evaluate.trajectory_evaluation as te


def main():
    parser = argparse.ArgumentParser(description='Reads evaluation.pickle and plots all summary plots')
    parser.add_argument('--input', type=str, required=True)

    args = parser.parse_args()

    output_root = os.path.dirname(args.input)
    filename = os.path.basename(args.input)

    s = read_evaluation_pickle(output_root, filename)

    i = 1

    for key, evaluation in s.data.items():
        output_folder = F"{i:>03}_{name_to_identifier(key)}"
        print(F"Plotting summary plots for '{key}' in subfolder '{output_folder}'")
        output_folder = os.path.join(output_root, output_folder)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        pe.plot_performance_plots(evaluation, output_folder)
        te.plot_trajectory_plots(evaluation, output_folder)
        fe.plot_feature_plots(evaluation, output_folder)

        params_yaml_file = os.path.join(output_folder, "params.yaml")
        if not os.path.exists(params_yaml_file):
            with open(params_yaml_file, 'w') as tmp_yaml_file:
                yaml.dump(evaluation.params, tmp_yaml_file)

        i += 1

    te.plot_summary_plots(s, output_root)
    te.create_summary_info(s)
    pe.plot_summary_plots(s, output_root)
    pe.print_realtime_factor_summary(s)
    fe.plot_summary_plots(s, output_root)


if __name__ == '__main__':
    main()
