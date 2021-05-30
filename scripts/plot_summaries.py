import argparse
import glob
import os

from typing import List, Dict

from evo.core import metrics
from x_evaluate.evaluation_data import EvaluationDataSummary
from x_evaluate.plots import PlotType
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

    s = read_evaluation_pickle(args.input)

    i = 1

    for key, evaluation in s.data.items():
        output_folder = F"{i:>03}_{name_to_identifier(key)}"
        print(F"Plotting summary plots for '{key}' in subfolder '{output_folder}'")
        output_folder = os.path.join(output_root, output_folder)

        pe.plot_performance_plots(evaluation, output_folder)
        te.plot_trajectory_plots(evaluation, output_folder)
        fe.plot_feature_plots(evaluation, output_folder)

        i += 1

    te.plot_summary_plots(s, output_root)
    te.create_summary_info(s)
    pe.plot_summary_plots(s, output_root)
    pe.print_realtime_factor_summary(s)
    fe.plot_summary_plots(s, output_root)


if __name__ == '__main__':
    main()
