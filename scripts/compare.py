import argparse
import glob
import os

from typing import List, Dict

from evo.core import metrics
from x_evaluate.evaluation_data import ErrorType, EvaluationDataSummary
from x_evaluate.plots import PlotType
from x_evaluate.utils import read_evaluation_pickle, name_to_identifier

import x_evaluate.tracking_evaluation as fe
import x_evaluate.performance_evaluation as pe
import x_evaluate.trajectory_evaluation as te


def main():
    parser = argparse.ArgumentParser(description='Comparison script for dealing with evaluation.pickle files')
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--sub_folders', type=str)
    parser.add_argument('--output_folder', type=str)

    args = parser.parse_args()

    input_folders = []
    if args.sub_folders is not None:
        input_folders = args.sub_folders.split(':')
        input_folders = [os.path.join(args.input_folder, f) for f in input_folders]
    else:
        matches = glob.glob(os.path.join(args.input_folder, "*"))
        matches.sort()
        for f in matches:
            if os.path.isdir(f) and os.path.isfile(os.path.join(f, "evaluation.pickle")):
                input_folders.append(f)

    input_folders = [os.path.normpath(f) for f in input_folders]

    if args.output_folder is None:
        print(F"Using '{input_folders[0]}' as output_folder")
        args.output_folder = input_folders[0]

    summaries: Dict[str, EvaluationDataSummary] = {}

    for f in input_folders:
        name = str(f.split('/')[-1])
        # print("Reading", os.path.join(f, "evaluation.pickle"))
        s = read_evaluation_pickle(os.path.join(f, "evaluation.pickle"))
        summaries[name] = s
        # print(args.input_folder)

    common_datasets = None

    for s in summaries.values():
        if common_datasets is None:
            common_datasets = set(s.data.keys())
        common_datasets = common_datasets.intersection(s.data.keys())

    print(F"Comparing {', '.join(summaries.keys())} on following datasets: {', '.join(common_datasets)}")

    names = list(summaries.keys())

    for dataset in common_datasets:
        d_id = name_to_identifier(dataset)

        evaluations = [s.data[dataset] for s in summaries.values()]

        file = os.path.join(args.output_folder, F"compare_ape_{d_id}.svg")

        te.plot_error_comparison(evaluations, file, metrics.PoseRelation.full_transformation,
                                 ErrorType.APE, PlotType.TIME_SERIES, names)

        file = os.path.join(args.output_folder, F"compare_rt_factor_{d_id}.svg")
        pe.plot_realtime_factor(evaluations, file, names)

        file = os.path.join(args.output_folder, F"compare_optimizations_{d_id}.svg")
        pe.plot_optimization_iterations(evaluations, file)
    

if __name__ == '__main__':
    main()
