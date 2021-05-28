import argparse
import glob
import os

from typing import Dict

from evo.core import metrics

from x_evaluate.comparisons import identify_common_datasets, compare_trajectory_performance_wrt_traveled_dist
from x_evaluate.evaluation_data import EvaluationDataSummary, FrontEnd
from x_evaluate.plots import PlotType, PlotContext
from x_evaluate.utils import name_to_identifier, n_to_grid_size
from x_evaluate.scriptlets import read_evaluation_pickle

import x_evaluate.performance_evaluation as pe
import x_evaluate.trajectory_evaluation as te
import x_evaluate.tracking_evaluation as fe


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

    common_datasets = identify_common_datasets(summaries)

    names = [s.name for s in summaries.values()]
    eklt_names = [s.name for s in summaries.values() if s.frontend == FrontEnd.EKLT]

    eklt_summaries = []
    for s in summaries.values():
        if s.frontend == FrontEnd.EKLT:
            eklt_summaries.append(s)

    print(F"Comparing {', '.join(summaries.keys())} on following datasets: {', '.join(common_datasets)}")

    # Computational performance
    #   - [x] Overall time
    #   - [x] Time / event
    #   - [x] Events / s
    #   - [x] Optimizations / s
    #   - [x] CPU usage
    #   - [x] Memory usage
    # EKLT performance
    #   - [x] Optimization iterations
    #   - [x] Number of tracked features
    #   - [x] Feature age
    #   - [x] Pixel tracking accuracy:
    #      - [x] w.r.t. KLT groundtruth on real data
    #      - [x] w.r.t. reprojection on simulated data
    #   - [ ] Pixel change histograms
    #   - [ ] Tracking error over traveled distance
    # Backend performance
    #   - [ ] SLAM / MSCKF / Opp number of features
    #   - [ ] Pose: APE, RPE, final pose error
    #   - [ ] Boxplots
    #   - [ ] Error in time

    #   - [x] Overall time
    with PlotContext(os.path.join(args.output_folder, F"compare_all_processing_times.svg")) as pc:
        pe.plot_processing_times(pc, summaries, common_datasets)

    #   - [x] CPU usage
    with PlotContext(os.path.join(args.output_folder, F"compare_all_cpu_usage.svg")) as pc:
        pe.plot_cpu_usage_boxplot_comparison(pc, summaries, common_datasets)

    #   - [x] Memory usage
    with PlotContext(os.path.join(args.output_folder, F"compare_all_memory_usage.svg")) as pc:
        pe.plot_memory_usage_boxplot_comparison(pc, summaries, common_datasets)

    #   - [x] Pixel tracking accuracy
    with PlotContext(os.path.join(args.output_folder, F"compare_all_feature_tracking.svg")) as pc:
        fe.plot_xvio_feature_tracking_comparison_boxplot(pc, list(summaries.values()), common_datasets)

    #   - [x] Feature update interval
    with PlotContext(os.path.join(args.output_folder, F"compare_all_feature_update_intervals.svg")) as pc:
        fe.plot_xvio_feature_update_interval_comparison_boxplot(pc, list(summaries.values()), common_datasets)

    if len(eklt_summaries) > 0:
        #   - [x] Optimization iterations
        with PlotContext(os.path.join(args.output_folder, F"compare_all_eklt_optimization_iterations.svg")) as pc:
            pe.plot_optimization_iterations_comparison(pc, eklt_summaries, common_datasets)

        #   - [x] Feature age
        with PlotContext(os.path.join(args.output_folder, F"compare_all_eklt_feature_ages.svg")) as pc:
            fe.plot_eklt_feature_age_comparison(pc, eklt_summaries, common_datasets)

        #   - [x] Feature update rate
        with PlotContext(os.path.join(args.output_folder, F"compare_all_eklt_feature_update_interval.svg")) as pc:
            fe.plot_eklt_feature_update_interval_comparison_boxplot(pc, eklt_summaries, common_datasets)

    for dataset in common_datasets:
        d_id = name_to_identifier(dataset)

        evaluations = [s.data[dataset] for s in summaries.values()]
        rows, cols = n_to_grid_size(len(evaluations))
        eklt_evaluations = [s.data[dataset] for s in summaries.values() if s.frontend == FrontEnd.EKLT]

        if len(eklt_evaluations) > 0:

            #   - [x] Time / event
            with PlotContext(os.path.join(args.output_folder, F"compare_{d_id}_event_processing_times.svg"),
                             subplot_rows=len(eklt_evaluations), base_height_inch=3) as pc:
                pc.figure.suptitle(F"Event processing times on '{dataset}'")
                pe.plot_event_processing_times(pc, eklt_evaluations, eklt_names)

            #   - [x] Events / s
            with PlotContext(os.path.join(args.output_folder, F"compare_{d_id}_events_per_second.svg")) as pc:
                pe.plot_events_per_seconds_comparison(pc, eklt_evaluations, eklt_names, dataset)

            #   - [x] Optimizations / s
            with PlotContext(os.path.join(args.output_folder, F"compare_{d_id}_optimizations_per_second.svg")) as pc:
                pe.plot_optimizations_per_seconds_comparison(pc, eklt_evaluations, eklt_names, dataset)

            #   - [x] Number of tracked features
            with PlotContext(os.path.join(args.output_folder, F"compare_{d_id}_eklt_num_features.svg")) as pc:
                fe.plot_eklt_num_features_comparison(pc, eklt_evaluations, eklt_names, dataset)

            #   - [x] Pixel tracking accuracy
            with PlotContext(os.path.join(args.output_folder, F"compare_{d_id}_eklt_feature_tracking.svg")) as pc:
                fe.plot_eklt_feature_tracking_comparison(pc, eklt_evaluations, eklt_names, dataset)

        #   - [x] CPU usage
        with PlotContext(os.path.join(args.output_folder, F"compare_{d_id}_cpu_usage_in_time.svg")) as pc:
            pe.plot_cpu_usage_in_time_comparison(pc, evaluations, names, dataset)

        #   - [x] Memory usage
        with PlotContext(os.path.join(args.output_folder, F"compare_{d_id}_memory_usage_in_time.svg")) as pc:
            pe.plot_memory_usage_in_time_comparison(pc, evaluations, names, dataset)

        #   - [x] Pixel tracking accuracy
        with PlotContext(os.path.join(args.output_folder, F"compare_{d_id}_feature_tracking.svg")) as pc:
            fe.plot_xvio_feature_tracking_comparison(pc, evaluations, names, dataset)

        #   - [x] Feature update interval
        with PlotContext(os.path.join(args.output_folder, F"compare_{d_id}_xvio_feature_update_interval.svg")) as pc:
            fe.plot_xvio_feature_update_interval_in_time(pc, evaluations, names, dataset)

        with PlotContext(os.path.join(args.output_folder, F"compare_ape_{d_id}.svg")) as pc:
            te.plot_error_comparison(pc, evaluations, str(metrics.APE(metrics.PoseRelation.translation_part)),
                                     PlotType.TIME_SERIES, names)

        with PlotContext(os.path.join(args.output_folder, F"compare_rt_factor_{d_id}.svg")) as pc:
            pe.plot_realtime_factor(pc, evaluations, names)

        with PlotContext(os.path.join(args.output_folder, F"compare_rt_factor_{d_id}_subplot.svg"), subplot_rows=rows,
                         subplot_cols=cols) as pc:
            for k, s in summaries.items():
                pe.plot_realtime_factor(pc, [s.data[dataset]], [k])

        with PlotContext(os.path.join(args.output_folder, F"compare_rpg_errors_{d_id}.svg"), subplot_cols=2) as pc:
            te.plot_rpg_error_arrays(pc, evaluations, names)

        with PlotContext(os.path.join(args.output_folder, F"compare_rpg_errors_{d_id}_log.svg"), subplot_cols=2) as pc:
            te.plot_rpg_error_arrays(pc, evaluations, names, use_log=True)

    result_table = compare_trajectory_performance_wrt_traveled_dist(summaries)
    print(result_table.to_latex(index=False))


if __name__ == '__main__':
    main()
