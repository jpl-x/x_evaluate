import argparse
import os

import numpy as np
import pandas as pd
from typing import Dict

from x_evaluate.comparisons import identify_common_datasets, identify_changing_parameters, \
    create_parameter_changes_table
from x_evaluate.evaluation_data import EvaluationDataSummary, FrontEnd
import x_evaluate.plots
from x_evaluate.plots import PlotType, ProgressPlotContextManager
from x_evaluate.utils import name_to_identifier, n_to_grid_size
from x_evaluate.scriptlets import read_evaluation_pickle, find_evaluation_files_recursively

import x_evaluate.performance_evaluation as pe
import x_evaluate.trajectory_evaluation as te
import x_evaluate.tracking_evaluation as fe


def main():
    parser = argparse.ArgumentParser(description='Comparison script for dealing with evaluation.pickle files')
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--sub_folders', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--use_paper_style_plots', action='store_true', default=False)
    parser.add_argument('--custom_rpe_distances', nargs='+', default=None, type=float)

    args = parser.parse_args()

    x_evaluate.plots.use_paper_style_plots = args.use_paper_style_plots

    input_folders = []
    if args.sub_folders is not None:
        input_folders = args.sub_folders.split(':')
        input_folders = [os.path.join(args.input_folder, f) for f in input_folders]
    else:
        root_folder = args.input_folder
        input_folders = find_evaluation_files_recursively(root_folder)
        input_folders = [os.path.dirname(f) for f in input_folders]

    # hack to remove uslam
    input_folders = [f for f in input_folders if 'uslam' not in f]

    output_folder = args.output_folder
    if output_folder is None:
        print(F"Using '{input_folders[0]}' as output_folder")
        output_folder = input_folders[0]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    summaries: Dict[str, EvaluationDataSummary] = {}

    for f in input_folders:
        name = str(f.split('/')[-1])
        # print("Reading", os.path.join(f, "evaluation.pickle"))
        s = read_evaluation_pickle(f)

        summaries[name] = s
        # print(args.input_folder)

    common_datasets = identify_common_datasets(list(summaries.values()))
    changing_parameters = identify_changing_parameters(list(summaries.values()))

    names = [s.name for s in summaries.values()]

    # input_folders = [os.path.dirname(f) for f in input_folders]
    eklt_names = [s.name for s in summaries.values() if s.frontend == FrontEnd.EKLT]

    eklt_summaries = [s for s in summaries.values() if s.frontend == FrontEnd.EKLT]

    feature_tracking_summaries = []

    for s in summaries.values():
        has_feature_tracks = [hasattr(s.data[dataset].feature_data, 'xvio_tracks_error') and s.data[
            dataset].feature_data.xvio_tracks_error is not None for dataset in s.data.keys()]
        has_feature_tracks = np.all(has_feature_tracks)

        if has_feature_tracks:
            feature_tracking_summaries.append(s)

    print(F"Comparing {', '.join(summaries.keys())} on following datasets: {', '.join(common_datasets)}")
    print(F"Overall the following parameters change (excluding initial filter states): {changing_parameters}")
    print()
    print("[1/2] Creating excel result tables")

    # the mixed error table is also dropped as plot later
    table_mixed_errors = create_excel_result_tables(common_datasets, output_folder, summaries)

    manager = ProgressPlotContextManager()

    # dry run with dummy plot context
    compare(common_datasets, eklt_names, eklt_summaries, feature_tracking_summaries, names, output_folder, summaries,
            args.custom_rpe_distances, table_mixed_errors, manager.dummy_plot_context)

    print()
    print(F"[2/2] Creating {manager.count} comparison plots")
    print()

    manager.init_progress_bar()

    # actual run with displaying progress
    compare(common_datasets, eklt_names, eklt_summaries, feature_tracking_summaries, names, output_folder, summaries,
            args.custom_rpe_distances,table_mixed_errors, manager.actual_plot_context)


def create_excel_result_tables(common_datasets, output_folder, summaries):
    ############################################# FIRST DUMP RESULT TABLES #############################################
    table_mixed_errors = te.compare_trajectory_performance_wrt_traveled_dist(list(summaries.values()))
    table_completion_rate = te.compare_trajectory_completion_rates(list(summaries.values()))
    table_pos_errors = table_mixed_errors.xs('Mean Position Error [%]', axis=1, level=1, drop_level=True)
    completion_rate = table_completion_rate.xs('Completion rate [%]', axis=1, level=1, drop_level=True)
    table_rot_errors = table_mixed_errors.xs('Mean Rotation error [deg/m]', axis=1, level=1, drop_level=True)
    parameter_changes_table = create_parameter_changes_table(list(summaries.values()), common_datasets)
    # print(table_mixed_errors.to_latex())
    completion_rate = pd.DataFrame(completion_rate.min(axis=0), columns=["Worst completion rate [%]"]).T
    additional_top_rows = pd.concat((completion_rate, parameter_changes_table))
    table_pos_errors = pd.concat((additional_top_rows, table_pos_errors))
    table_rot_errors = pd.concat((additional_top_rows, table_rot_errors))
    table_pos_errors = pd.concat((completion_rate, table_pos_errors))
    with pd.ExcelWriter(os.path.join(output_folder, "result_tables.xlsx")) as writer:
        pos_sheet = 'Pos errors w.r.t. traveled dist'
        table_pos_errors.to_excel(writer, sheet_name=pos_sheet, startrow=1)
        writer.sheets[pos_sheet].cell(row=1, column=1).value = 'Translation error w.r.t. traveled distance [%]'
        rot_sheet = 'Rot errors w.r.t. traveled dist'
        table_rot_errors.to_excel(writer, sheet_name=rot_sheet, startrow=1)
        writer.sheets[rot_sheet].cell(row=1, column=1).value = 'Rotation error w.r.t. traveled distance [deg/m]'
        table_mixed_errors.to_excel(writer, sheet_name='Errors w.r.t. traveled dist')
        parameter_changes_table.to_excel(writer, sheet_name='Parameter changes')
        table_completion_rate.to_excel(writer, sheet_name='Completion rate')
    return table_mixed_errors


def compare(common_datasets, eklt_names, eklt_summaries, feature_tracking_summaries, names, output_folder, summaries,
            custom_rpe_distances, table_mixed_errors, PlotContext):

    ########################################### CREATE ALL COMPARISON PLOTS ############################################
    scaled_width_datasets = max(10 * len(common_datasets) / 6, 10)
    scaled_with_runs = max(10 * len(summaries) / 6, 10)
    #   - [x] Overview plot of ATEs
    with PlotContext(os.path.join(output_folder, F"compare_all_ate_wrt_traveled_dist"),
                     base_width_inch=scaled_with_runs * 1.5) as pc:
        te.plot_trajectory_comparison_overview(pc, table_mixed_errors)
    #   - [x] Overview plot of ATEs
    with PlotContext(os.path.join(output_folder, F"compare_all_ate_wrt_traveled_dist_log"),
                     base_width_inch=scaled_with_runs * 1.5) as pc:
        te.plot_trajectory_comparison_overview(pc, table_mixed_errors, use_log=True)
    #   - [x] Overall time
    with PlotContext(os.path.join(output_folder, F"compare_all_processing_times"),
                     base_width_inch=scaled_width_datasets) as pc:
        pe.plot_processing_times(pc, summaries, common_datasets)
    #   - [x] CPU usage
    with PlotContext(os.path.join(output_folder, F"compare_all_cpu_usage"),
                     base_width_inch=scaled_width_datasets) as pc:
        pe.plot_cpu_usage_boxplot_comparison(pc, summaries, common_datasets)
    #   - [x] Memory usage
    with PlotContext(os.path.join(output_folder, F"compare_all_memory_usage"),
                     base_width_inch=scaled_width_datasets) as pc:
        pe.plot_memory_usage_boxplot_comparison(pc, summaries, common_datasets)
    if len(feature_tracking_summaries) > 0:
        #   - [x] Pixel tracking accuracy
        with PlotContext(os.path.join(output_folder, F"compare_all_feature_tracking")) as pc:
            fe.plot_xvio_feature_tracking_comparison_boxplot(pc, feature_tracking_summaries, common_datasets)

        #   - [x] Feature update interval
        with PlotContext(os.path.join(output_folder, F"compare_all_feature_update_intervals")) as pc:
            fe.plot_xvio_feature_update_interval_comparison_boxplot(pc, feature_tracking_summaries, common_datasets)

        # #   - [x] Backend feature age boxplot
        # with PlotContext(os.path.join(output_folder, F"compare_all_backend_feature_age")) as pc:
        #     fe.plot_backend_feature_age_comparison_boxplot(pc, feature_tracking_summaries, common_datasets)
        #
        # #   - [x] Backend feature age boxplot
        # with PlotContext(os.path.join(output_folder, F"compare_all_backend_feature_age_log")) as pc:
        #     fe.plot_backend_feature_age_comparison_boxplot(pc, feature_tracking_summaries, common_datasets,
        #                                                    use_log=True)`
    if len(eklt_summaries) > 0:
        #   - [x] Optimization iterations
        with PlotContext(os.path.join(output_folder, F"compare_all_eklt_optimization_iterations")) as pc:
            pe.plot_optimization_iterations_comparison(pc, eklt_summaries, common_datasets)

        #   - [x] Feature age
        with PlotContext(os.path.join(output_folder, F"compare_all_eklt_feature_ages")) as pc:
            fe.plot_eklt_feature_age_comparison(pc, eklt_summaries, common_datasets)

        #   - [x] Feature update rate
        with PlotContext(os.path.join(output_folder, F"compare_all_eklt_feature_update_interval")) as pc:
            fe.plot_eklt_feature_update_interval_comparison_boxplot(pc, eklt_summaries, common_datasets)
    ########################################## PER-DATASET COMPARISON PLOTS ############################################
    for dataset in common_datasets:
        d_id = name_to_identifier(dataset)

        evaluations = [s.data[dataset] for s in summaries.values()]
        rows, cols = n_to_grid_size(len(evaluations))

        feature_tracking_evaluations = [s.data[dataset] for s in summaries.values()
                                        if hasattr(s.data[dataset].feature_data, 'xvio_tracks_error')
                                        and s.data[dataset].feature_data.xvio_tracks_error is not None]
        for s in summaries.values():
            has_feature_tracks = [hasattr(s.data[dataset].feature_data, 'xvio_tracks_error') and s.data[
                dataset].feature_data.xvio_tracks_error is not None]
            has_feature_tracks = np.all(has_feature_tracks)

            if has_feature_tracks:
                feature_tracking_summaries.append(s)
        eklt_evaluations = [s.data[dataset] for s in summaries.values() if s.frontend == FrontEnd.EKLT]
        eklt_rows, eklt_cols = n_to_grid_size(len(eklt_evaluations))

        trajectories_data = [s.data[dataset].trajectory_data for s in summaries.values()
                             if s.data[dataset].trajectory_data is not None]

        if len(eklt_evaluations) > 0:
            # EDIT: currently failing as DistributionSummary does not set bins_log, apparenty there are some 0s
            # procesisng times
            # #   - [x] Time / event
            # with PlotContext(os.path.join(output_folder, F"compare_eklt_event_processing_times_{d_id}"),
            #                  subplot_rows=len(eklt_evaluations), base_height_inch=3) as pc:
            #     pc.figure.suptitle(F"Event processing times on '{dataset}'")
            #     pe.plot_event_processing_times(pc, eklt_evaluations, eklt_names)

            #   - [x] Events / s
            with PlotContext(os.path.join(output_folder, F"compare_eklt_events_per_second_{d_id}")) as pc:
                pe.plot_events_per_seconds_comparison(pc, eklt_evaluations, eklt_names, dataset)

            #   - [x] Optimizations / s
            with PlotContext(os.path.join(output_folder, F"compare_eklt_optimizations_per_second_{d_id}")) as pc:
                pe.plot_optimizations_per_seconds_comparison(pc, eklt_evaluations, eklt_names, dataset)

            #   - [x] Number of tracked features
            with PlotContext(os.path.join(output_folder, F"compare_eklt_num_features_{d_id}")) as pc:
                fe.plot_eklt_num_features_comparison(pc, eklt_evaluations, eklt_names, dataset)

            #   - [x] Pixel change histograms
            with PlotContext(os.path.join(output_folder, F"compare_eklt_feature_pos_changes_{d_id}"),
                             subplot_rows=eklt_rows, subplot_cols=eklt_cols) as pc:
                fe.plot_eklt_all_feature_pos_changes(pc, eklt_evaluations, eklt_names)
            #
            # #   - [x] Pixel tracking accuracy
            # with PlotContext(os.path.join(args.output_folder, F"compare_eklt_feature_tracking_{d_id}")) as pc:
            #     fe.plot_eklt_feature_tracking_comparison(pc, eklt_evaluations, eklt_names, dataset)

        #   - [x] CPU usage
        with PlotContext(os.path.join(output_folder, F"compare_cpu_usage_in_time_{d_id}")) as pc:
            pe.plot_cpu_usage_in_time_comparison(pc, evaluations, names, dataset)

        #   - [x] Memory usage
        with PlotContext(os.path.join(output_folder, F"compare_memory_usage_in_time_{d_id}")) as pc:
            pe.plot_memory_usage_in_time_comparison(pc, evaluations, names, dataset)

        if len(feature_tracking_evaluations) > 0:
            #   - [x] Pixel tracking accuracy
            with PlotContext(os.path.join(output_folder, F"compare_backend_feature_tracking_{d_id}")) as pc:
                fe.plot_xvio_feature_tracking_comparison(pc, feature_tracking_evaluations, names, dataset)

            # #   - [x] Pixel tracking accuracy
            # with PlotContext(os.path.join(output_folder, F"compare_backend_feature_age_{d_id}")) as pc:
            #     fe.plot_backend_feature_age_comparison(pc, feature_tracking_evaluations, names, dataset)

            #   - [x] Pixel tracking accuracy
            with PlotContext(os.path.join(output_folder, F"compare_backend_feature_tracking_zero_aligned_{d_id}")) as \
                    pc:
                fe.plot_xvio_feature_tracking_zero_aligned_comparison(pc, feature_tracking_evaluations, names, dataset)

            #   - [x] Pixel change histograms
            with PlotContext(os.path.join(output_folder, F"compare_backend_feature_pos_changes_{d_id}"),
                             subplot_rows=rows, subplot_cols=cols) as pc:
                fe.plot_xvio_all_feature_pos_changes(pc, feature_tracking_evaluations, names)

            #   - [x] Pixel change histograms
            with PlotContext(os.path.join(output_folder, F"compare_backend_feature_optical_flows_{d_id}"),
                             subplot_rows=rows, subplot_cols=cols) as pc:
                fe.plot_xvio_all_feature_optical_flows(pc, feature_tracking_evaluations, names)

            #   - [x] Feature update interval
            with PlotContext(os.path.join(output_folder, F"compare_backend_feature_update_interval_{d_id}")) as pc:
                fe.plot_xvio_feature_update_interval_in_time(pc, feature_tracking_evaluations, names, dataset)

        #   - [x] Error in time
        with PlotContext(os.path.join(output_folder, F"compare_ape_in_time_{d_id}"), subplot_cols=2,
                         base_width_inch=scaled_with_runs) as pc:
            pc.figure.suptitle(F"APE comparison on '{dataset}'")
            te.plot_ape_error_comparison(pc, evaluations, PlotType.TIME_SERIES, names)

        #   - [x] Error in time
        with PlotContext(os.path.join(output_folder, F"compare_ape_boxplot_{d_id}"), subplot_cols=2,
                         base_width_inch=scaled_with_runs) as pc:
            pc.figure.suptitle(F"APE comparison on '{dataset}'")
            te.plot_ape_error_comparison(pc, evaluations, PlotType.BOXPLOT, names)

        #   - [x] Error in time
        with PlotContext(os.path.join(output_folder, F"compare_ape_log_in_time_{d_id}"), subplot_cols=2) as pc:
            pc.figure.suptitle(F"APE comparison on '{dataset}' in log scale")
            te.plot_ape_error_comparison(pc, evaluations, PlotType.TIME_SERIES, names, use_log=True)

        #   - [x] Error in time
        with PlotContext(os.path.join(output_folder, F"compare_ape_log_boxplot_{d_id}"), subplot_cols=2) as pc:
            pc.figure.suptitle(F"APE comparison on '{dataset}' in log scale")
            te.plot_ape_error_comparison(pc, evaluations, PlotType.BOXPLOT, names, use_log=True)

        #   - [x] Realtime factor
        with PlotContext(os.path.join(output_folder, F"compare_rt_factor_{d_id}")) as pc:
            pe.plot_realtime_factor(pc, evaluations, names, title=F"Realtime factor on '{dataset}'")

        #   - [x] Realtime factor
        with PlotContext(os.path.join(output_folder, F"compare_rt_factor_log_{d_id}")) as pc:
            pe.plot_realtime_factor(pc, evaluations, names, use_log=True,
                                    title=F"Realtime factor on '{dataset}' in log scale")

        # with PlotContext(os.path.join(args.output_folder, F"compare_rt_factor_{d_id}_subplot"), subplot_rows=rows,W
        #                  subplot_cols=cols) as pc:
        #     for k, s in summaries.items():
        #         pe.plot_realtime_factor(pc, [s.data[dataset]], [k])

        #   - [x] Boxplots
        with PlotContext(os.path.join(output_folder, F"compare_rpg_errors_{d_id}"), subplot_cols=2) as pc:
            pc.figure.suptitle(F"Relative pose errors for all pairs at different distances on '{dataset}'")
            te.plot_rpg_error_arrays(pc, trajectories_data, names)

        with PlotContext(os.path.join(output_folder, F"compare_rpg_errors_percent_{d_id}"), subplot_cols=2) as pc:
            pc.figure.suptitle(F"Relative pose errors for all pairs at different distances on '{dataset}'")
            te.plot_rpg_error_arrays(pc, trajectories_data, names, realtive_to_trav_dist=True)

        if custom_rpe_distances is not None:
            with PlotContext(os.path.join(output_folder, F"compare_rpg_errors_custom_{d_id}"), subplot_cols=2) as pc:
                pc.figure.suptitle(F"Relative pose errors for all pairs at different distances on '{dataset}'")
                te.plot_rpg_error_arrays(pc, trajectories_data, names, desired_distances=custom_rpe_distances)

            with PlotContext(os.path.join(output_folder, F"compare_rpg_errors_custom_percent_{d_id}"), subplot_cols=2)\
                    as pc:
                pc.figure.suptitle(F"Relative pose errors for all pairs at different distances on '{dataset}'")
                te.plot_rpg_error_arrays(pc, trajectories_data, names, realtive_to_trav_dist=True,
                                         desired_distances=custom_rpe_distances)

        #   - [x] Boxplots
        with PlotContext(os.path.join(output_folder, F"compare_rpg_errors_log_{d_id}"), subplot_cols=2) as pc:
            pc.figure.suptitle(F"Relative pose errors for all pairs at different distances on '{dataset}' in log scale")
            te.plot_rpg_error_arrays(pc, trajectories_data, names, use_log=True)

        with PlotContext(os.path.join(output_folder, F"compare_rpg_errors_percent_log_{d_id}"), subplot_cols=2) as \
                pc:
            pc.figure.suptitle(F"Relative pose errors for all pairs at different distances on '{dataset}' in log scale")
            te.plot_rpg_error_arrays(pc, trajectories_data, names, use_log=True, realtive_to_trav_dist=True)

        #   - [x] SLAM / MSCKF / Opp number of features
        with PlotContext(os.path.join(output_folder, F"compare_backend_num_features_{d_id}"),
                         subplot_cols=2, subplot_rows=2) as pc:
            fe.plot_xvio_num_features(pc, evaluations, names, title=F"Number of features in backend on '{dataset}'")

        #   - [x] SLAM / MSCKF / Opp number of features
        with PlotContext(os.path.join(output_folder, F"compare_backend_num_features_boxplot_{d_id}"),
                         base_width_inch=scaled_width_datasets * 0.6) as pc:
            fe.plot_num_features_boxplot_comparison(pc, evaluations, names,
                                                    title=F"Number of features in backend on '{dataset}'")

        has_imu_bias = [hasattr(s.data[dataset].trajectory_data, 'imu_bias') and s.data[
            dataset].trajectory_data.imu_bias is not None for s in summaries.values()]
        has_imu_bias = np.all(has_imu_bias)

        if has_imu_bias:
            with PlotContext(os.path.join(output_folder, F"compare_imu_bias_{d_id}"), subplot_rows=rows,
                             subplot_cols=cols) as pc:
                for k, s in summaries.items():
                    te.plot_imu_bias_in_one(pc, s.data[dataset].trajectory_data.imu_bias, s.data[dataset].name, s.name)


if __name__ == '__main__':
    main()
