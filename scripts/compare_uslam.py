import argparse
import os

import numpy as np
import pandas as pd
from typing import Dict

from x_evaluate.comparisons import identify_common_datasets, identify_changing_parameters, \
    create_parameter_changes_table
from x_evaluate.evaluation_data import EvaluationDataSummary, FrontEnd
import x_evaluate.plots
from x_evaluate.plots import PlotContext, PlotType
from x_evaluate.utils import name_to_identifier, n_to_grid_size
from x_evaluate.scriptlets import read_evaluation_pickle, find_evaluation_files_recursively

import x_evaluate.performance_evaluation as pe
import x_evaluate.trajectory_evaluation as te
import x_evaluate.tracking_evaluation as fe


def main():
    parser = argparse.ArgumentParser(description='Comparison script for USLAM evaluation.pickle files')
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--use_paper_style_plots', action='store_true', default=False)
    parser.add_argument('--custom_rpe_distances', nargs='+', default=None, type=float)

    args = parser.parse_args()

    x_evaluate.plots.use_paper_style_plots = args.use_paper_style_plots

    root_folder = args.input_folder
    input_folders = find_evaluation_files_recursively(root_folder)

    input_folders = [os.path.dirname(f) for f in input_folders]

    output_folder = args.output_folder

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

    names = [s.name for s in summaries.values()]

    print(F"Comparing {', '.join(summaries.keys())} on following datasets: {', '.join(common_datasets)}")

    ############################################# FIRST DUMP RESULT TABLES #############################################

    table_mixed_errors = te.compare_trajectory_performance_wrt_traveled_dist(list(summaries.values()))
    table_completion_rate = te.compare_trajectory_completion_rates(list(summaries.values()))
    table_pos_errors = table_mixed_errors.xs('Mean Position Error [%]', axis=1, level=1, drop_level=True)
    completion_rate = table_completion_rate.xs('Completion rate [%]', axis=1, level=1, drop_level=True)
    table_rot_errors = table_mixed_errors.xs('Mean Rotation error [deg/m]', axis=1, level=1, drop_level=True)

    # print(table_mixed_errors.to_latex())

    completion_rate = pd.DataFrame(completion_rate.min(axis=0), columns=["Worst completion rate [%]"]).T

    table_pos_errors = pd.concat((completion_rate, table_pos_errors))

    with pd.ExcelWriter(os.path.join(output_folder, "result_tables.xlsx")) as writer:
        pos_sheet = 'Pos errors w.r.t. traveled dist'
        table_pos_errors.to_excel(writer, sheet_name=pos_sheet, startrow=1)
        writer.sheets[pos_sheet].cell(row=1, column=1).value = 'Translation error w.r.t. traveled distance [%]'
        rot_sheet = 'Rot errors w.r.t. traveled dist'
        table_rot_errors.to_excel(writer, sheet_name=rot_sheet, startrow=1)
        writer.sheets[rot_sheet].cell(row=1, column=1).value = 'Rotation error w.r.t. traveled distance [deg/m]'
        table_mixed_errors.to_excel(writer, sheet_name='Errors w.r.t. traveled dist')
        table_completion_rate.to_excel(writer, sheet_name='Completion rate')

    ########################################### CREATE ALL COMPARISON PLOTS ############################################

    scaled_width_datasets = max(10 * len(common_datasets) / 6, 10)
    scaled_with_runs = max(10 * len(summaries) / 6, 10)

    #   - [x] Overview plot of ATEs
    with PlotContext(os.path.join(output_folder, F"compare_all_ate_wrt_traveled_dist"),
                     base_width_inch=scaled_with_runs*1.5) as pc:
        te.plot_trajectory_comparison_overview(pc, table_mixed_errors)

    #   - [x] Overview plot of ATEs
    with PlotContext(os.path.join(output_folder, F"compare_all_ate_wrt_traveled_dist_log"),
                     base_width_inch=scaled_with_runs*1.5) as pc:
        te.plot_trajectory_comparison_overview(pc, table_mixed_errors, use_log=True)

    for dataset in common_datasets:
        d_id = name_to_identifier(dataset)

        evaluations = [s.data[dataset] for s in summaries.values()]
        rows, cols = n_to_grid_size(len(evaluations))

        trajectories_data = [s.data[dataset].trajectory_data for s in summaries.values()
                             if s.data[dataset].trajectory_data is not None]

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

        if args.custom_rpe_distances is not None:
            with PlotContext(os.path.join(output_folder, F"compare_rpg_errors_custom_{d_id}"), subplot_cols=2) as pc:
                pc.figure.suptitle(F"Relative pose errors for all pairs at different distances on '{dataset}'")
                te.plot_rpg_error_arrays(pc, trajectories_data, names, desired_distances=args.custom_rpe_distances)

            with PlotContext(os.path.join(output_folder, F"compare_rpg_errors_custom_percent_{d_id}"), subplot_cols=2)\
                    as pc:
                pc.figure.suptitle(F"Relative pose errors for all pairs at different distances on '{dataset}'")
                te.plot_rpg_error_arrays(pc, trajectories_data, names, realtive_to_trav_dist=True,
                                         desired_distances=args.custom_rpe_distances)

        #   - [x] Boxplots
        with PlotContext(os.path.join(output_folder, F"compare_rpg_errors_{d_id}"), subplot_cols=2) as pc:
            pc.figure.suptitle(F"Relative pose errors for all pairs at different distances on '{dataset}'")
            te.plot_rpg_error_arrays(pc, trajectories_data, names)

        with PlotContext(os.path.join(output_folder, F"compare_rpg_errors_percent_{d_id}"), subplot_cols=2) as pc:
            pc.figure.suptitle(F"Relative pose errors for all pairs at different distances on '{dataset}'")
            te.plot_rpg_error_arrays(pc, trajectories_data, names, realtive_to_trav_dist=True)

        #   - [x] Boxplots
        with PlotContext(os.path.join(output_folder, F"compare_rpg_errors_log_{d_id}"), subplot_cols=2) as pc:
            pc.figure.suptitle(F"Relative pose errors for all pairs at different distances on '{dataset}' in log scale")
            te.plot_rpg_error_arrays(pc, trajectories_data, names, use_log=True)

        with PlotContext(os.path.join(output_folder, F"compare_rpg_errors_percent_log_{d_id}"), subplot_cols=2) as\
                pc:
            pc.figure.suptitle(F"Relative pose errors for all pairs at different distances on '{dataset}' in log scale")
            te.plot_rpg_error_arrays(pc, trajectories_data, names, use_log=True, realtive_to_trav_dist=True)


        # has_imu_bias = [hasattr(s.data[dataset].trajectory_data, 'imu_bias') and s.data[
        #     dataset].trajectory_data.imu_bias is not None for s in summaries.values()]
        # has_imu_bias = np.all(has_imu_bias)
        #
        # if has_imu_bias:
        #     with PlotContext(os.path.join(output_folder, F"compare_imu_bias_{d_id}"), subplot_rows=rows,
        #                      subplot_cols=cols) as pc:
        #         for k, s in summaries.items():
        #             te.plot_imu_bias_in_one(pc, s.data[dataset], s.name)


if __name__ == '__main__':
    main()
