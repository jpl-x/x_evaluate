import argparse
import os.path
from typing import Dict

import numpy as np
import x_evaluate.plots
from matplotlib import pyplot as plt

from x_evaluate.comparisons import identify_common_datasets
from x_evaluate.evaluation_data import FrontEnd, EvaluationDataSummary
from x_evaluate.performance_evaluation import plot_realtime_factor, plot_events_per_second, RT_FACTOR_RESOLUTION
from x_evaluate.plots import PlotContext, plot_evo_trajectory_with_euler_angles, time_series_plot
from x_evaluate.rpg_tracking_analysis.plot_tracks import plot_num_features
from x_evaluate.scriptlets import find_evaluation_files_recursively, read_evaluation_pickle
from x_evaluate.tracking_evaluation import plot_xvio_num_features, plot_eklt_num_features_comparison
from x_evaluate.utils import name_to_identifier


def main():
    parser = argparse.ArgumentParser(description='RT factor vs event rate')
    parser.add_argument('--input_folder', type=str, required=True, help='Input folder containing evaluation.pickle '
                                                                        'files in subdirs, with ideally EKLT, '
                                                                        'HASTE and XVIO frontend performance runs')

    args = parser.parse_args()

    output_folder = os.path.join(args.input_folder, "results")

    eval_files = find_evaluation_files_recursively(args.input_folder)



    evaluations = [read_evaluation_pickle(os.path.dirname(f)) for f in eval_files]

    summaries: Dict[FrontEnd, EvaluationDataSummary] = {e.frontend: e for e in evaluations}

    for k in summaries.keys():
        if len([e for e in evaluations if e.frontend == k]) > 1:
            print(F"WARNING: multiple evaluation files for frontend '{k}' found, using {summaries[k].name}")

    frontends = list(summaries.keys())
    frontend_labels = [str(f) for f in frontends]
    summary_list = [summaries[f] for f in frontends]
    common_datasets = list(identify_common_datasets(summary_list))
    common_datasets.sort()

    x_evaluate.plots.use_paper_style_plots = True

    rtf_per_event_dict = dict()
    event_rate_dict = dict()

    rtf_per_event_overall = np.array([])
    event_rate_overall = np.array([])

    def filename_overall(plot_name):
        return os.path.join(output_folder, F"rtf_overall_{plot_name}")

    for dataset in common_datasets:
        print(F"Plotting for sequence {dataset}")

        def filename_dataset(plot_name):
            return os.path.join(output_folder, F"rtf_{plot_name}_{name_to_identifier(dataset)}")

        eklt_eval = summaries[FrontEnd.EKLT].data[dataset]
        evaluations = [s.data[dataset] for s in summary_list]
        t_er = np.arange(0.0, len(eklt_eval.eklt_performance_data.events_per_sec_sim))

        # for combined plots...
        t_rtf = np.arange(0.0, len(eklt_eval.performance_data.rt_factors)) * RT_FACTOR_RESOLUTION
        t_nf = eklt_eval.feature_data.df_eklt_num_features['t']
        nf = np.interp(t_rtf, t_nf, eklt_eval.feature_data.df_eklt_num_features['num_features'])
        nf_min_1 = np.max(np.vstack((nf, np.ones_like(nf))), axis=0)
        rt_factor = eklt_eval.performance_data.rt_factors
        rt_factor_per_feature = rt_factor / nf_min_1
        event_rate = np.interp(t_rtf, t_er, eklt_eval.eklt_performance_data.events_per_sec_sim)

        k = 5
        event_rate_smooth = np.convolve(event_rate, np.ones(k), 'same') / k
        rtf_per_feature_smooth = np.convolve(rt_factor_per_feature, np.ones(k), 'same') / k

        rtf_per_event_overall = np.hstack((rtf_per_event_overall, rtf_per_feature_smooth))
        event_rate_overall = np.hstack((event_rate_overall, event_rate_smooth))

        rtf_per_event_dict[dataset] = rtf_per_feature_smooth
        event_rate_dict[dataset] = event_rate_smooth

        with PlotContext(filename_dataset("overview"), subplot_cols=3, subplot_rows=2) as pc:
            plot_realtime_factor(pc, evaluations, frontend_labels)
            plot_eklt_num_features_comparison(pc, [eklt_eval], ["EKLT"], dataset)
            time_series_plot(pc, t_er, [eklt_eval.eklt_performance_data.events_per_sec_sim], ["events/s"])

            time_series_plot(pc, t_rtf, [rt_factor_per_feature], ["RTF / feature"])

            time_series_plot(pc, [event_rate], [rt_factor_per_feature], ["event rate - RTF/F"],
                             xlabel="Event rate", ylabel="RTF per feature", use_scatter=True)

            time_series_plot(pc, [event_rate_smooth], [rtf_per_feature_smooth], ["event rate - RTF/F"],
                             xlabel="Event rate", ylabel="RTF per feature", use_scatter=True, title="Same but smooth")

            # smooth_rt_factor = np.convolve(eklt_eval.performance_data.rt_factors, np.ones(10), 'same') / 10
            # time_series_plot(pc, t_rtf, [smooth_rt_factor], ["Smooth RT factor EKLT"])

        with PlotContext(filename_dataset("rtfpf_vs_event_rate")) as pc:
            plot_rtfpf_vs_event_rate(pc.get_axis(), event_rate_smooth, rtf_per_feature_smooth)

    with PlotContext(filename_overall("rtfpf_vs_event_rate")) as pc:
        ax = pc.get_axis()
        plot_rtfpf_vs_event_rate(ax, event_rate_overall, rtf_per_event_overall)

    # plt.show()


def plot_rtfpf_vs_event_rate(ax, event_rate, rtf_per_feature):
    ax.scatter(event_rate/1e6, rtf_per_feature, alpha=0.5)
    ax.set_xlabel("Event rate [Me/s]")
    ax.set_ylabel("Realtime factor / feature")

    # ax.scatter(times[i], d, s=size, label=labels[i], alpha=0.5, color=DEFAULT_COLORS[i])
    # time_series_plot(pc, [event_rate_smooth], [rtf_per_feature_smooth], ["event rate - RTF/F"],
    #                  xlabel="Event rate", ylabel="RTF per feature", use_scatter=True, title="Same but smooth")


if __name__ == '__main__':
    main()
