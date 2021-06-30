import copy
import os
from typing import List

import matplotlib.colors
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from x_evaluate.evaluation_data import FeatureTrackingData, PerformanceData, EvaluationData, EvaluationDataSummary, \
    FrontEnd
from x_evaluate.utils import timestamp_to_rosbag_time_zero, convert_eklt_to_rpg_tracks, convert_xvio_to_rpg_tracks, \
    get_quantized_statistics_along_axis
from x_evaluate.plots import boxplot, time_series_plot, PlotContext, boxplot_compare, hist_from_bin_values, \
    bubble_plot, plot_moving_boxplot_in_time_from_stats


def evaluate_feature_tracking(perf_data: PerformanceData, df_features: pd.DataFrame,
                              df_eklt_tracks: pd.DataFrame = None) -> FeatureTrackingData:
    d = FeatureTrackingData()

    feature_times = timestamp_to_rosbag_time_zero(df_features['ts'], perf_data.df_realtime)
    df_features['ts'] = feature_times

    d.df_xvio_num_features = df_features.rename(columns={'ts': 't'})

    if df_eklt_tracks is not None:
        d.df_eklt_tracks = df_eklt_tracks.copy()
        track_times = timestamp_to_rosbag_time_zero(df_eklt_tracks['ts'], perf_data.df_realtime)
        df_eklt_tracks['ts'] = track_times
        df_eklt_tracks = df_eklt_tracks.rename(columns={'ts': 't'})
        df_eklt_tracks['change'] = 0
        # df_eklt_tracks.loc[df_eklt_tracks.update_type == "Bootstrap", 'change'] = 1
        # df_eklt_tracks.loc[df_eklt_tracks.update_type == "Lost", 'change'] = -1
        # df_eklt_tracks['number_of_features'] = df_eklt_tracks['change'].cumsum()
        #
        # useful_tracks = df_eklt_tracks.groupby('id').nth(2)

        useful_tracks = df_eklt_tracks.groupby('id', as_index=False).nth(2)
        useful_tracks = useful_tracks.loc[useful_tracks.update_type == 'Update']
        df_eklt_tracks.loc[useful_tracks.index, 'change'] = 1
        eol_mask = (df_eklt_tracks.update_type == "Lost") & (df_eklt_tracks.id.isin(useful_tracks.id))
        df_eklt_tracks.loc[eol_mask, 'change'] = -1

        df_eklt_tracks['num_features'] = df_eklt_tracks['change'].cumsum()
        d.df_eklt_num_features = df_eklt_tracks.loc[df_eklt_tracks.change != 0][['t', 'num_features']]

        starting_times = df_eklt_tracks.loc[useful_tracks.index].sort_values('id')
        eol_times = df_eklt_tracks.loc[eol_mask].sort_values('id')

        i1 = pd.Index(starting_times.id)
        i2 = pd.Index(eol_times.id)

        idx = i1.intersection(i2)
        starting_times = starting_times.loc[df_eklt_tracks.id.isin(idx)]
        eol_times = eol_times.loc[df_eklt_tracks.id.isin(idx)]

        df_feature_age = starting_times[['t', 'id']].copy()
        feature_age = eol_times['t'].to_numpy() - starting_times['t'].to_numpy()
        df_feature_age['age'] = feature_age

        d.df_eklt_feature_age = df_feature_age

        # df_tracks_count = df_tracks[df_tracks.change != 0][['ts', 'change']]
        # number_of_features_tracked = df_tracks_count['change'].cumsum()
    return d


def plot_feature_plots(d: EvaluationData, output_folder):
    with PlotContext(os.path.join(output_folder, "backend_num_features"), subplot_rows=2, subplot_cols=2) as pc:
        plot_xvio_num_features(pc, [d])

    if d.feature_data.df_eklt_num_features is not None:
        df = d.feature_data.df_eklt_num_features

        with PlotContext(os.path.join(output_folder, "eklt_features")) as pc:
            time_series_plot(pc, df['t'], [df['num_features']], ["EKLT features"], "Number of tracked features")


def plot_eklt_num_features_comparison(pc: PlotContext, eval_data: List[EvaluationData], labels, dataset_name):
    data = []
    times = []

    for e in eval_data:
        data.append(e.feature_data.df_eklt_num_features['num_features'])
        times.append(e.feature_data.df_eklt_num_features['t'])

    time_series_plot(pc, times, data, labels, F"Number of tracked EKLT features on '{dataset_name}'",
                     "number of features")


def plot_eklt_feature_age_comparison(pc: PlotContext, summaries: List[EvaluationDataSummary], common_datasets):
    data = [[s.data[k].feature_data.df_eklt_feature_age['age']
             for k in common_datasets] for s in summaries]

    summary_labels = [s.name for s in summaries]
    dataset_labels = common_datasets
    boxplot_compare(pc.get_axis(), dataset_labels, data, summary_labels, ylabel="feature age [s]",
                    title="EKLT feature age")


def plot_xvio_num_features(pc: PlotContext, evaluations: List[EvaluationData], labels=None, title=None):
    ax0 = pc.get_axis()
    ax1 = pc.get_axis()
    ax2 = pc.get_axis()
    ax3 = pc.get_axis()
    if title:
        pc.figure.suptitle(title)
    else:
        pc.figure.suptitle("Number of tracked features")
    # plt.title("Number of tracked features")
    ax0.set_title("SLAM")
    ax1.set_title("MSCKF")
    ax2.set_title("Opportunistic")
    ax3.set_title("Potential")

    for i, d in enumerate(evaluations):
        t = d.feature_data.df_xvio_num_features['t']
        if len(evaluations) == 1:
            ax0.plot(t, d.feature_data.df_xvio_num_features['num_slam_features'], color="blue")
            ax1.plot(t, d.feature_data.df_xvio_num_features['num_msckf_features'], color="red")
            ax2.plot(t, d.feature_data.df_xvio_num_features['num_opportunistic_features'], color="green")
            ax3.plot(t, d.feature_data.df_xvio_num_features['num_potential_features'], color="orange")
        else:
            label = d.name
            if labels:
                label = labels[i]
            ax0.plot(t, d.feature_data.df_xvio_num_features['num_slam_features'], label=label)
            ax1.plot(t, d.feature_data.df_xvio_num_features['num_msckf_features'], label=label)
            ax2.plot(t, d.feature_data.df_xvio_num_features['num_opportunistic_features'], label=label)
            ax3.plot(t, d.feature_data.df_xvio_num_features['num_potential_features'], label=label)

    if len(evaluations) > 1:
        ax0.legend()
        ax1.legend()
        ax2.legend()
        ax3.legend()


def plot_num_features_boxplot_comparison(pc: PlotContext, evaluations: List[EvaluationData], labels, title):
    data = [[d.feature_data.df_xvio_num_features['num_slam_features'],
             d.feature_data.df_xvio_num_features['num_msckf_features'],
             d.feature_data.df_xvio_num_features['num_opportunistic_features'],
             d.feature_data.df_xvio_num_features['num_potential_features']] for d in evaluations]

    x_tick_labels = ["SLAM", "MSCKF", "Opportunistic", "Potential"]
    boxplot_compare(pc.get_axis(), x_tick_labels, data, labels, ylabel="Number of features", title=title)


def plot_summary_plots(summary: EvaluationDataSummary, output_folder):
    # EKLT plots only for now
    if summary.frontend != FrontEnd.EKLT:
        return

    with PlotContext(os.path.join(output_folder, "eklt_feature_ages")) as pc:
        plot_eklt_feature_age(pc, summary)


def plot_eklt_feature_age(pc, summary):
    data = [d.feature_data.df_eklt_feature_age['age'] for d in summary.data.values()]
    auto_labels = [d.name for d in summary.data.values()]
    boxplot(pc, data, auto_labels, "Feature age comparison", [1, 99])
