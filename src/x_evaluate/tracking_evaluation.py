import os
from typing import List

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from x_evaluate.evaluation_data import FeatureTrackingData, PerformanceData, EvaluationData, EvaluationDataSummary
from x_evaluate.utils import timestamp_to_rosbag_time_zero
from x_evaluate.plots import boxplot, time_series_plot, PlotContext, boxplot_compare


def evaluate_feature_tracking(perf_data: PerformanceData, df_features: pd.DataFrame,
                              df_eklt_tracks: pd.DataFrame = None) -> FeatureTrackingData:
    d = FeatureTrackingData()

    feature_times = timestamp_to_rosbag_time_zero(df_features['ts'], perf_data.df_realtime)
    df_features['ts'] = feature_times

    d.df_x_vio_features = df_features.rename(columns={'ts': 't'})

    if df_eklt_tracks is not None:
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
        d.df_eklt_features = df_eklt_tracks.loc[df_eklt_tracks.change != 0][['t', 'num_features']]

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
    with PlotContext(os.path.join(output_folder, "xvio_features.svg"), subplot_rows=2, subplot_cols=2) as pc:
        plot_xvio_num_features(pc, d)

    if d.feature_data.df_eklt_features is not None:
        df = d.feature_data.df_eklt_features

        with PlotContext(os.path.join(output_folder, "eklt_features.svg")) as pc:
            time_series_plot(pc, df['t'], [df['num_features']], ["EKLT features"], "Number of tracked features")


def plot_eklt_num_features_comparison(pc: PlotContext, eval_data: List[EvaluationData], labels, dataset_name):
    data = []
    times = []

    for e in eval_data:
        data.append(e.feature_data.df_eklt_features['num_features'])
        times.append(e.feature_data.df_eklt_features['t'])

    time_series_plot(pc, times, data, labels, F"Number of tracked EKLT features on '{dataset_name}')",
                     "number of features")


def plot_eklt_feature_age_comparison(pc: PlotContext, summaries: List[EvaluationDataSummary], common_datasets):
    data = [[s.data[k].feature_data.df_eklt_feature_age['age']
             for k in common_datasets] for s in summaries]

    summary_labels = [s.name for s in summaries]
    dataset_labels = common_datasets
    boxplot_compare(pc.get_axis(), dataset_labels, data, summary_labels, ylabel="feature age [s]",
                    title="EKLT feature age")


def plot_xvio_num_features(pc: PlotContext, d: EvaluationData):
    ax0 = pc.get_axis()
    ax1 = pc.get_axis()
    ax2 = pc.get_axis()
    ax3 = pc.get_axis()
    t = d.feature_data.df_x_vio_features['t']
    pc.figure.suptitle("Number of tracked features")
    # plt.title("Number of tracked features")
    ax0.set_title("SLAM")
    ax0.plot(t, d.feature_data.df_x_vio_features['num_slam_features'], color="blue")
    ax1.set_title("MSCKF")
    ax1.plot(t, d.feature_data.df_x_vio_features['num_msckf_features'], color="red")
    ax2.set_title("Opportunistic")
    ax2.plot(t, d.feature_data.df_x_vio_features['num_opportunistic_features'], color="green")
    ax3.set_title("Potential")
    ax3.plot(t, d.feature_data.df_x_vio_features['num_potential_features'], color="orange")


def plot_summary_plots(summary: EvaluationDataSummary, output_folder):

    data = []
    auto_labels = []

    for e in summary.data.values():
        if e.feature_data.df_eklt_feature_age is not None:
            data.append(e.feature_data.df_eklt_feature_age['age'])
            auto_labels.append(e.name)

    if len(data) <= 0:
        return
    with PlotContext(os.path.join(output_folder, "eklt_feature_ages.svg")) as pc:
        boxplot(pc, data, auto_labels, "Feature age comparison", [1, 99])
