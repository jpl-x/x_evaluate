import os
from typing import List

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from x_evaluate.evaluation_data import FeatureTrackingData, PerformanceData, EvaluationData, EvaluationDataSummary, \
    FrontEnd
from x_evaluate.utils import timestamp_to_rosbag_time_zero, convert_eklt_to_rpg_tracks
from x_evaluate.plots import boxplot, time_series_plot, PlotContext, boxplot_compare


def evaluate_feature_tracking(perf_data: PerformanceData, df_features: pd.DataFrame,
                              df_eklt_tracks: pd.DataFrame = None) -> FeatureTrackingData:
    d = FeatureTrackingData()

    feature_times = timestamp_to_rosbag_time_zero(df_features['ts'], perf_data.df_realtime)
    df_features['ts'] = feature_times

    d.df_x_vio_features = df_features.rename(columns={'ts': 't'})

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


def tracker_config_to_info_string(tracker_config):
    if tracker_config["type"] == "KLT":
        return "KLT"
    if tracker_config["type"] == "reprojection":
        return "depthmap reprojection"
    raise ValueError(F"Unknown tracking evaluation GT type {tracker_config['type']}")


def plot_tracking_error(pc: PlotContext, tracks_error, tracker_config, title):
    c = "blue"
    ax = pc.get_axis()
    buckets, stats = get_tracking_error_statistics(tracks_error)

    ax.plot(buckets, stats['median'], color=c)
    # ax.plot(t_mean, euclidean_errors, color=color, label=label.replace("_", " "), linewidth=2)

    ax.set_ylabel("Tracking error [px]")
    ax.set_xlabel("time [s]")

    ax.set_title(F"{title} w.r.t. {tracker_config_to_info_string(tracker_config)}")

    # 50% range
    ax.fill_between(buckets, stats['q25'], stats['q75'], alpha=0.5, lw=0, facecolor=c)

    # 90% range
    ax.fill_between(buckets, stats['q05'], stats['q25'], alpha=0.25, lw=0, facecolor=c)
    ax.fill_between(buckets, stats['q75'], stats['q95'], alpha=0.25, lw=0, facecolor=c)

    # MIN-MAX
    ax.fill_between(buckets, stats['min'], stats['q25'], alpha=0.1, lw=0, facecolor=c)
    ax.fill_between(buckets, stats['q75'], stats['max'], alpha=0.1, lw=0, facecolor=c)

    # right_y_axis = ax.twinx()
    # right_y_axis.plot(buckets, stats['num'])


def get_tracking_error_statistics(eklt_tracks_error):
    t = eklt_tracks_error[:, 1]
    t_uniq = np.unique(t)
    resolution = t_uniq[1] - t_uniq[0]
    resolution = 0.1
    buckets = np.arange(np.min(t), np.max(t), resolution)
    bucket_index = np.digitize(eklt_tracks_error[:, 1], buckets)
    indices = np.unique(bucket_index)

    # filter empty buckets:  (-1 to convert upper bound --> lower bound, as we always take the first errors per bucket)
    buckets = buckets[np.clip(indices-1, 0, len(buckets))]

    stats_func = {
        'mean': lambda x: np.mean(x),
        'median': lambda x: np.median(x),
        'min': lambda x: np.min(x),
        'max': lambda x: np.max(x),
        'q25': lambda x: np.quantile(x, 0.25),
        'q75': lambda x: np.quantile(x, 0.75),
        'q05': lambda x: np.quantile(x, 0.05),
        'q95': lambda x: np.quantile(x, 0.95),
        'num': lambda x: len(x),
    }

    stats = {x: np.empty((len(indices))) for x in stats_func.keys()}

    for i, idx in enumerate(indices):
        # tracking_errors = euclidean_error[bucket_index == idx]
        tracking_errors = eklt_tracks_error[bucket_index == idx]

        # only account for each track once per bucket
        plot_ids, first_ids = np.unique(tracking_errors[:, 0], return_index=True)
        tracking_errors = tracking_errors[first_ids, :]
        euclidean_error = np.linalg.norm(tracking_errors[:, 2:3], axis=1)

        for k, v in stats.items():
            stats[k][i] = stats_func[k](euclidean_error)

    return buckets, stats


def plot_feature_tracking_comparison(pc: PlotContext, eval_data: List[EvaluationData], labels, title,
                                     feature_data_to_error_and_config):
    means = []
    maxima = []
    times = []

    tracker_info = None

    for e in eval_data:
        tracks_error, config = feature_data_to_error_and_config(e.feature_data)
        time, stats = get_tracking_error_statistics(tracks_error)
        means.append(stats['mean'])
        maxima.append(stats['max'])
        times.append(time)
        info_string = tracker_config_to_info_string(config)
        if tracker_info is None:
            tracker_info = info_string
        else:
            assert tracker_info == info_string, F"Expecting same feature tracking evaluation settings on common " \
                                                F"dataset: '{tracker_info}' != '{tracker_info}"

    time_series_plot(pc, times, means, labels, F"{title}  w.r.t. {tracker_info}", "error [px]")


def plot_eklt_feature_tracking_comparison(pc: PlotContext, eval_data: List[EvaluationData], labels, dataset_name):
    plot_feature_tracking_comparison(pc, eval_data, labels, F"EKLT feature tracking error on '{dataset_name}'",
                                     lambda x: (x.eklt_tracks_error, x.eklt_tracking_evaluation_config))


def plot_xvio_feature_tracking_comparison(pc: PlotContext, eval_data: List[EvaluationData], labels, dataset_name):
    plot_feature_tracking_comparison(pc, eval_data, labels, F"SLAM and MSCKF feature tracking error on "
                                                            F"'{dataset_name}'",
                                     lambda x: (x.xvio_tracks_error, x.xvio_tracking_evaluation_config))


def plot_feature_tracking_comparison_boxplot(pc: PlotContext, summaries: List[EvaluationDataSummary],
                                             common_datasets, title, feature_data_to_error, feature_data_to_config):
    data = [[np.linalg.norm(feature_data_to_error(s.data[k].feature_data)[:, 2:3], axis=1)
             for k in common_datasets] for s in summaries]

    summary_labels = [s.name for s in summaries]
    info_strings = [tracker_config_to_info_string(feature_data_to_config(summaries[0].data[k].feature_data))
                    for k in common_datasets]
    dataset_labels = [F"{d} w.r.t. {info_strings[i]}" for i, d in enumerate(common_datasets)]
    boxplot_compare(pc.get_axis(), dataset_labels, data, summary_labels, ylabel="error [px]",
                    title=title)


def plot_eklt_feature_tracking_comparison_boxplot(pc: PlotContext, summaries: List[EvaluationDataSummary], common_datasets):
    plot_feature_tracking_comparison_boxplot(pc, summaries, common_datasets, "EKLT feature tracking error",
                                             lambda x: x.eklt_tracks_error, lambda x: x.eklt_tracking_evaluation_config)


def plot_xvio_feature_tracking_comparison_boxplot(pc: PlotContext, summaries: List[EvaluationDataSummary],
                                                  common_datasets):
    plot_feature_tracking_comparison_boxplot(pc, summaries, common_datasets, "SLAM and MSCKF feature tracking error",
                                             lambda x: x.xvio_tracks_error, lambda x: x.xvio_tracking_evaluation_config)


def plot_feature_plots(d: EvaluationData, output_folder):
    with PlotContext(os.path.join(output_folder, "xvio_num_features.svg"), subplot_rows=2, subplot_cols=2) as pc:
        plot_xvio_num_features(pc, d)

    with PlotContext(os.path.join(output_folder, "tracking_error.svg")) as pc:
        plot_tracking_error(pc, d.feature_data.xvio_tracks_error, d.feature_data.xvio_tracking_evaluation_config,
                            "SLAM and MSCKF feature tracking errors")

    if d.feature_data.df_eklt_num_features is not None:
        df = d.feature_data.df_eklt_num_features

        with PlotContext(os.path.join(output_folder, "eklt_features.svg")) as pc:
            time_series_plot(pc, df['t'], [df['num_features']], ["EKLT features"], "Number of tracked features")

    if d.feature_data.eklt_tracks_error is not None:
        with PlotContext(os.path.join(output_folder, "eklt_tracking_error.svg")) as pc:
            plot_tracking_error(pc, d.feature_data.eklt_tracks_error, d.feature_data.eklt_tracking_evaluation_config,
                                "EKLT feature tracking errors")


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

    # EKLT plots only for now
    if summary.frontend != FrontEnd.EKLT:
        return

    with PlotContext(os.path.join(output_folder, "eklt_feature_ages.svg")) as pc:
        plot_eklt_feature_age(pc, summary)


def plot_eklt_feature_age(pc, summary):
    data = [d.feature_data.df_eklt_feature_age['age'] for d in summary.data.values()]
    auto_labels = [d.name for d in summary.data.values()]
    boxplot(pc, data, auto_labels, "Feature age comparison", [1, 99])
