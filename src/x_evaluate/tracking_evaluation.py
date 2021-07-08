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


def tracker_config_to_info_string(tracker_config):
    if tracker_config["type"] == "KLT":
        return "KLT"
    if tracker_config["type"] == "reprojection":
        return "depthmap reprojection"
    raise ValueError(F"Unknown tracking evaluation GT type {tracker_config['type']}")


def plot_tracking_error(pc: PlotContext, tracks_error, tracker_config, title):
    t, stats = get_tracking_error_statistics(tracks_error)
    title = F"{title} w.r.t. {tracker_config_to_info_string(tracker_config)}"
    plot_moving_boxplot_in_time_from_stats(pc, t, stats, title, ylabel="Feature tracking error [px]")


def get_tracking_error_statistics(tracks_error):
    t = tracks_error[:, 1]

    def calculate_euclidian_error_for_unique_ids(sliced_errors):
        plot_ids, first_ids = np.unique(sliced_errors[:, 0], return_index=True)
        sliced_errors = sliced_errors[first_ids, :]
        euclidean_error = np.linalg.norm(sliced_errors[:, 2:3], axis=1)
        return euclidean_error

    t_quantized, statistics = get_quantized_statistics_along_axis(t, tracks_error,
                                                                  data_filter=calculate_euclidian_error_for_unique_ids)

    return t_quantized, statistics


def plot_feature_tracking_comparison(pc: PlotContext, eval_data: List[EvaluationData], labels, title,
                                     feature_data_to_track_info, pre_tracks_error_hook=None, xlabel=None):
    means = []
    maxima = []
    times = []

    tracker_info = None

    for e in eval_data:
        tracks_error, tracks, config = feature_data_to_track_info(e.feature_data)
        if pre_tracks_error_hook:
            tracks_error = pre_tracks_error_hook(tracks_error, tracks)
        time, stats = get_tracking_error_statistics(tracks_error)
        means.append(stats['mean'])
        maxima.append(stats['max'])
        times.append(time - time[0])
        info_string = tracker_config_to_info_string(config)
        if tracker_info is None:
            tracker_info = info_string
        else:
            assert tracker_info == info_string, F"Expecting same feature tracking evaluation settings on common " \
                                                F"dataset: '{tracker_info}' != '{tracker_info}"

    time_series_plot(pc, times, means, labels, F"{title}  w.r.t. {tracker_info}", "error [px]", xlabel=xlabel)


def plot_backend_feature_age_comparison_boxplot(pc: PlotContext, summaries: List[EvaluationDataSummary],
                                                common_datasets, title=None, use_log=False):
    def get_only_feature_age(feature_data: FeatureTrackingData):
        ages, starting_times = get_feature_ages(convert_xvio_to_rpg_tracks(feature_data.df_xvio_tracks))
        return ages

    data = [[get_only_feature_age(s.data[k].feature_data) for k in common_datasets] for s in summaries]
    summary_labels = [s.name for s in summaries]

    if not title:
        title = "Overall SLAM and MSCKF feature ages comparison"

    ax = pc.get_axis()
    boxplot_compare(ax, common_datasets, data, summary_labels, ylabel="feature age [s]", title=title)

    if use_log:
        ax.set_yscale('log')



def plot_backend_feature_age_comparison(pc: PlotContext, eval_data: List[EvaluationData], labels, dataset_name):
    times = []
    data = []

    for e in eval_data:
        age, start_time = get_feature_ages(convert_xvio_to_rpg_tracks(e.feature_data.df_xvio_tracks))

        # coarse resolution, since for each SLAM / MSCKF feature track there is only a single age (little data)
        time, stats = get_quantized_statistics_along_axis(start_time, age, resolution=5)

        times.append(time)
        data.append(stats['median'])

    time_series_plot(pc, times, data, labels, F"SLAM and MSCKF average feature age throughout sequence {dataset_name}",
                     "Average feature age [s]")


def eklt_feature_to_track_info(f: FeatureTrackingData):
    return f.eklt_tracks_error, convert_eklt_to_rpg_tracks(f.df_eklt_tracks), f.eklt_tracking_evaluation_config


def xvio_feature_to_track_info(f: FeatureTrackingData):
    return f.xvio_tracks_error, convert_xvio_to_rpg_tracks(f.df_xvio_tracks), f.xvio_tracking_evaluation_config


def plot_eklt_feature_tracking_comparison(pc: PlotContext, eval_data: List[EvaluationData], labels, dataset_name):
    plot_feature_tracking_comparison(pc, eval_data, labels, F"Average EKLT feature tracking error on '"
                                                            F"{dataset_name}'", eklt_feature_to_track_info)


def plot_xvio_feature_tracking_comparison(pc: PlotContext, eval_data: List[EvaluationData], labels, dataset_name):
    plot_feature_tracking_comparison(pc, eval_data, labels, F"Average SLAM and MSCKF feature tracking error on "
                                                            F"'{dataset_name}'", xvio_feature_to_track_info)


def plot_xvio_feature_tracking_zero_aligned_comparison(pc: PlotContext, eval_data: List[EvaluationData], labels, dataset_name):

    def zero_align_errors(errors, tracks):
        # traveled_distance = get_traveled_distance_from_features(tracks)
        # errors[:, 1] = traveled_distance
        return get_zero_aligned_feature_errors(errors)

    plot_feature_tracking_comparison(pc, eval_data, labels,
                                     F"Average SLAM and MSCKF feature tracking error on '{dataset_name}'",
                                     xvio_feature_to_track_info, zero_align_errors,
                                     xlabel="Relative time since feature initialization [s]")


def plot_xvio_feature_update_interval_in_time(pc: PlotContext, eval_data: List[EvaluationData], names, dataset):
    data = []

    for e in eval_data:
        tu = get_feature_update_times(convert_xvio_to_rpg_tracks(e.feature_data.df_xvio_tracks))
        tu[:, 1] *= 1e3  # in ms
        tu[:, 0] -= tu[0, 0]
        data.append(tu)

    bubble_plot(pc, data, names, y_resolution=1, x_resolution=0.1, title=F"Update intervals of SLAM and MSCKF "
                                                                         F"features on '{dataset}'",
                ylabel="interval [ms]", xlabel="time [s]")

    # Old scatter-plot (produces big files!):
    # data = []
    # times = []
    #
    # for e in eval_data:
    #     tu = get_feature_update_times(convert_xvio_to_rpg_tracks(e.feature_data.df_xvio_tracks))
    #     data.append(tu[:, 1] * 1e3)
    #     times.append(tu[:, 0])
    #
    # time_series_plot(pc, times, data, names, F"Update intervals of SLAM and MSCKF features on '{dataset}'",
    #                  "interval [ms]", use_scatter=True)


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


def plot_feature_update_interval_comparison_boxplot(pc: PlotContext, summaries: List[EvaluationDataSummary],
                                                    common_datasets, feature_data_to_tracks, title):
    # def tracks_to_update_rate(tracks):
    #     time_changes = get_feature_update_times(tracks)
    #     time_changes = time_changes[time_changes > 0]
    #     updates_per_sec = 1 / time_changes
    #     return updates_per_sec

    data = [[get_feature_update_times(feature_data_to_tracks(s.data[k].feature_data))[:, 1] * 1e3  # in [ms]
             for k in common_datasets] for s in summaries]

    summary_labels = [s.name for s in summaries]
    boxplot_compare(pc.get_axis(), common_datasets, data, summary_labels, ylabel="Interval [ms]", title=title)


def plot_eklt_feature_update_interval_comparison_boxplot(pc: PlotContext, summaries: List[EvaluationDataSummary],
                                                         common_datasets):
    feature_data_to_tracks = lambda f: convert_eklt_to_rpg_tracks(f.df_eklt_tracks)
    plot_feature_update_interval_comparison_boxplot(pc, summaries, common_datasets, feature_data_to_tracks, "EKLT feature update rates")


def plot_xvio_feature_update_interval_comparison_boxplot(pc: PlotContext, summaries: List[EvaluationDataSummary],
                                                         common_datasets):
    feature_data_to_tracks = lambda f: convert_xvio_to_rpg_tracks(f.df_xvio_tracks)
    plot_feature_update_interval_comparison_boxplot(pc, summaries, common_datasets, feature_data_to_tracks,
                                                    "SLAM and MSCKF feature update rates")


def plot_feature_position_change_comparison_boxplot(pc: PlotContext, summaries: List[EvaluationDataSummary],
                                                    common_datasets, feature_data_to_tracks, title):
    data = [[np.linalg.norm(get_feature_changes_xy(feature_data_to_tracks(s.data[k].feature_data)), axis=1)
             for k in common_datasets] for s in summaries]

    summary_labels = [s.name for s in summaries]
    boxplot_compare(pc.get_axis(), common_datasets, data, summary_labels, ylabel="Euclidean change / update [px]",
                    title=title)


def plot_eklt_feature_position_change_comparison_boxplot(pc: PlotContext, summaries: List[EvaluationDataSummary],
                                                     common_datasets):
    feature_data_to_tracks = lambda f: convert_eklt_to_rpg_tracks(f.df_eklt_tracks)
    plot_feature_position_change_comparison_boxplot(pc, summaries, common_datasets, feature_data_to_tracks,
                                                    "EKLT feature pixel changes")


def plot_xvio_feature_position_change_comparison_boxplot(pc: PlotContext, summaries: List[EvaluationDataSummary],
                                                     common_datasets):
    feature_data_to_tracks = lambda f: convert_xvio_to_rpg_tracks(f.df_xvio_tracks)
    plot_feature_position_change_comparison_boxplot(pc, summaries, common_datasets, feature_data_to_tracks,
                                                    "SLAM and MSCKF feature update rates")


def plot_eklt_feature_tracking_comparison_boxplot(pc: PlotContext, summaries: List[EvaluationDataSummary], common_datasets):
    plot_feature_tracking_comparison_boxplot(pc, summaries, common_datasets, "EKLT feature tracking error",
                                             lambda x: x.eklt_tracks_error, lambda x: x.eklt_tracking_evaluation_config)


def plot_xvio_feature_tracking_comparison_boxplot(pc: PlotContext, summaries: List[EvaluationDataSummary],
                                                  common_datasets):
    plot_feature_tracking_comparison_boxplot(pc, summaries, common_datasets, "SLAM and MSCKF feature tracking error",
                                             lambda x: x.xvio_tracks_error, lambda x: x.xvio_tracking_evaluation_config)


def plot_xvio_features_position_changes(pc: PlotContext, d: EvaluationData):
    tracks = convert_xvio_to_rpg_tracks(d.feature_data.df_xvio_tracks)
    xychanges = get_feature_changes_xy(tracks)
    bins, hist = create_pixel_change_histogram(xychanges[:, 0])
    ax = pc.get_axis()
    ax.set_title("Feature position change in x")
    hist_from_bin_values(ax, bins, hist, "Differential feature position [px]", True)
    bins, hist = create_pixel_change_histogram(xychanges[:, 1])
    ax = pc.get_axis()
    ax.set_title("Feature position change in y")
    hist_from_bin_values(ax, bins, hist, "Differential feature position [px]", True)


def get_feature_changes_xy(tracks):
    tracks = tracks[tracks[:, 1].argsort(kind='stable')]
    tracks = tracks[tracks[:, 0].argsort(kind='stable')]
    # pick id, x, y
    tracks = tracks[:, [0, 2, 3]]
    diff = tracks[1:, :] - tracks[:-1, :]
    changes_xy = diff[diff[:, 0] == 0, 1:]
    return changes_xy


def get_feature_optical_flow(tracks):
    tracks = tracks[tracks[:, 1].argsort(kind='stable')]
    tracks = tracks[tracks[:, 0].argsort(kind='stable')]
    diff = tracks[1:, :] - tracks[:-1, :]
    diff = diff[diff[:, 0] == 0, 1:]
    time_diff = diff[:, 0]
    time_diff[time_diff < 1e-3] = 1
    of_xy = diff[:, 1:]
    of_xy[:, 0] /= time_diff
    of_xy[:, 1] /= time_diff
    return of_xy


def get_feature_ages(tracks):
    tracks = tracks[tracks[:, 1].argsort(kind='stable')]
    tracks = tracks[tracks[:, 0].argsort(kind='stable')]

    id_diff = tracks[1:, 0] - tracks[:-1, 0]
    mask_start = np.hstack(([1], id_diff)) == 1
    mask_end = np.hstack((id_diff, [1])) == 1

    feature_ages = tracks[mask_end, 1] - tracks[mask_start, 1]
    start_times = tracks[mask_start, 1]

    sorted = start_times.argsort(kind='stable')
    return feature_ages[sorted], start_times[sorted]


def get_zero_aligned_feature_errors(errors):
    errors = errors[errors[:, 1].argsort(kind='stable')]
    errors = errors[errors[:, 0].argsort(kind='stable')]

    id_diff = errors[1:, 0] - errors[:-1, 0]
    mask_start = np.hstack(([1], id_diff)) == 1
    mask_end = np.hstack((id_diff, [1])) == 1

    start_indices = np.argwhere(mask_start)
    end_indices = np.argwhere(mask_end)
    lengths = (end_indices - start_indices).flatten() + 1
    start_times = errors[mask_start, 1]

    output = copy.copy(errors)

    output[:, 1] -= np.repeat(start_times, lengths)

    return output


def get_traveled_distance_from_features(tracks):
    changes = get_feature_changes_xy(tracks)
    tracks = tracks[tracks[:, 1].argsort(kind='stable')]
    tracks = tracks[tracks[:, 0].argsort(kind='stable')]
    # pick id, x, y
    tracks = tracks[:, [0, 2, 3]]
    diff = tracks[1:, :] - tracks[:-1, :]
    diff[diff[:, 0] == 1, 1:] = [0, 0]

    id_diff = diff[:, 0]
    mask_start = np.hstack(([1], id_diff)) == 1
    mask_end = np.hstack((id_diff, [1])) == 1
    start_indices = np.argwhere(mask_start)
    end_indices = np.argwhere(mask_end)
    diff = np.vstack(([1, 0, 0], diff))

    traveled_distance = np.cumsum(np.linalg.norm(diff[:, 1:], axis=1))
    start_distance = traveled_distance[mask_start]

    lengths = (end_indices - start_indices).flatten() + 1

    traveled_distance -= np.repeat(start_distance, lengths)

    return changes


def get_feature_update_times(tracks):
    tracks = tracks[tracks[:, 1].argsort(kind='stable')]
    tracks = tracks[tracks[:, 0].argsort(kind='stable')]
    # pick id, t
    tracks = tracks[:, [0, 1]]
    diff = tracks[1:, :] - tracks[:-1, :]

    # re-add time
    diff = np.hstack((tracks[:-1, 1:], diff))
    stamped_update_times = diff[diff[:, 1] == 0, :]
    stamped_update_times = stamped_update_times[:, [0, 2]]
    stamped_update_times = stamped_update_times[stamped_update_times[:, 0].argsort(kind='stable')]
    return stamped_update_times


def create_pixel_change_histogram(data):
    left = np.min([-1, np.floor(np.min(data))])
    right = np.max([1, np.ceil(np.max(data))])
    bins = np.hstack((np.arange(left, -1), np.arange(-1, 1, 0.25), np.arange(1, right + 1)))
    hist, bins = np.histogram(data, bins)
    return bins, hist


def plot_xvio_features_update_interval(pc: PlotContext, d: EvaluationData):
    tracks = convert_xvio_to_rpg_tracks(d.feature_data.df_xvio_tracks)
    time_changes = get_feature_update_times(tracks)

    # time_changes = time_changes[time_changes > 0]
    # updates_per_sec = 1 / time_changes

    boxplot(pc, [time_changes[:, 1]], ["Test"], "Update interval")


def plot_feature_plots(d: EvaluationData, output_folder):
    with PlotContext(os.path.join(output_folder, "backend_num_features"), subplot_rows=2, subplot_cols=2) as pc:
        plot_xvio_num_features(pc, [d])

    if d.feature_data.df_eklt_num_features is not None:
        df = d.feature_data.df_eklt_num_features
    with PlotContext(os.path.join(output_folder, "xvio_feature_pos_changes "), subplot_rows=1, subplot_cols=2) as pc:
        plot_xvio_features_position_changes(pc, d)

    with PlotContext(os.path.join(output_folder, "xvio_feature_update_interval "), subplot_rows=1, subplot_cols=2)\
            as pc:
        plot_xvio_features_update_interval(pc, d)

    with PlotContext(os.path.join(output_folder, "backend_feature_pos_changes ")) as pc:
        plot_xvio_feature_pos_changes(pc, d)

    with PlotContext(os.path.join(output_folder, "backend_feature_tracking_error ")) as pc:
        plot_tracking_error(pc, d.feature_data.xvio_tracks_error, d.feature_data.xvio_tracking_evaluation_config,
                            "SLAM and MSCKF feature tracking errors")

    if d.feature_data.df_eklt_num_features is not None:
        df = d.feature_data.df_eklt_num_features

        with PlotContext(os.path.join(output_folder, "eklt_features")) as pc:
            time_series_plot(pc, df['t'], [df['num_features']], ["EKLT features"], "Number of tracked features")

    if d.feature_data.eklt_tracks_error is not None:
        with PlotContext(os.path.join(output_folder, "eklt_feature_tracking_error ")) as pc:
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


def plot_eklt_all_feature_pos_changes(pc: PlotContext, evaluations: List[EvaluationData], names):
    plot_all_feature_pos_changes(pc, evaluations, lambda x: convert_eklt_to_rpg_tracks(x.df_eklt_tracks), names)


def plot_eklt_feature_pos_changes(pc: PlotContext, d: EvaluationData, x_lim=None, y_lim=None, title=None):
    plot_feature_pos_changes(pc, d, lambda x: convert_eklt_to_rpg_tracks(x.df_eklt_tracks), x_lim, y_lim, title)


def plot_xvio_all_feature_pos_changes(pc: PlotContext, evaluations: List[EvaluationData], names):
    plot_all_feature_pos_changes(pc, evaluations, lambda x: convert_xvio_to_rpg_tracks(x.df_xvio_tracks), names)


def plot_xvio_all_feature_optical_flows(pc: PlotContext, evaluations: List[EvaluationData], names):
    plot_all_feature_pos_changes(pc, evaluations, lambda x: convert_xvio_to_rpg_tracks(x.df_xvio_tracks), names,
                                 use_optical_flow=True)


def plot_xvio_feature_pos_changes(pc: PlotContext, d: EvaluationData, x_lim=None, y_lim=None, title=None):
    plot_feature_pos_changes(pc, d, lambda x: convert_xvio_to_rpg_tracks(x.df_xvio_tracks), x_lim, y_lim, title)


def plot_all_feature_pos_changes(pc: PlotContext, evaluations: List[EvaluationData], feature_data_to_rpg_tracks,
                                 names, use_optical_flow=False):

    tracks_to_xy_changes = get_feature_changes_xy

    if use_optical_flow:
        tracks_to_xy_changes = get_feature_optical_flow

    # determine common boundaries for easier comparison
    x_max = 1
    y_max = 1
    for e in evaluations:
        tracks = feature_data_to_rpg_tracks(e.feature_data)
        xy_changes = tracks_to_xy_changes(tracks)
        x_max = np.max(np.abs([x_max, np.quantile(xy_changes[:, 0], 0.01), np.quantile(xy_changes[:, 0], 0.99)]))
        y_max = np.max(np.abs([y_max, np.quantile(xy_changes[:, 1], 0.01), np.quantile(xy_changes[:, 1], 0.99)]))

    for i, e in enumerate(evaluations):
        title = F"Feature position changes on '{e.name}' ({names[i]})"
        if use_optical_flow:
            title = F"Feature's optical flow on '{e.name}' ({names[i]})"
        plot_feature_pos_changes(pc, e, feature_data_to_rpg_tracks, x_lim=[-x_max, x_max], y_lim=[-y_max, y_max],
                                 title=title, tracks_to_xy_changes=tracks_to_xy_changes)


def plot_feature_pos_changes(pc: PlotContext, d: EvaluationData, feature_data_to_rpg_tracks, x_lim=None, y_lim=None,
                             title=None, tracks_to_xy_changes=get_feature_changes_xy):
    ax = pc.get_axis()

    tracks = feature_data_to_rpg_tracks(d.feature_data)
    xy_changes = tracks_to_xy_changes(tracks)

    cmap = copy.copy(plt.get_cmap("Greens"))
    cmap.set_bad('white')
    cmap.set_under('white')

    _, _, _, im = ax.hist2d(xy_changes[:, 0], xy_changes[:, 1], bins=100, cmap=cmap, norm=matplotlib.colors.LogNorm())

    ax.spines['bottom'].set_color("black")
    ax.spines['left'].set_color("black")
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.set_facecolor('white')

    plt.axvline(0, color='black', linestyle='--')
    plt.axhline(0, color='black', linestyle='--')

    if not x_lim:
        x_lim = [np.quantile(xy_changes[:, 0], 0.01), np.quantile(xy_changes[:, 0], 0.99)]
        x_lim_max = np.max(np.abs(x_lim))
        x_lim = [-x_lim_max, x_lim_max]

    if not y_lim:
        y_lim = [np.quantile(xy_changes[:, 1], 0.01), np.quantile(xy_changes[:, 1], 0.99)]
        y_lim_max = np.max(np.abs(y_lim))
        y_lim = [-y_lim_max, y_lim_max]

    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])

    if not title:
        title = F"Feature position changes on '{d.name}'"
    ax.set_title(title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    pc.figure.colorbar(im, cax=cax, orientation='vertical')
    cax.set_ylabel("Occurrence")


def plot_summary_plots(summary: EvaluationDataSummary, output_folder):

    data = []
    auto_labels = []

    for e in summary.data.values():
        if e.feature_data.df_eklt_feature_age is not None:
            data.append(e.feature_data.df_eklt_feature_age['age'])
            auto_labels.append(e.name)

    if len(data) <= 0:
        return

    with PlotContext(os.path.join(output_folder, "eklt_feature_ages ")) as pc:
        plot_eklt_feature_age(pc, summary)


def plot_eklt_feature_age(pc, summary):
    data = [d.feature_data.df_eklt_feature_age['age'] for d in summary.data.values()]
    auto_labels = [d.name for d in summary.data.values()]
    boxplot(pc, data, auto_labels, "Feature age comparison", [1, 99])
