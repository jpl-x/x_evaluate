import os
from typing import Collection, Dict, List

import pandas as pd
import numpy as np
from x_evaluate.evaluation_data import PerformanceData, EKLTPerformanceData, EvaluationData, EvaluationDataSummary, \
    DistributionSummary
from x_evaluate.utils import timestamp_to_real_time, timestamp_to_rosbag_time_zero
from x_evaluate.plots import time_series_plot, PlotContext, boxplot_from_summary, barplot_compare, hist_from_bin_values, \
    boxplot_compare, summary_to_dict, draw_lines_on_top_of_comparison_plots, DEFAULT_COLORS

RT_FACTOR_RESOLUTION = 0.2


def evaluate_computational_performance(df_rt: pd.DataFrame, df_resources: pd.DataFrame) -> PerformanceData:
    # for now only save those columns, later also save average processing times for events, IMU, etc.
    d = PerformanceData()
    d.df_realtime = df_rt[['t_sim', 't_real', 'ts_real']]
    realtime_factors = calculate_realtime_factors(df_rt)
    d.rt_factors = realtime_factors
    d.df_resources = df_resources
    return d


def calculate_realtime_factors(df_rt, resolution=RT_FACTOR_RESOLUTION):
    rt_factor_calculation = df_rt[['t_sim', 't_real']].to_numpy()
    t_0 = rt_factor_calculation[0, 0]
    delta_t = rt_factor_calculation[-1, 0] - t_0
    t_targets = np.arange(0.0, delta_t, resolution)
    interpolated = np.interp(t_targets, rt_factor_calculation[:, 0] - t_0, rt_factor_calculation[:, 1])
    realtime_factors = np.gradient(interpolated, t_targets)
    return realtime_factors


def evaluate_ektl_performance(perf_data: PerformanceData, df_events: pd.DataFrame,
                              df_optimizations: pd.DataFrame) -> EKLTPerformanceData:
    d = EKLTPerformanceData()
    df_rt = perf_data.df_realtime

    # calculate events / s in simulated and processing time
    event_times = timestamp_to_real_time(df_events['ts_start'], df_rt)
    event_times_end = timestamp_to_real_time(df_events['ts_stop'], df_rt)
    bins = np.arange(0.0, event_times[-1], 1.0)
    d.events_per_sec, _ = np.histogram(event_times, bins=bins)
    event_times_sim = timestamp_to_rosbag_time_zero(df_events['ts_start'], df_rt)
    bins_sim = np.arange(0.0, event_times_sim[-1], 1.0)
    d.events_per_sec_sim, _ = np.histogram(event_times_sim, bins=bins_sim)

    optimization_times = timestamp_to_real_time(df_optimizations['ts_start'], df_rt)
    bins = np.arange(0.0, optimization_times[-1], 1.0)
    d.optimizations_per_sec, _ = np.histogram(optimization_times, bins=bins)
    d.optimization_iterations = DistributionSummary(df_optimizations['num_iterations'].to_numpy())
    d.event_processing_times = DistributionSummary(event_times_end - event_times)

    return d


def plot_cpu_usage(pc: PlotContext, eval_data: EvaluationData):
    df_rt = eval_data.performance_data.df_realtime
    df_resources = eval_data.performance_data.df_resources

    resource_times = timestamp_to_real_time(df_resources['ts'], df_rt)

    data = [df_resources['cpu_usage'], df_resources['cpu_user_mode_usage'], df_resources['cpu_kernel_mode_usage']]
    labels = ["CPU total", "CPU user mode", "CPU kernel mode"]
    time_series_plot(pc, resource_times, data, labels, F"CPU Usage ({eval_data.name})", "cpu time : real time [%]")


def plot_cpu_usage_in_time_comparison(pc: PlotContext, eval_data: List[EvaluationData], labels, dataset_name):
    data = []
    times = []

    for e in eval_data:
        data.append(e.performance_data.df_resources['cpu_usage'])
        times.append(timestamp_to_real_time(e.performance_data.df_resources['ts'], e.performance_data.df_realtime))

    time_series_plot(pc, times, data, labels, F"CPU usage on '{dataset_name}'", "cpu time : real time [%]")


def plot_cpu_usage_boxplot_comparison(pc: PlotContext, summaries: Dict[str, EvaluationDataSummary], common_datasets):
    data = [[s.data[k].performance_data.df_resources['cpu_usage'] for k in common_datasets] for s in summaries.values()]

    summary_labels = [s.name for s in summaries.values()]
    dataset_labels = common_datasets
    boxplot_compare(pc.get_axis(), dataset_labels, data, summary_labels, ylabel="CPU usage [%]", title="Total CPU usage")


def plot_memory_usage(pc: PlotContext, eval_data: EvaluationData):
    df_rt = eval_data.performance_data.df_realtime
    df_resources = eval_data.performance_data.df_resources
    resource_times = timestamp_to_real_time(df_resources['ts'], df_rt)
    mem_usage = df_resources['memory_usage_in_bytes'].to_numpy() / 1024 / 1024
    mem_usage_debug = df_resources['debug_memory_in_bytes'].to_numpy() / 1024 / 1024

    actual_mem = mem_usage - mem_usage_debug

    data = [actual_mem, mem_usage_debug]
    labels = ["memory usage", "additional debug memory"]
    time_series_plot(pc, resource_times, data, labels, F"Memory Usage ({eval_data.name})", "MB")


def plot_memory_usage_in_time_comparison(pc: PlotContext, eval_data: List[EvaluationData], labels, dataset_name):
    data = []
    times = []

    for e in eval_data:
        df_resources = e.performance_data.df_resources
        resource_times = timestamp_to_real_time(df_resources['ts'], e.performance_data.df_realtime)
        mem_usage = df_resources['memory_usage_in_bytes'].to_numpy() / 1024 / 1024
        data.append(mem_usage)
        times.append(resource_times)

    time_series_plot(pc, times, data, labels, F"Memory usage on '{dataset_name}')", "Memory usage [MB]")


def plot_memory_usage_boxplot_comparison(pc: PlotContext, summaries: Dict[str, EvaluationDataSummary], common_datasets):
    data = [[s.data[k].performance_data.df_resources['memory_usage_in_bytes'].to_numpy() / 1024 / 1024
             for k in common_datasets] for s in summaries.values()]

    summary_labels = [s.name for s in summaries.values()]
    dataset_labels = common_datasets
    boxplot_compare(pc.get_axis(), dataset_labels, data, summary_labels, ylabel="Memory usage [MB]",
                    title="Total memory usage")


def plot_performance_plots(eval_data: EvaluationData, output_folder):
    with PlotContext(os.path.join(os.path.join(output_folder, "realtime_factor"))) as pc:
        plot_realtime_factor(pc, [eval_data])
    with PlotContext(os.path.join(output_folder, "cpu_usage")) as pc:
        plot_cpu_usage(pc, eval_data)
    with PlotContext(os.path.join(output_folder, "memory_usage")) as pc:
        plot_memory_usage(pc, eval_data)
    if eval_data.eklt_performance_data is not None:
        with PlotContext(os.path.join(output_folder, "events_per_second")) as pc:
            plot_events_per_second(pc, eval_data)
        with PlotContext(os.path.join(output_folder, "optimizations_per_second")) as pc:
            plot_optimizations_per_second(pc, eval_data)
        with PlotContext(os.path.join(output_folder, "optimization_iterations")) as pc:
            plot_optimization_iterations(pc, [eval_data.eklt_performance_data], [eval_data.name])


def plot_optimization_iterations(pc: PlotContext, evaluations: List[EKLTPerformanceData], labels):
    data = [e.optimization_iterations for e in evaluations]

    boxplot_from_summary(pc, data, labels, "Optimization iterations")


def plot_optimization_iterations_comparison(pc: PlotContext, summaries: List[EvaluationDataSummary], common_datasets):
    data = [[summary_to_dict(s.data[k].eklt_performance_data.optimization_iterations)
             for k in common_datasets] for s in summaries]

    summary_labels = [s.name for s in summaries]
    dataset_labels = common_datasets
    boxplot_compare(pc.get_axis(), dataset_labels, data, summary_labels, ylabel="Number of iterations",
                    title="Optimization iterations")
    # boxplot_compare()
    # boxplot_from_summary(pc, data, labels, "Optimization iterations")


def plot_realtime_factor(pc: PlotContext, evaluations: Collection[EvaluationData], labels=None, use_log=False,
                         title=None):
    ax = pc.get_axis()
    max_length = 0

    for i, d in enumerate(evaluations):
        length = len(d.performance_data.rt_factors)
        max_length = max(length, max_length)
        t_targets = np.arange(0.0, length) * RT_FACTOR_RESOLUTION
        label = d.name
        if labels is not None:
            label = labels[i]
            # this causes issues, quick fix:
            if label.startswith('_'):
                label = label[1:]
        ax.plot(t_targets, d.performance_data.rt_factors, label=label, color=DEFAULT_COLORS[i])

    t_targets = np.arange(0.0, max_length) * RT_FACTOR_RESOLUTION
    ax.plot(t_targets, np.ones_like(t_targets), label="boundary", linestyle="--", color="black")
    ax.legend()
    ax.set_ylabel("Realtime factor")
    ax.set_xlabel("t [s]")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(F"Realtime factor")
    if use_log:
        ax.set_yscale('log')


def plot_summary_plots(summary: EvaluationDataSummary, output_folder):
    eklt_performance_data = [e.eklt_performance_data for e in summary.data.values() if e.eklt_performance_data is not None]
    eklt_names = [e.name for e in summary.data.values() if e.eklt_performance_data is not None]

    if len(eklt_performance_data) > 0:
        with PlotContext(os.path.join(output_folder, "optimization_iterations")) as pc:
            plot_optimization_iterations(pc, eklt_performance_data, eklt_names)


def print_realtime_factor_summary(eval_data_summary: EvaluationDataSummary):
    rt_factors = combine_rt_factors(eval_data_summary.data.values())
    print(F"Realtime factor (worst case): {np.median(rt_factors):>6.2f} ({np.max(rt_factors):.2f})")


def combine_rt_factors(evaluations: Collection[EvaluationData]) -> np.ndarray:
    arrays = tuple([d.performance_data.rt_factors for d in evaluations])
    return np.hstack(arrays)


def plot_events_per_second(pc: PlotContext, eval_data: EvaluationData):
    ax = pc.get_axis()
    ax.set_title(F"Events per second ({eval_data.name})")
    t = np.arange(0.0, len(eval_data.eklt_performance_data.events_per_sec))
    ax.plot(t, eval_data.eklt_performance_data.events_per_sec, label="processed")
    t = np.arange(0.0, len(eval_data.eklt_performance_data.events_per_sec_sim))
    ax.plot(t, eval_data.eklt_performance_data.events_per_sec_sim, label="demanded")
    ax.set_xlabel("time")
    ax.set_ylabel("events/s")
    ax.legend()


def plot_events_per_seconds_comparison(pc: PlotContext, eval_data: List[EvaluationData], labels, dataset_name):
    times = []
    data = []

    for e in eval_data:
        times.append(np.arange(0.0, len(e.eklt_performance_data.events_per_sec)))
        data.append(e.eklt_performance_data.events_per_sec)

    time_series_plot(pc, times, data, labels, F"Processed events on '{dataset_name}'", "events/s")


def plot_optimizations_per_second(pc: PlotContext, eval_data: EvaluationData):
    ax = pc.get_axis()
    ax.set_title(F"Optimizations per second ({eval_data.name})")
    t = np.arange(0.0, len(eval_data.eklt_performance_data.optimizations_per_sec))
    ax.plot(t, eval_data.eklt_performance_data.optimizations_per_sec)
    ax.set_xlabel("time")
    ax.set_ylabel("optimizations/s")
    # ax.legend()


def plot_optimizations_per_seconds_comparison(pc: PlotContext, eval_data: List[EvaluationData], labels, dataset_name):
    times = []
    data = []

    for e in eval_data:
        times.append(np.arange(0.0, len(e.eklt_performance_data.optimizations_per_sec)))
        data.append(e.eklt_performance_data.optimizations_per_sec)

    time_series_plot(pc, times, data, labels, F"Performed optimizations on '{dataset_name}'", "optimizations/s")


def plot_processing_times(pc: PlotContext, summaries: Dict[str, EvaluationDataSummary], common_datasets: List[str]):
    data = [[s.data[k].performance_data.df_realtime.iloc[-1]['t_real'] for k in common_datasets] for s in
            summaries.values()]

    summary_labels = [s.name for s in summaries.values()]
    dataset_labels = common_datasets

    ax = pc.get_axis()
    barplot_compare(ax, dataset_labels, data, summary_labels, ylabel="Total processing time [s]",
                    title="Total processing times")

    data = [[list(summaries.values())[0].data[k].performance_data.df_realtime.iloc[-1]['t_sim'] - list(
        summaries.values())[0].data[k].performance_data.df_realtime.iloc[0]['t_sim']] for k in common_datasets]
    draw_lines_on_top_of_comparison_plots(ax, data, len(summary_labels))


def plot_event_processing_times(pc: PlotContext, eklt_evaluations: List[EvaluationData], names: List[str]):
    share_axis = None

    axis = []

    for idx, e in enumerate(eklt_evaluations):
        if share_axis is None:
            share_axis = pc.get_axis()
            ax = share_axis
        else:
            ax = pc.get_axis(sharex=share_axis)
        hist_from_bin_values(ax, e.eklt_performance_data.event_processing_times.bins_log,
                             e.eklt_performance_data.event_processing_times.hist_log, "Time / event [s]", True, True)

        ax.set_title(names[idx])
        axis.append(ax)

    for a in axis:
        a.label_outer()

    #
    #
    # ax = pc.get_axis()
    # ax.set_xlabel("Distance traveled [m]")
    # if use_log:
    #     ax.set_ylabel(F"APE (log {m.unit.name})")
    # else:
    #     ax.set_ylabel(F"APE ({m.unit.name})")
    # boxplot_compare(ax, distances, data, labels)

