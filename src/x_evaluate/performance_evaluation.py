import os
from typing import Collection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from x_evaluate.evaluation_data import PerformanceData, EKLTPerformanceData, EvaluationData, EvaluationDataSummary
from x_evaluate.utils import timestamp_to_real_time, timestamp_to_rosbag_time_zero
from x_evaluate.plots import boxplot, time_series_plot

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
    bins = np.arange(0.0, event_times[-1], 1.0)
    d.events_per_sec, _ = np.histogram(event_times, bins=bins)
    event_times_sim = timestamp_to_rosbag_time_zero(df_events['ts_start'], df_rt)
    bins_sim = np.arange(0.0, event_times_sim[-1], 1.0)
    d.events_per_sec_sim, _ = np.histogram(event_times_sim, bins=bins_sim)

    optimization_times = timestamp_to_real_time(df_optimizations['ts_start'], df_rt)
    bins = np.arange(0.0, optimization_times[-1], 1.0)
    d.optimizations_per_sec, _ = np.histogram(optimization_times, bins=bins)
    d.optimization_iterations = df_optimizations['num_iterations'].to_numpy()

    return d


def plot_resources(eval_data: EvaluationData, output_folder):
    df_rt = eval_data.performance_data.df_realtime
    df_resources = eval_data.performance_data.df_resources

    resource_times = timestamp_to_real_time(df_resources['ts'], df_rt)

    data = [df_resources['cpu_usage'], df_resources['cpu_user_mode_usage'], df_resources['cpu_kernel_mode_usage']]
    labels = ["CPU total", "CPU user mode", "CPU kernel mode"]
    file = os.path.join(output_folder, "cpu_usage.svg")
    time_series_plot(file, resource_times, data, labels, F"CPU Usage ({eval_data.name})", "cpu time : real time [%]")

    mem_usage = df_resources['memory_usage_in_bytes'].to_numpy() / 1024 / 1024
    mem_usage_debug = df_resources['debug_memory_in_bytes'].to_numpy() / 1024 / 1024

    actual_mem = mem_usage - mem_usage_debug

    data = [actual_mem, mem_usage_debug]
    labels = ["memory usage", "additional debug memory"]
    file = os.path.join(output_folder, "memory_usage.svg")
    time_series_plot(file, resource_times, data, labels, F"Memory Usage ({eval_data.name})", "MB")


def plot_performance_plots(eval_data: EvaluationData, output_folder):
    plot_realtime_factor([eval_data], os.path.join(output_folder, "realtime_factor.svg"))
    plot_resources(eval_data, output_folder)
    if eval_data.eklt_performance_data is not None:
        plot_events_per_second(eval_data, os.path.join(output_folder, "events_per_second.svg"))
        plot_optimizations_per_second(eval_data, os.path.join(output_folder, "optimizations_per_second.svg"))
        plot_optimization_iterations([eval_data], os.path.join(output_folder, "optimization_iterations.svg"))


def plot_optimization_iterations(evaluations: Collection[EvaluationData], filename, labels=None):
    auto_labels = []
    data = []
    for d in evaluations:
        if d.eklt_performance_data is not None:
            data.append(d.eklt_performance_data.optimization_iterations)
            auto_labels.append(d.name)

    if len(auto_labels) <= 0:
        return

    if labels is None:
        labels = auto_labels

    boxplot(filename, data, labels, "Optimization iterations")


def plot_realtime_factor(evaluations: Collection[EvaluationData], filename, labels=None):
    plt.figure()
    max_length = 0

    i = 0

    for d in evaluations:
        length = len(d.performance_data.rt_factors)
        max_length = max(length, max_length)
        t_targets = np.arange(0.0, length) * RT_FACTOR_RESOLUTION
        label = d.name
        if labels is not None:
            label = labels[i]
            # this causes issues, quick fix:
            if label.startswith('_'):
                label = label[1:]
            i += 1
        plt.plot(t_targets, d.performance_data.rt_factors, label=label)

    t_targets = np.arange(0.0, max_length) * RT_FACTOR_RESOLUTION
    plt.plot(t_targets, np.ones_like(t_targets), label="boundary", linestyle="--")
    plt.legend()
    plt.title(F"Realtime factor")
    plt.savefig(filename)
    plt.clf()


def plot_summary_plots(summary: EvaluationDataSummary, output_folder):
    plot_optimization_iterations(summary.data.values(), os.path.join(output_folder, "optimization_iterations.svg"))


def print_realtime_factor_summary(eval_data_summary: EvaluationDataSummary):
    rt_factors = combine_rt_factors(eval_data_summary.data.values())
    print(F"Realtime factor (worst case): {np.median(rt_factors):>6.2f} ({np.max(rt_factors):.2f})")


def combine_rt_factors(evaluations: Collection[EvaluationData]) -> np.ndarray:
    arrays = tuple([d.performance_data.rt_factors for d in evaluations])
    return np.hstack(arrays)


def plot_events_per_second(eval_data: EvaluationData, filename):
    plt.figure()
    plt.title(F"Events per second ({eval_data.name})")
    t = np.arange(0.0, len(eval_data.eklt_performance_data.events_per_sec))
    plt.plot(t, eval_data.eklt_performance_data.events_per_sec, label="processed")
    t = np.arange(0.0, len(eval_data.eklt_performance_data.events_per_sec_sim))
    plt.plot(t, eval_data.eklt_performance_data.events_per_sec_sim, label="demanded")
    plt.xlabel("time")
    plt.ylabel("events/s")
    plt.legend()
    plt.savefig(filename)
    plt.clf()


def plot_optimizations_per_second(eval_data: EvaluationData, filename):
    plt.figure()
    plt.title(F"Optimizations per second ({eval_data.name})")
    t = np.arange(0.0, len(eval_data.eklt_performance_data.optimizations_per_sec))
    plt.plot(t, eval_data.eklt_performance_data.optimizations_per_sec)
    plt.xlabel("time")
    plt.ylabel("optimizations/s")
    # plt.legend()
    plt.savefig(filename)
    plt.clf()
