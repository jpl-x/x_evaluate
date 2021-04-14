import os
from typing import List, Collection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from x_evaluate.evaluation_data import PerformanceData, EKLTPerformanceData, EvaluationData, EvaluationDataSummary

RT_FACTOR_RESOLUTION = 0.2


def evaluate_computational_performance(df_rt: pd.DataFrame) -> PerformanceData:
    # for now only save those columns, later also save average processing times for events, IMU, etc.
    d = PerformanceData()
    d.df_realtime = df_rt[['t_sim', 't_real', 'ts_real']]
    realtime_factors = calculate_realtime_factors(df_rt)
    d.rt_factors = realtime_factors
    return d


def calculate_realtime_factors(df_rt, resolution=RT_FACTOR_RESOLUTION):
    rt_factor_calculation = df_rt[['t_sim', 't_real']].to_numpy()
    t_0 = rt_factor_calculation[0, 0]
    delta_t = rt_factor_calculation[-1, 0] - t_0
    t_targets = np.arange(0.0, delta_t, resolution)
    interpolated = np.interp(t_targets, rt_factor_calculation[:, 0] - t_0, rt_factor_calculation[:, 1])
    realtime_factors = np.gradient(interpolated, t_targets)
    return realtime_factors


def evaluate_ektl_performance(perf_data: PerformanceData, df_events: pd.DataFrame, df_optimizations: pd.DataFrame,
                              df_tracks: pd.DataFrame) -> EKLTPerformanceData:
    d = EKLTPerformanceData()
    df_rt = perf_data.df_realtime

    # calculate events / s in simulated and processing time
    event_times = np.interp(df_events['ts_start'], df_rt['ts_real'], df_rt['t_real'])
    bins = np.arange(0.0, event_times[-1], 1.0)
    d.events_per_sec, _ = np.histogram(event_times, bins=bins)
    event_times_sim = np.interp(df_events['ts_start'], df_rt['ts_real'], df_rt['t_sim']) - df_rt['t_sim'][0]
    bins_sim = np.arange(0.0, event_times_sim[-1], 1.0)
    d.events_per_sec_sim, _ = np.histogram(event_times_sim, bins=bins_sim)
    return d


def plot_performance_plots(eval_data: EvaluationData, output_folder):
    plot_realtime_factor(eval_data, os.path.join(output_folder, "realtime_factor.svg"))
    if eval_data.eklt_performance_data is not None:
        plot_events_per_second(eval_data, os.path.join(output_folder, "events_per_second.svg"))


def plot_realtime_factor(eval_data: EvaluationData, filename):
    t_targets = np.arange(0.0, len(eval_data.performance_data.rt_factors)) * RT_FACTOR_RESOLUTION
    plt.figure()
    plt.plot(t_targets, eval_data.performance_data.rt_factors, label=eval_data.name)
    plt.plot(t_targets, np.ones_like(eval_data.performance_data.rt_factors), label="boundary", linestyle="--")
    plt.legend()
    plt.title(F"Realtime factor ({eval_data.name})")
    plt.savefig(filename)
    plt.clf()


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
