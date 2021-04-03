import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class PerformanceEvaluator:

    def __init__(self):
        self._rt_factor = np.array([])
        self._realtime = dict()

    def evaluate(self, name, df_realtime: pd.DataFrame, output_folder):
        self._realtime[name] = df_realtime[['t_sim', 't_real', 'ts_real']]
        df_rt_factor = df_realtime.dropna()[['t_sim', 'rt_factor']]

        rt_factor_calculation = df_realtime[['t_sim', 't_real']].to_numpy()
        t_0 = rt_factor_calculation[0, 0]
        delta_t = rt_factor_calculation[-1, 0] - t_0
        t_targets = np.arange(0.0, delta_t, 0.2)
        interpolated = np.interp(t_targets, rt_factor_calculation[:, 0] - t_0, rt_factor_calculation[:, 1])
        grad = np.gradient(interpolated, t_targets)
        # data = np.vstack((t_targets, grad)).T
        plt.figure()
        plt.plot(t_targets, grad, label=name)
        plt.plot(t_targets, np.ones_like(grad), label="boundary", linestyle="--")
        plt.legend()
        plt.title(F"Realtime factor ({name})")
        plt.savefig(os.path.join(output_folder, "realtime_factor.svg"))

        self._rt_factor = np.hstack((self._rt_factor, df_rt_factor["rt_factor"].to_numpy()))

        # eventually analyze profiling_json here e.g.:
        # json_data['threads'][0]['children']

        # print(df_rt_factor)
        # np.median(df_rt_factor["rt_factor"].to_numpy())

    def evaluate_eklt(self, name, df_events: pd.DataFrame, df_optimizations: pd.DataFrame,
                 df_tracks: pd.DataFrame, output_folder):
        print("Here we go")
        df_realtime = self._realtime[name]

        event_times = np.interp(df_events['ts_start'], df_realtime['ts_real'], df_realtime['t_real'])

        bins = np.arange(0.0, event_times[-1], 1.0)
        events_per_second, _ = np.histogram(event_times, bins=bins)

        event_times_sim = np.interp(df_events['ts_start'], df_realtime['ts_real'], df_realtime['t_sim']) -  \
                          df_realtime['t_sim'][0]
        bins_sim = np.arange(0.0, event_times_sim[-1], 1.0)
        events_per_second_sim, _ = np.histogram(event_times_sim, bins=bins_sim)

        plt.figure()
        plt.title(F"Events per second ({name})")
        plt.plot(bins[:-1], events_per_second, label="processed")
        plt.plot(bins_sim[:-1], events_per_second_sim, label="demanded")
        plt.xlabel("time")
        plt.ylabel("events/s")
        plt.legend()
        plt.savefig(os.path.join(output_folder, "events_per_second.svg"))

    def print_summary(self):
        print(F"Realtime factor (worst case): {np.median(self._rt_factor):>6.2f} ({np.max(self._rt_factor):.2f})")
