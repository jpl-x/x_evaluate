import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class PerformanceEvaluator:

    def __init__(self):
        self._rt_factor = np.array([])

    def evaluate(self, name, df_realtime: pd.DataFrame, profiling_json, output_folder):
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

    def print_summary(self):
        print(F"Realtime factor (worst case): {np.median(self._rt_factor):>6.2f} ({np.max(self._rt_factor):.2f})")
