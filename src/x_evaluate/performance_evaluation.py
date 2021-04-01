import pandas as pd
import numpy as np


class PerformanceEvaluator:

    def __init__(self):
        self._rt_factor = np.array([])

    def evaluate(self, df_realtime: pd.DataFrame, profiling_json):
        df_rt_factor = df_realtime.dropna()[['t_sim', 'rt_factor']]

        self._rt_factor = np.hstack((self._rt_factor, df_rt_factor["rt_factor"].to_numpy()))

        # eventually analyze profiling_json here e.g.:
        # json_data['threads'][0]['children']

        # print(df_rt_factor)
        # np.median(df_rt_factor["rt_factor"].to_numpy())

    def print_summary(self):
        print(F"Realtime factor (worst case): {np.median(self._rt_factor):>6.2f} ({np.max(self._rt_factor):.2f})")
