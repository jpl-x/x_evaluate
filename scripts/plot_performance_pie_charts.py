import argparse
import glob
import os

import numpy as np
import orjson
import pandas as pd
from matplotlib import pyplot as plt

from x_evaluate.evaluation_data import FrontEnd

from x_evaluate.plots import PlotContext, DEFAULT_COLORS

import x_evaluate.plots

from x_evaluate.utils import name_to_identifier

from x_evaluate.scriptlets import find_evaluation_files_recursively, read_evaluation_pickle, cache


# https://stackoverflow.com/a/49601444
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def main():
    parser = argparse.ArgumentParser(description='Plots pie charts from profiling.json files')
    parser.add_argument('--input_folder', type=str, required=True)

    args = parser.parse_args()

    root_folder = args.input_folder
    evaluation_files = find_evaluation_files_recursively(root_folder)
    input_folders = [os.path.dirname(f) for f in evaluation_files]

    x_evaluate.plots.use_paper_style_plots = True

    summaries = []
    for input_folder in input_folders:
        summaries.append(read_evaluation_pickle(input_folder))

    for i, s in enumerate(summaries):
        second_level_table = None
        third_level_table = None
        for dataset in s.data.keys():
            subfolder = name_to_identifier(dataset)

            match_str = os.path.join(input_folders[i], F"*{subfolder}*/profiling.json")

            matches = glob.glob(match_str)
            if len(matches) == 0:
                print(F"WARNING: no profiling.json found for '{subfolder}'")
                continue

            if len(matches) > 1:
                print(f"Warning multiple profiling.json found for '{match_str}' ({matches}), taking first")

            def flatten_to_df():
                with open(matches[0], "rb") as f:
                    profiling_json = orjson.loads(f.read())

                data = []  # [descriptor, id, name, start, stop, level, parent]

                for row in profiling_json['threads'][0]['children']:
                    def recursive_flattening(element, level, parent):
                        ret = []
                        ret += [[element['descriptor'], element['id'], element['name'], element['start'], element['stop'],
                                level, parent]]
                        if 'children' in element.keys():
                            for c in element['children']:
                                ret += recursive_flattening(c, level+1, element['id'])
                        return ret
                    data += recursive_flattening(row, 0, -1)

                return pd.DataFrame(data, columns=['descriptor', 'id', 'name', 'start', 'stop', 'level', 'parent'])

            stats = cache(F"{matches[0]}.pickle", flatten_to_df)

            top_level = stats.loc[stats.level == 0]

            relevant_processing_time = get_processing_time(top_level.loc[top_level.name != 'GT Message'])

            print(F"Overall relevant time: {relevant_processing_time}")

            second_level_timings = get_stats_for_level(stats, 1)
            third_level_timings = get_stats_for_level(stats, 2)

            # print(second_level_timings)
            # print(np.sum(list(second_level_timings.values())))
            # print()
            # print(third_level_timings)
            # print(np.sum(list(third_level_timings.values())))
            # print()

            second = pd.DataFrame(second_level_timings, [dataset])
            third = pd.DataFrame(third_level_timings, [dataset])

            if second_level_table is None:
                second_level_table = second
            else:
                second_level_table = pd.concat([second_level_table, second])

            if third_level_table is None:
                third_level_table = third
            else:
                third_level_table = pd.concat([third_level_table, third])

        pd.options.display.max_colwidth = None
        pd.options.display.width = 0

        second_level_table['Total'] = second_level_table.sum(axis=1)
        third_level_table['Total'] = third_level_table.sum(axis=1)

        second_level_overall = second_level_table.sum()
        third_level_overall = third_level_table.sum()

        second_level_overall_perc = (second_level_overall / second_level_overall['Total'] * 100)
        third_level_overall_perc = (third_level_overall / third_level_overall['Total'] * 100)

        inner_labels = ["\\textbf{Frontend}", "\\textbf{Backend}"]
        if s.frontend == FrontEnd.XVIO:
            # FIXME an easier data structure would be sufficient, much redundancies in here...
            backend = second_level_overall_perc[["EKF IMU Update", "Backend EKF Update", "Backend feature management"]].sum()
            inner_numbers = [second_level_overall["XVIO frontend"], backend]
            outer_labels = ["Detection", "Tracking", "RANSAC", "\\noindent\\hspace{1.5em}IMU\\\\propagation",
                            "\\noindent "
                                                                                                 "Visual\\\\ update",
                            "\\noindent \\hspace{.9em} Feature\\\\ management"]
            inner_outer_lengths = [3, 3]
            outer_numbers = list(third_level_overall_perc * inner_numbers[0] / 100)[:3] + list(
                second_level_overall_perc[["EKF IMU Update", "Backend EKF Update", "Backend feature management"]])
        elif s.frontend == FrontEnd.HASTE:
            f = 100 / second_level_overall['Total']

            inner_numbers = [(second_level_overall['HASTE Tracking'] + third_level_overall['RANSAC outlier removal'])*f,
                             (third_level_overall[['Backend feature management', 'Backend EKF Update']].sum() +
                              second_level_overall['EKF IMU Update'])*f]

            outer_numbers = [third_level_overall['RANSAC outlier removal']*f, second_level_overall['HASTE Tracking']*f,
                             second_level_overall['EKF IMU Update']*f,
                             third_level_overall['Backend EKF Update']*f, third_level_overall['Backend feature management']*f]

            outer_labels = ["RANSAC", "HASTE Tracking", "\\noindent\\hspace{1.5em}IMU\\\\propagation",
                            "\\noindent "
                            "Visual\\\\ update",
                            "\\noindent \\hspace{.9em} Feature\\\\ management"]
            inner_outer_lengths = [2, 3]
        elif s.frontend == FrontEnd.EKLT:
            f = 100 / second_level_overall['Total']

            inner_numbers = [second_level_overall['EKLT Tracking']*f,
                             (third_level_overall[['Backend feature management', 'Backend EKF Update']].sum() +
                              second_level_overall['EKF IMU Update'])*f]

            outer_numbers = [second_level_overall['EKLT Tracking']*f,
                             # second_level_overall['EKF IMU Update']*f,
                             third_level_overall['Backend EKF Update']*f, third_level_overall['Backend feature management']*f]

            outer_labels = ["\\noindent\\hspace{.9em}EKLT\\\\ Tracking",
                            "\\noindent "
                            "Visual\\\\ update",
                            "\\noindent \\hspace{.9em} Feature\\\\ management"]
            inner_outer_lengths = [1, 2]
        else:
            print("NOT IMPLEMENTED")
            continue

        with PlotContext(os.path.join(input_folders[i], "computational_pie_plot")) as pc:
            ax = pc.get_axis()
            size = 0.3

            inner_colors = DEFAULT_COLORS[:len(inner_outer_lengths)]
            outer_colors = []
            for i, c in enumerate(inner_colors):
                outer_colors += [lighten_color(c, d) for d in np.linspace(0.25, .7, inner_outer_lengths[i])]

            _, texts, _ = ax.pie(outer_numbers, radius=1,
                   labels=outer_labels,
                   autopct='%1.0f%%',
                   colors=outer_colors,
                   pctdistance=.85,
                   # labeldistance=1.2,
                   wedgeprops=dict(width=size, edgecolor='w')
                   )

            # for tex in texts:
            #     tex.set_fontweight(5)

            ax.pie(inner_numbers, radius=1 - size,
                   labels=inner_labels,
                   autopct='%1.0f%%',
                   labeldistance=0.15,
                   pctdistance=0.75,
                   wedgeprops=dict(width=size, edgecolor='w'),
                   colors=inner_colors
                   )

            ax.set(aspect="equal", title='Pie plot with `ax.pie`')

            # print(len(data))

        plt.show()

    print("DONE")


def get_stats_for_level(stats, level):
    second_level_timings = dict()
    for name in stats.loc[stats.level == level].name.unique():
        t = get_processing_time(stats.loc[(stats.level == level) & (stats.name == name)])
        second_level_timings[name] = t
    return second_level_timings


def get_processing_time(df_stats):
    return (df_stats['stop'] - df_stats['start']).sum() / 1e9


if __name__ == '__main__':
    main()
