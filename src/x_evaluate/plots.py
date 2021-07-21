from copy import copy
from enum import Enum
from typing import List
import numpy as np
from matplotlib import pyplot as plt
from x_evaluate.evaluation_data import DistributionSummary

import matplotlib.colors as mcolors

from x_evaluate.utils import get_quantized_statistics_along_axis

DEFAULT_COLORS = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.BASE_COLORS.values()) + \
                 list(mcolors.CSS4_COLORS.values())


class PlotType(Enum):
    BOXPLOT = 1
    TIME_SERIES = 2


class PlotContext:
    figure: plt.Figure
    axis: List[plt.Axes]
    FORMATS = [".png"]

    def __init__(self, filename=None, subplot_rows=1, subplot_cols=1, base_width_inch=10, base_height_inch=7):
        self.filename = filename
        self.subplot_rows = subplot_rows
        self.subplot_cols = subplot_cols
        self.width_inch = base_width_inch * subplot_cols
        self.height_inch = base_height_inch * subplot_rows
        self.axis = []

    def __enter__(self):
        # plt.rc('text', usetex=True)
        # plt.rc('axes', linewidth=3)
        self.figure = plt.figure()
        self.subplot_idx = 0
        self.figure.set_size_inches(self.width_inch, self.height_inch)
        return self

    def get_axis(self, subplot_arg=None, **kwargs) -> plt.Axes:
        self.subplot_idx += 1
        if subplot_arg:
            ax = self.figure.add_subplot(subplot_arg)
        else:
            ax = self.figure.add_subplot(self.subplot_rows, self.subplot_cols, self.subplot_idx, **kwargs)
        self.axis.append(ax)
        return ax

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.figure.tight_layout()
        if self.filename is None:
            self.figure.show()
        else:
            for f in self.FORMATS:
                self.figure.savefig(self.filename + f)

        for a in self.axis:
            a.set_xscale('linear')  # workaround for https://github.com/matplotlib/matplotlib/issues/9970
            a.set_yscale('linear')  # workaround for https://github.com/matplotlib/matplotlib/issues/9970

        self.figure.clf()
        plt.close(self.figure)


def boxplot(pc: PlotContext, data, labels, title="", outlier_params=1.5, use_log=False):
    ax = pc.get_axis()
    ax.boxplot(data, vert=True, labels=labels, whis=outlier_params)
    ax.set_title(title)
    if use_log:
        ax.set_yscale('log')


def summary_to_dict(distribution_summary: DistributionSummary, label=None, use_95_quantiles_as_min_max=False,
                    scaling=1):
    result_dict = {
        'q1': distribution_summary.quantiles[0.25] * scaling,  # First quartile (25th percentile)
        'med': distribution_summary.quantiles[0.5] * scaling,  # Median         (50th percentile)
        'q3': distribution_summary.quantiles[0.75] * scaling,  # Third quartile (75th percentile)
        'fliers': []  # Outliers
    }

    if use_95_quantiles_as_min_max:
        result_dict['whislo'] = distribution_summary.quantiles[0.05] * scaling  # Bottom whisker position
        result_dict['whishi'] = distribution_summary.quantiles[0.95] * scaling  # Top whisker position
    else:
        result_dict['whislo'] = distribution_summary.min * scaling  # Bottom whisker position
        result_dict['whishi'] = distribution_summary.max * scaling  # Top whisker position

    if label is not None:
        result_dict['label'] = label,

    return result_dict


def boxplot_from_summary(pc: PlotContext, distribution_summaries: List[DistributionSummary], labels, title=""):
    ax = pc.get_axis()
    boxes = []

    for i in range(len(distribution_summaries)):
        boxes.append(summary_to_dict(distribution_summaries[i], labels[i]))
    ax.bxp(boxes, showfliers=False)
    ax.set_title(title)


def time_series_plot(pc: PlotContext, time, data, labels, title="", ylabel=None, use_scatter=False, use_log=False,
                     xlabel=None, subplot_arg=None):
    ax = pc.get_axis(subplot_arg)
    for i in range(len(data)):

        # this causes issues, quick fix:
        label = labels[i]
        if label.startswith('_'):
            label = label[1:]

        if isinstance(time, list):
            t = time[i]
        else:
            t = time

        if use_scatter:
            ax.scatter(t, data[i], label=label, color=DEFAULT_COLORS[i])
        else:
            ax.plot(t, data[i], label=label, color=DEFAULT_COLORS[i])

    ax.legend()
    ax.set_title(title)
    if not xlabel:
        xlabel = "Time [s]"
    ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if use_log:
        ax.set_yscale('log')


def bubble_plot(pc: PlotContext, xy_data, labels, y_resolution=0.1, x_resolution=0.1, title=None, ylabel=None,
                xlabel=None, use_log=False):
    ax = pc.get_axis()

    data = []
    times = []
    sizes = []
    s_min = np.iinfo(np.int32).max
    s_max = 0

    for i, xy in enumerate(xy_data):
        x, y, size = create_bubbles_from_2d_point_cloud(xy, x_resolution, y_resolution)
        s_min = min(np.min(size), s_min)
        s_max = max(np.max(size), s_max)
        data.append(y)
        times.append(x)
        sizes.append(size)

    px_size_min = 5
    px_size_max = 15

    for i, d in enumerate(data):
        size = px_size_min + (sizes[i] - s_min) / (s_max - s_min) * (px_size_max - px_size_min)
        size = np.power(size, 2)
        ax.scatter(times[i], d, s=size, label=labels[i], alpha=0.5, color=DEFAULT_COLORS[i])

    ax.legend()
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if use_log:
        ax.set_ylabel('log')


def create_bubbles_from_2d_point_cloud(xy, x_resolution=0.1, y_resolution=0.1):
    x = xy[:, 0]
    y = xy[:, 1]
    x_buckets = np.arange(np.min(x), np.max(x), x_resolution)
    y_buckets = np.arange(np.min(y), np.max(y), y_resolution)
    x_bucket_idx = np.digitize(x, x_buckets)
    x_bucket_idx_uniq = np.unique(x_bucket_idx)

    # filter empty buckets:  (-1 to convert upper bound --> lower bound, as we always take the first errors per bucket)
    # x_buckets = x_buckets[np.clip(x_bucket_idx_uniq - 1, 0, len(x_buckets))]

    xys = np.empty((len(x), 3))

    i = 0

    for idx in x_bucket_idx_uniq:
        # tracking_errors = euclidean_error[bucket_index == idx]
        current_bucket_y = xy[x_bucket_idx == idx, 1]
        y_bucket_idx = np.digitize(current_bucket_y, y_buckets)
        y_bucket_idx_uniq = np.unique(y_bucket_idx)
        used_y_buckets = y_buckets[np.clip(y_bucket_idx_uniq - 1, 0, len(y_buckets))]
        xxs = np.ones_like(used_y_buckets) * x_buckets[idx-1]
        sizes = np.bincount(y_bucket_idx)[1:]

        sizes = sizes[sizes != 0]

        n = len(used_y_buckets)
        xys[i:(i+n), 0] = xxs
        xys[i:(i+n), 1] = used_y_buckets
        xys[i:(i+n), 2] = sizes
        i += n

    xys = xys[:i, :]

    return xys[:, 0], xys[:, 1], xys[:, 2]


def color_box(bp, color):
    elements = ['medians', 'boxes', 'caps', 'whiskers']
    # Iterate over each of the elements changing the color
    for elem in elements:
        [plt.setp(bp[elem][idx], color=color, linestyle='-', lw=1.0)
         for idx in range(len(bp[elem]))]
    return


def barplot_compare(ax: plt.Axes, x_tick_labels, data, legend_labels, ylabel=None, colors=None, legend=True,
                    title=None):
    if colors is None:
        colors = DEFAULT_COLORS

    n_data = len(data)
    n_xlabel = len(x_tick_labels)

    for idx, d in enumerate(data):
        positions, widths = evenly_distribute_plot_positions(idx, n_xlabel, n_data, rel_space_btw_entries=0)

        ax.bar(positions, d, widths, label=legend_labels[idx], color=colors[idx])

    ax.set_xticks(np.arange(n_xlabel))
    ax.set_xticklabels(x_tick_labels)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if legend:
        ax.legend()

    if title:
        ax.set_title(title)


def hist_from_bin_values(ax: plt.Axes, bins, hist, xlabel=None, use_percentages=False, use_log=False):
    widths = bins[1:] - bins[:-1]

    if use_percentages:
        hist = 100 * hist / np.sum(hist)
        ax.set_ylabel("Occurrence [%]")
    else:
        ax.set_ylabel("Absolut occurrence")

    ax.bar(bins[:-1], hist, width=widths)

    if use_log:
        ax.set_xscale('log')

    if xlabel is not None:
        ax.set_xlabel(xlabel)


def boxplot_compare(ax: plt.Axes, x_tick_labels, data, legend_labels, colors=None, legend=True, ylabel=None,
                    title=None, showfliers=True):
    if colors is None:
        colors = DEFAULT_COLORS

    n_data = len(data)
    n_xlabel = len(x_tick_labels)
    leg_handles = []
    leg_labels = []
    bps = []

    for idx, d in enumerate(data):
        positions, widths = evenly_distribute_plot_positions(idx, n_xlabel, n_data)
        props = {
            'color': colors[idx],
            'linestyle': '-',
            'lw': 2
        }
        flier_props = {
            'markeredgecolor': colors[idx],
        }
        box_props = {
            'facecolor': mcolors.to_rgba(colors[idx], 0.3),
            'edgecolor': colors[idx],
            'linestyle': '-',
            'lw': 2
        }
        if isinstance(d[0], dict):
            bp = ax.bxp(d, positions=positions, widths=widths, patch_artist=True, capprops=props, meanprops=props,
                        boxprops=box_props, flierprops=flier_props, medianprops=props, whiskerprops=props,
                        showfliers=showfliers)
        else:
            bp = ax.boxplot(d, positions=positions, widths=widths, patch_artist=True, capprops=props, meanprops=props,
                            boxprops=box_props, flierprops=flier_props, medianprops=props, whiskerprops=props,
                            showfliers=showfliers)
        bps.append(bp)
        tmp, = plt.plot([1, 1], c=colors[idx], alpha=0)
        leg_handles.append(tmp)
        leg_labels.append(legend_labels[idx])

    ax.set_xticks(np.arange(n_xlabel))
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlim(-.6, n_xlabel-.4)
    if legend:
        # ax.legend(leg_handles, leg_labels, bbox_to_anchor=(
            # 1.05, 1), loc=2, borderaxespad=0.)
        # ax.legend(leg_handles, leg_labels)
        ax.legend([element["boxes"][0] for element in bps],
                  [legend_labels[idx] for idx, _ in enumerate(data)])

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    # map(lambda x: x.set_visible(False), leg_handles)


def draw_lines_on_top_of_comparison_plots(ax: plt.Axes, data, num_comparisons):
    n_xlabel = len(data)
    n_data = num_comparisons
    positions_start, widths_start = evenly_distribute_plot_positions(0, n_xlabel, n_data)
    positions_end, widths_end = evenly_distribute_plot_positions(num_comparisons - 1, n_xlabel, n_data)
    for idx, d in enumerate(data):
        x_start = positions_start[idx] - widths_start[idx]/2
        x_end = positions_end[idx] + widths_end[idx]/2

        if len(d) == 1:
            val = d[0]
            d = [val, val]

        x = np.linspace(x_start, x_end, len(d))
        ax.plot(x, d, color="black", linestyle="--", linewidth=1)


def evenly_distribute_plot_positions(idx, num_slots, num_entries, space_btw_groups=0.5, rel_space_btw_entries=0.15):
    width = (1 - space_btw_groups) / (num_entries + (num_entries - 1) * rel_space_btw_entries)
    # width = 1 / (1.5 * num_entries + 1.5)
    widths = [width] * num_slots

    positions = [pos - 0.5 + (space_btw_groups + width) / 2 + width*(1+rel_space_btw_entries)*idx
                 for pos in np.arange(num_slots)]

    return positions, widths


# https://stackoverflow.com/a/41259922
def align_yaxis(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = np.array([ax1, ax2])
    extrema = np.array([ax.get_ylim() for ax in axes])
    tops = extrema[:, 1] / (extrema[:, 1] - extrema[:, 0])
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [a[::-1] for a in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    extrema[0, 1] = extrema[0, 0] + tot_span * (extrema[0, 1] - extrema[0, 0])
    extrema[1, 0] = extrema[1, 1] + tot_span * (extrema[1, 0] - extrema[1, 1])
    [axes[i].set_ylim(*extrema[i]) for i in range(2)]


def plot_moving_boxplot_in_time(pc: PlotContext, t, data, title=None, ylabel=None, color=None, t_resolution=0.1,
                                data_filter=None):
    t_quantized, stats = get_quantized_statistics_along_axis(t, data, data_filter, t_resolution)
    plot_moving_boxplot_in_time_from_stats(pc, t_quantized, stats, title, ylabel, color)


def plot_moving_boxplot_in_time_from_stats(pc: PlotContext, t, stats, title=None, ylabel=None, color=None, xlabel=None):
    if not color:
        color = DEFAULT_COLORS[0]
    ax = pc.get_axis()

    ax.plot(t, stats['median'], color=color)

    if ylabel:
        ax.set_ylabel(ylabel)
    if not xlabel:
        xlabel = "time [s]"
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)

    # 50% range
    ax.fill_between(t, stats['q25'], stats['q75'], alpha=0.5, lw=0, facecolor=color)

    # 90% range
    ax.fill_between(t, stats['q05'], stats['q25'], alpha=0.25, lw=0, facecolor=color)
    ax.fill_between(t, stats['q75'], stats['q95'], alpha=0.25, lw=0, facecolor=color)

    # MIN-MAX
    ax.fill_between(t, stats['min'], stats['q25'], alpha=0.1, lw=0, facecolor=color)
    ax.fill_between(t, stats['q75'], stats['max'], alpha=0.1, lw=0, facecolor=color)
