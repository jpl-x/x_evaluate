from enum import Enum
from typing import List
import numpy as np
from matplotlib import pyplot as plt
from x_evaluate.evaluation_data import DistributionSummary

import matplotlib.colors as mcolors


class PlotType(Enum):
    BOXPLOT = 1
    TIME_SERIES = 2


class PlotContext:
    figure: plt.Figure
    axis: List[plt.Axes]

    def __init__(self, filename=None, subplot_rows=1, subplot_cols=1, base_width_inch=8, base_height_inch=6):
        self.filename = filename
        self.subplot_rows = subplot_rows
        self.subplot_cols = subplot_cols
        self.width_inch = base_width_inch * subplot_cols
        self.height_inch = base_height_inch * subplot_rows
        self.axis = []

    def __enter__(self):
        self.figure = plt.figure()
        self.subplot_idx = 0
        self.figure.set_size_inches(self.width_inch, self.height_inch)
        return self

    def get_axis(self, **kwargs) -> plt.Axes:
        self.subplot_idx += 1
        ax = self.figure.add_subplot(self.subplot_rows, self.subplot_cols, self.subplot_idx, **kwargs)
        self.axis.append(ax)
        return ax

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.figure.tight_layout()
        if self.filename is None:
            self.figure.show()
        else:
            self.figure.savefig(self.filename)

        for a in self.axis:
            a.set_xscale('linear')  # workaround for https://github.com/matplotlib/matplotlib/issues/9970
            a.set_yscale('linear')  # workaround for https://github.com/matplotlib/matplotlib/issues/9970

        self.figure.clf()
        plt.close(self.figure)


def boxplot(pc: PlotContext, data, labels, title="", outlier_params=1.5):
    ax = pc.get_axis()
    ax.boxplot(data, vert=True, labels=labels, whis=outlier_params)
    ax.set_title(title)


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


def time_series_plot(pc: PlotContext, time, data, labels, title="", ylabel=None):
    ax = pc.get_axis()
    for i in range(len(data)):

        # this causes issues, quick fix:
        label = labels[i]
        if label.startswith('_'):
            label = label[1:]

        if isinstance(time, list):
            ax.plot(time[i], data[i], label=label)
        else:
            ax.plot(time, data[i], label=label)

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Time [s]")

    if ylabel is not None:
        ax.set_ylabel(ylabel)


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
        colors = list(mcolors.TABLEAU_COLORS.values())

    n_data = len(data)
    n_xlabel = len(x_tick_labels)
    w = 1 / (1.5 * n_data + 1.5)

    for idx, d in enumerate(data):
        positions = [pos - 0.5 + 1.5 * w + idx * w
                     for pos in np.arange(n_xlabel)]

        ax.bar(positions, d, w, label=legend_labels[idx], color=colors[idx])

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
                    title=None):
    if colors is None:
        colors = list(mcolors.TABLEAU_COLORS.values())

    n_data = len(data)
    n_xlabel = len(x_tick_labels)
    leg_handles = []
    leg_labels = []
    bps = []
    w = 1 / (1.5 * n_data + 1.5)
    widths = [w] * n_xlabel
    for idx, d in enumerate(data):
        positions = [pos - 0.5 + 1.5 * w + idx * w
                     for pos in np.arange(n_xlabel)]
        props = {
            'facecolor': colors[idx]
        }

        if isinstance(d[0], dict):
            bp = ax.bxp(d, positions=positions, widths=widths, patch_artist=True, boxprops=props)
        else:
            bp = ax.boxplot(d, positions=positions, widths=widths, patch_artist=True, boxprops=props)
        # boxprops=dict(
        # facecolor=colors[idx]))
        color_box(bp, colors[idx])
        bps.append(bp)
        tmp, = plt.plot([1, 1], c=colors[idx], alpha=0)
        leg_handles.append(tmp)
        leg_labels.append(legend_labels[idx])

    ax.set_xticks(np.arange(n_xlabel))
    ax.set_xticklabels(x_tick_labels)
    # xlims = ax.get_xlim()
    # ax.set_xlim([xlims[0]-0.1, xlims[1]-0.1])
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
