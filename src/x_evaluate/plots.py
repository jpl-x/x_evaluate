from enum import Enum
from typing import List

from matplotlib import pyplot as plt

from x_evaluate.evaluation_data import DistributionSummary


class PlotType(Enum):
    BOXPLOT = 1
    TIME_SERIES = 2


class PlotContext:
    figure: plt.Figure

    def __init__(self, filename=None, subplot_rows=1, subplot_cols=1, width_inch=10, height_inch=7):
        self.filename = filename
        self.subplot_rows = subplot_rows
        self.subplot_cols = subplot_cols
        self.width_inch = width_inch
        self.height_inch = height_inch

    def __enter__(self):
        self.figure = plt.figure()
        self.subplot_idx = 0
        self.figure.set_size_inches(self.width_inch, self.height_inch)
        return self

    def get_axis(self) -> plt.Axes:
        self.subplot_idx += 1
        return self.figure.add_subplot(self.subplot_rows, self.subplot_cols, self.subplot_idx)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.filename is None:
            self.figure.show()
        else:
            self.figure.savefig(self.filename)
        self.figure.clf()
        plt.close(self.figure)


def boxplot(filename, data, labels, title="", outlier_params=1.5):
    with PlotContext(filename) as f:
        ax = f.get_axis()
        ax.boxplot(data, vert=True, labels=labels, whis=outlier_params)
        ax.set_title(title)


def summary_to_dict(distribution_summary: DistributionSummary, label):
    return {
        'label': label,
        'whislo': distribution_summary.min,  # Bottom whisker position
        'q1': distribution_summary.quantiles[0.25],  # First quartile (25th percentile)
        'med': distribution_summary.quantiles[0.5],  # Median         (50th percentile)
        'q3': distribution_summary.quantiles[0.75],  # Third quartile (75th percentile)
        'whishi': distribution_summary.max,  # Top whisker position
        'fliers': []  # Outliers
    }


def boxplot_from_summary(filename, distribution_summaries: List[DistributionSummary], labels, title=""):
    with PlotContext(filename) as f:
        ax = f.get_axis()
        boxes = []

        for i in range(len(distribution_summaries)):
            boxes.append(summary_to_dict(distribution_summaries[i], labels[i]))
        ax.bxp(boxes, showfliers=False)
        ax.set_title(title)


def time_series_plot(filename, time, data, labels, title="", ylabel=None):
    with PlotContext(filename) as f:
        ax = f.get_axis()

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
