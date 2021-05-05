from enum import Enum

from matplotlib import pyplot as plt


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


def time_series_plot(filename, time, data, labels, title="", ylabel=None):
    with PlotContext(filename) as f:
        ax = f.get_axis()

        for i in range(len(data)):

            # this causes issues, quick fix:
            label = labels[i]
            if label.startswith('_'):
                label = label[1:]

            if isinstance(time, list):
                plt.plot(time[i], data[i], label=label)
            else:
                plt.plot(time, data[i], label=label)

        plt.legend()
        plt.title(title)
        plt.xlabel("Time [s]")

        if ylabel is not None:
            plt.ylabel(ylabel)

