import os
import pickle

from evo.core.trajectory import PoseTrajectory3D
from matplotlib import pyplot as plt

from x_evaluate.evaluation_data import EvaluationDataSummary


def convert_to_evo_trajectory(df_poses, prefix="") -> PoseTrajectory3D:
    xyz_est = df_poses[[prefix + 'p_x', prefix + 'p_y', prefix + 'p_z']].to_numpy()
    wxyz_est = df_poses[[prefix + 'q_w', prefix + 'q_x', prefix + 'q_y', prefix + 'q_z']].to_numpy()
    return PoseTrajectory3D(xyz_est, wxyz_est, df_poses[['t']].to_numpy())


def boxplot(filename, data, labels, title=""):
    plt.figure()
    # WHIS explanation see https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.boxplot.html, it's worth it
    plt.boxplot(data, vert=True, labels=labels, whis=1.5)
    plt.title(title)

    if filename is None or len(filename) == 0:
        plt.show()
    else:
        plt.savefig(filename)
    plt.clf()


def time_series_plot(filename, t_arrays, data, labels, title=""):
    plt.figure()
    for i in range(len(t_arrays)):
        plt.plot(t_arrays[i], data[i], label=labels[i])

    plt.legend()
    plt.title(title)
    plt.xlabel("Time [s]")

    if filename is None or len(filename) == 0:
        plt.show()
    else:
        plt.savefig(filename)
    plt.clf()


def read_evaluation_pickle(input_folder, filename) -> EvaluationDataSummary:
    file = os.path.join(input_folder, filename)
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data