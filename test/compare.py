import argparse

from evo.core import metrics

from x_evaluate.evaluation_data import PlotType, ErrorType
from x_evaluate.trajectory_evaluation import plot_error_comparison
from x_evaluate.utils import read_evaluation_pickle


def main():
    parser = argparse.ArgumentParser(description='Comparison script for dealing with evaluation.pickle files')
    parser.add_argument('--input_folder', type=str, required=True)

    args = parser.parse_args()

    print(args.input_folder)

    no_undistortion = read_evaluation_pickle(args.input_folder, "evaluation_no_undistortion.pickle")
    new_undistortion = read_evaluation_pickle(args.input_folder, "evaluation_rt_fixed.pickle")
    eklt = read_evaluation_pickle(args.input_folder, "evaluation.pickle")
    rt = read_evaluation_pickle(args.input_folder, "evaluation_radial_tangential.pickle")
    rt_pub = read_evaluation_pickle(args.input_folder, "evaluation_radial_tangential_published.pickle")
    # plot_ape_comparison(summary.data.values(), filename, r)
    data = [no_undistortion.data['Boxes 6DOF'], rt.data['Boxes 6DOF'], new_undistortion.data['Boxes 6DOF'],
            eklt.data['Boxes 6DOF EKLT']]
    labels = ["No undistortion", "Radial-tangential", "Radial-tangential FIXED", "EKLT"]
    plot_error_comparison(data, None, metrics.PoseRelation.full_transformation,
                          ErrorType.APE, PlotType.TIME_SERIES, labels)


if __name__ == '__main__':
    main()
