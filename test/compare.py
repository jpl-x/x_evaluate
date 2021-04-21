import argparse
import os

from evo.core import metrics
from x_evaluate.evaluation_data import PlotType, ErrorType
from x_evaluate.utils import read_evaluation_pickle

import x_evaluate.tracking_evaluation as fe
import x_evaluate.performance_evaluation as pe
import x_evaluate.trajectory_evaluation as te


def main():
    parser = argparse.ArgumentParser(description='Comparison script for dealing with evaluation.pickle files')
    parser.add_argument('--input_folder', type=str, required=True)

    args = parser.parse_args()

    print(args.input_folder)

    # no_undistortion = read_evaluation_pickle(args.input_folder, "evaluation_no_undistortion.pickle")
    # new_undistortion = read_evaluation_pickle(args.input_folder, "evaluation_rt_fixed.pickle")
    eklt = read_evaluation_pickle(args.input_folder, "evaluation.pickle")
    # rt = read_evaluation_pickle(args.input_folder, "evaluation_radial_tangential.pickle")
    # rt_pub = read_evaluation_pickle(args.input_folder, "evaluation_radial_tangential_published.pickle")
    # # plot_ape_comparison(summary.data.values(), filename, r)
    # data = [no_undistortion.data['Boxes 6DOF'], rt.data['Boxes 6DOF'], new_undistortion.data['Boxes 6DOF'],
    #         eklt.data['Boxes 6DOF EKLT']]
    # labels = ["No undistortion", "Radial-tangential", "Radial-tangential FIXED", "EKLT"]
    # plot_error_comparison(data, None, metrics.PoseRelation.full_transformation,
    #                       ErrorType.APE, PlotType.TIME_SERIES, labels)
    eklt.data.pop("Boxes 6DOF")
    eklt.data.pop("Boxes 6DOF EKLT")
    eklt.data.pop("HDR Poster")
    eklt.data.pop("HDR Poster EKLT")

    # fe.plot_feature_plots(eklt.data['Boxes 6DOF EKLT'], os.path.join(args.input_folder, "001_boxes_6dof_eklt"))
    # fe.plot_summary_plots(eklt, args.input_folder)
    te.plot_error_comparison(eklt.data.values(), None, metrics.PoseRelation.full_transformation, ErrorType.APE,
                             PlotType.BOXPLOT)


if __name__ == '__main__':
    main()
