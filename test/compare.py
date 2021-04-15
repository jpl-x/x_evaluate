import argparse

from evo.core import metrics

from x_evaluate.trajectory_evaluation import plot_ape_comparison, plot_rpe_comparison
from x_evaluate.utils import read_evaluation_pickle


def main():
    parser = argparse.ArgumentParser(description='Comparison script for dealing with evaluation.pickle files')
    parser.add_argument('--input_folder', type=str, required=True)

    args = parser.parse_args()

    print(args.input_folder)

    no_undistortion = read_evaluation_pickle(args.input_folder, "evaluation_no_undistortion.pickle")
    rt = read_evaluation_pickle(args.input_folder, "evaluation_radial_tangential.pickle")
    rt_pub = read_evaluation_pickle(args.input_folder, "evaluation_radial_tangential_published.pickle")
    # plot_ape_comparison(summary.data.values(), filename, r)
    data = [no_undistortion.data['Boxes 6DOF'], rt.data['Boxes 6DOF'], rt_pub.data['Boxes 6DOF']]
    labels = ["No undistortion", "Radial-tangential calib file", "Radial-tangential camera_info topic"]
    plot_ape_comparison(data[:-1], None, metrics.PoseRelation.full_transformation, labels[:-1])
    plot_rpe_comparison(data[:-1], None, metrics.PoseRelation.full_transformation, labels[:-1])


if __name__ == '__main__':
    main()
