import argparse
import os

import numpy as np
import tqdm

from x_evaluate.scriptlets import read_evaluation_pickle, write_evaluation_pickle
from x_evaluate.trajectory_evaluation import calculate_rpe_errors_for_pairs_at_different_distances


def main():
    parser = argparse.ArgumentParser(description='Reads evaluation.pickle and re-calculates RPE error arrays for new '
                                                 'pose pair distances')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--distances', nargs='+', required=True, type=float)
    parser.add_argument('--force_recalculations', default=False, action='store_true')

    args = parser.parse_args()

    output_root = os.path.dirname(args.input)
    filename = os.path.basename(args.input)

    print(F"Reading {args.input}")
    s = read_evaluation_pickle(output_root, filename)

    for k, e in tqdm.tqdm(s.data.items()):
        rpe_error_t = e.trajectory_data.rpe_error_t
        rpe_error_r = e.trajectory_data.rpe_error_r

        # print(F"For dataset '{k}', the following RPE pair distances are available:")
        # print(rpe_error_r.keys())
        # print(rpe_error_t.keys())

        to_calculate = [d for d in args.distances if d not in rpe_error_t or d not in rpe_error_r or
                        args.force_recalculations]

        # print(F"Calculating {to_calculate}")

        error_t, error_r = calculate_rpe_errors_for_pairs_at_different_distances(to_calculate,
                                                                                 e.trajectory_data.traj_gt_synced,
                                                                                 e.trajectory_data.traj_est_aligned)

        rpe_error_t.update(error_t)
        rpe_error_r.update(error_r)

    print("Writing evaluation pickle")
    write_evaluation_pickle(s, output_root, filename)


if __name__ == '__main__':
    main()
