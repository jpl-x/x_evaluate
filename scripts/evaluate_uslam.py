import argparse
import os
import shutil

import pandas as pd

import roslaunch

from x_evaluate.evaluation_data import EvaluationData, EvaluationDataSummary
from x_evaluate.scriptlets import write_evaluation_pickle
from x_evaluate.trajectory_evaluation import evaluate_trajectory, plot_trajectory_plots, plot_summary_plots, \
    create_summary_info


def main():
    parser = argparse.ArgumentParser(description='Hand-made script for getting all result tables')
    parser.add_argument('--uslam_folder', type=str, required=True)
    parser.add_argument('--catkin_root', type=str, required=True)
    parser.add_argument('--cfg_filename', type=str, default=None)
    parser.add_argument('--calib_filename', type=str, default="DAVIS-IJRR17.yaml")
    parser.add_argument('--name', type=str, default="Debug")

    args = parser.parse_args()

    # datasets = ["Boxes 6DOF", "Boxes Translation", "Dynamic 6DOF", "Dynamic Translation", "HDR Boxes",
    #                  "HDR Poster", "Poster 6DOF", "Poster Translation", "Shapes 6DOF", "Shapes Translation"]
    #
    # must_datasets = [x.lower().replace(' ', '_') for x in datasets]
    # print(must_datasets)

    datasets = []

    datasets = [
        # "Mars Straight Vmax 3.2 Offset 2.5",
                "Mars Circle Vmax 7.2 Offset 2.5",
                "Mars Circle Vmax 7.2 Offset 2.5 no bootsrapping",
                # "Mars Circle Vmax 7.2 Offset 5.0",
                # "Mars Circle Vmax 7.2 Offset 10.0",
                "Mars Vertical Circle Vmax 2.4 Offset 2.5",
                "Mars Vertical Circle Vmax 2.4 Offset 2.5 no bootstrapping",
                "Mars Mellon Vmax 12.4 Offset 10",
                "Mars Mellon Vmax 12.4 Offset 10 no bootsrapping",
                # "Mars Circle Vmax 7.2 Offset 10", "Mars Circle Vmax 16.6 Offset 10"
    ]



    must_datasets = [
        # "neuro_bem_esim_straight_vmax_3.2_offset_2.5",
                     "neuro_bem_esim_circle_vmax_7.2_offset_2.5",
                     "neuro_bem_esim_circle_vmax_7.2_offset_2.5_no_bootstrapping",
                     # "neuro_bem_esim_circle_vmax_7.2_offset_5",
                     # "neuro_bem_esim_circle_vmax_7.2_offset_10",
                     "neuro_bem_esim_vcircle_vmax_2.4_offset_2.5",
                     "neuro_bem_esim_vcircle_vmax_2.4_offset_2.5_no_bootstrapping",
        #              "neuro_bem_esim_circle_vmax_7.2_offset_10_no_bootstrapping",
        #                 "neuro_bem_esim_circle_vmax_16.6_offset_10_no_bootstrapping",
                        "neuro_bem_esim_mellon_vmax_12.4_offset_10",
                        "neuro_bem_esim_mellon_vmax_12.4_offset_10_no_bootstrapping",
    ]

    # output_folders = [os.path.join(args.uslam_folder, "run_ijrr_" + x) for x in must_datasets]

    output_folders = [os.path.join(args.uslam_folder, "run_" + args.name.lower().replace(' ', '_'))]

    for output_folder in output_folders:

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # for d in must_datasets:
        #     print()

        s = EvaluationDataSummary()
        s.name = os.path.basename(output_folder)

        for i, d in enumerate(must_datasets):
            sequence_folder = os.path.join(output_folder, F"{i+1:>03}_{d}")

            if not os.path.exists(sequence_folder):
                os.makedirs(sequence_folder)

            # if i >= 1:
            #     break

            bag_filename = F"rpg_davis_data/{d}.bag"
            bag_filename = F"sim/{d}.bag"
            cfg_filename = F"{d}.conf"

            if args.cfg_filename:
                cfg_filename = args.cfg_filename

            run_uslam(bag_filename, args.catkin_root, cfg_filename, args.calib_filename, sequence_folder)

            gt_file = os.path.join(args.uslam_folder, F"gt/{d}/gt.csv")
            output_file = "/tmp/uslam_pose.csv"

            # shutil.copy(output_file, os.path.join(sequence_folder, "pose.csv"))  # COMMENT ME OUT
            # continue

            # output_file = os.path.join(sequence_folder, "pose.csv")

            uslam_traj_es = pd.read_csv(os.path.join(sequence_folder, "traj_es.csv"))
            uslam_traj_es['timestamp'] /= 1e9
            uslam_traj_es['update_modality'] = "USLAM OUTPUT"
            mapping = {'timestamp': 't', ' x': 'estimated_p_x', ' y': 'estimated_p_y', ' z': 'estimated_p_z',
                       ' qx': 'estimated_q_x', ' qy': 'estimated_q_y', ' qz': 'estimated_q_z', ' qw': 'estimated_q_w'}
            uslam_traj_es = uslam_traj_es.rename(columns=mapping)
            uslam_traj_es = uslam_traj_es.drop([' vx', ' vy', ' vz', ' bgx', ' bgy', ' bgz', ' bax', ' bay', ' baz'],
                                               axis=1)

            # df_poses = pd.read_csv(output_file, delimiter=";")
            df_poses = uslam_traj_es
            df_groundtruth = pd.read_csv(gt_file, delimiter=";")

            d = EvaluationData()
            d.name = must_datasets[i]

            print("Analyzing output trajectory...")

            d.trajectory_data = evaluate_trajectory(df_poses, df_groundtruth)

            print("Plotting trajectory plots...")

            plot_trajectory_plots(d.trajectory_data, "USLAM", sequence_folder)

            s.data[datasets[i]] = d

        plot_summary_plots(s, output_folder)
        create_summary_info(s, output_folder)

        write_evaluation_pickle(s, output_folder)


def run_uslam(bag_filename, catkin_root, cfg_filename, calib_filename, log_dir):
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    launch_file = "src/rpg_ultimate_slam_pro/applications/ze_vio_ceres/launch/xvio_datasets.launch"
    cli_args = [os.path.join(catkin_root, launch_file), F"bag_filename:={bag_filename}",
                F"cfg_filename:={cfg_filename}", F'calib_filename:={calib_filename}', F"log_dir:={log_dir}"]
    roslaunch_args = cli_args[1:]
    roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
    parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
    parent.start()
    parent.spin()


if __name__ == '__main__':
    main()
