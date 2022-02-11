#!/usr/bin/env python
import errno
import sys
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(
        prog='qualitative_analysis.py',
        description='Checks if parameter runs are sucessfull or if they are diverging.')
    parser.add_argument('--output_dir', type=str, help='output directory',
                        default=None, metavar="output_dir")
    parser.add_argument('input_dir', type=str, help='input directory')
    args = parser.parse_args()
    return args

def get_sub_dirs(input_dir):

    directories = os.listdir(input_dir)
    directories.sort()

    eval_dirs = []

    for directory in directories:
        if len(directory) > 3:
            test_str = directory[0:3]
            if test_str.isnumeric():
                eval_dirs.append(directory)

    return eval_dirs

def highlight_if_true(df, color = "green"):

    attr = 'background-color: {}'.format(color)
    df_bool = pd.DataFrame(df.apply(lambda x: [True if v == True else False for v in x],axis=1).apply(pd.Series),
                      index=df.index)
    df_bool.columns =df.columns
    return pd.DataFrame(np.where(df_bool, attr, ""),
                       index= df.index, columns=df.columns)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = parse_args()

    work_dir = os.getcwd()

    run_name = args.input_dir.split('/')[-1]

    colors = list(mcolors.TABLEAU_COLORS)

    if args.output_dir is None:
        args.output_dir = work_dir + "/" + run_name
        try:
            os.makedirs(args.output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    else:
        if not os.path.isdir(args.input_dir):
            print("Output directory not found! (" + args.output_dir + ")")
            exit(1)


    if not os.path.isdir(args.input_dir):
        print("Input directory not found! (" + args.input_dir + ")")
        exit(1)

    eval_dirs = get_sub_dirs(args.input_dir)

    all_pose_files = []
    dataset_dirs = get_sub_dirs(args.input_dir + "/" + eval_dirs[0])

    for directory in eval_dirs:
        pose_files = []
        current_dir = args.input_dir + "/" + directory
        for dataset_dir in dataset_dirs:
            pose_file = current_dir + "/" + dataset_dir + "/pose.csv"
            if os.path.isfile(pose_file):
                #print(pose_file)
                pose_files.append(pd.read_csv(pose_file, header=0, delimiter=";"))
            else:
                pose_files.append(pd.DataFrame())
        all_pose_files.append(pose_files)

    t_max = 10
    x_max = 30
    y_max = 30
    z_max = 3
    limit = 2

    ## Make xyz-t-plot
    for i, dataset in enumerate(dataset_dirs):
        plt.figure(figsize=(9, 3.2))
        plt.subplot(131)
        for j, experiment in enumerate(all_pose_files):
            try:
                t = experiment[i]['t'].to_numpy()
                p_x = experiment[i]['estimated_p_x'].to_numpy()
                p_x = p_x[t != -1]
                t = t[t != -1]
                plt.plot(t-t[0], p_x, color=colors[j])
                plt.xlabel('t')
                plt.ylabel('x')
                plt.xlim(0, t_max)
                plt.ylim(-x_max, x_max)
            except:
                pass
        plt.subplot(132)
        for j, experiment in enumerate(all_pose_files):
            try:
                t = experiment[i]['t'].to_numpy()
                p_y = experiment[i]['estimated_p_y'].to_numpy()
                p_y = p_y[t != -1]
                t = t[t != -1]
                plt.plot(t-t[0], p_y, color=colors[j])
                plt.xlabel('t')
                plt.ylabel('y')
                plt.xlim(0, t_max)
                plt.ylim(-y_max, y_max)
            except:
                pass
        plt.subplot(133)
        for j, experiment in enumerate(all_pose_files):
            try:
                t = experiment[i]['t'].to_numpy()
                p_z = experiment[i]['estimated_p_z'].to_numpy()
                p_z = p_z[t != -1]
                t = t[t != -1]
                plt.plot(t-t[0], p_z, color=colors[j])
                plt.legend(eval_dirs)
                plt.xlabel('t')
                plt.ylabel('z')
                plt.xlim(0, t_max)
                plt.ylim(-z_max, z_max)
            except:
                pass
        plt.suptitle('position over time')
        plt.tight_layout()
        plt.savefig(args.output_dir + '/' + dataset + '_xyz_t.svg', format='svg', dpi=1200)

        ## Make xyz-t-plot
        for i, dataset in enumerate(dataset_dirs):
            plt.figure(figsize=(9, 3.2))
            plt.subplot(131)
            for j, experiment in enumerate(all_pose_files):
                try:
                    t = experiment[i]['t'].to_numpy()
                    p_x = experiment[i]['estimated_p_x'].to_numpy()
                    p_y = experiment[i]['estimated_p_y'].to_numpy()
                    p_x = p_x[t != -1]
                    p_y = p_y[t != -1]
                    t = t[t != -1]
                    plt.plot(p_x, p_y, color=colors[j])
                    plt.xlim(-limit, limit)
                    plt.ylim(-limit, limit)
                    plt.xlabel('x')
                    plt.ylabel('y')
                except:
                    pass
            plt.subplot(132)
            for j, experiment in enumerate(all_pose_files):
                try:
                    t = experiment[i]['t'].to_numpy()
                    p_y = experiment[i]['estimated_p_y'].to_numpy()
                    p_z = experiment[i]['estimated_p_z'].to_numpy()
                    p_y = p_y[t != -1]
                    p_z = p_z[t != -1]
                    t = t[t != -1]
                    plt.plot(p_y, p_z, color=colors[j])
                    plt.xlabel('y')
                    plt.ylabel('z')
                except:
                    pass
            plt.subplot(133)
            for j, experiment in enumerate(all_pose_files):
                try:
                    try:
                        t = experiment[i]['t'].to_numpy()
                        p_x = experiment[i]['estimated_p_x'].to_numpy()
                        p_z = experiment[i]['estimated_p_z'].to_numpy()
                        p_x = p_x[t != -1]
                        p_z = p_z[t != -1]
                        t = t[t != -1]
                        plt.plot(p_x, p_z, color=colors[j])
                        plt.xlabel('x')
                        plt.ylabel('z')
                    except:
                        pass
                    plt.legend(eval_dirs)
                    #plt.legend(["EKLT-VIO", "KLT-VIO"])
                except:
                    pass
            plt.suptitle('position over time')
            plt.tight_layout()
            plt.savefig(args.output_dir + '/' + dataset + '_position.svg', format='svg', dpi=1200)
            plt.close()

    ## Make xyz-t-plot
    for i, dataset in enumerate(dataset_dirs):
        plt.figure(figsize=(4, 3))
        for j, experiment in enumerate(all_pose_files):
            try:
                t = experiment[i]['t'].to_numpy()
                p_x = experiment[i]['estimated_p_x'].to_numpy()
                p_y = experiment[i]['estimated_p_y'].to_numpy()
                p_z = experiment[i]['estimated_p_y'].to_numpy()
                dist = np.sqrt(p_x * p_x + p_y * p_y + p_z * p_z)
                p_x = p_x[t != -1]
                p_y = p_y[t != -1]
                p_z = p_z[t != -1]
                dist = dist[t != -1]
                t = t[t != -1]
                plt.plot(t-t[0], dist, color=colors[j])
                plt.legend(eval_dirs)
                plt.xlabel('Time [s]')
                plt.ylabel('Distance [m]')
                plt.grid(True)
                plt.tight_layout()
            except:
                pass
        plt.savefig(args.output_dir + '/' + dataset + '_distance.svg', format='svg', dpi=1200)
        plt.close()

    #success = np.zeros((len(dataset_dirs), len(all_pose_files)))
    dist_max = np.inf * np.ones((len(dataset_dirs), len(all_pose_files)))
    max_dist = 42.0

    for i, dataset in enumerate(dataset_dirs):
        for j, experiment in enumerate(all_pose_files):
            try:
                t = experiment[i]['t'].to_numpy()
                p_x = experiment[i]['estimated_p_x'].to_numpy()
                p_y = experiment[i]['estimated_p_y'].to_numpy()
                p_z = experiment[i]['estimated_p_z'].to_numpy()
                p_x = p_x[t != -1]
                p_y = p_y[t != -1]
                p_z = p_z[t != -1]
                t = t[t != -1]
                max_xx = np.max(p_x**2)
                max_yy = np.max(p_y**2)
                max_zz = np.max(p_z**2)
                dist_max[i, j] = np.sqrt(max_xx + max_yy + max_zz)
            except:
                dist_max[i, j] = np.inf
    #print(pd.DataFrame(dist_max, dataset_dirs, eval_dirs))
    success = dist_max < max_dist
    results = pd.DataFrame(success, dataset_dirs, eval_dirs)
    results.style. \
        apply(highlight_if_true, axis=None). \
        to_excel(args.output_dir + '/' + 'results.xlsx', engine="openpyxl")

    victory = False
    for column in results:
        victory = all(results[column] == True)
        if victory:
            print("Run " + column + " was successful on all datasets!!")