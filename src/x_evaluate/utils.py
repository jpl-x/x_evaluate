import os
import pickle
from typing import Dict

import git
import numpy as np
import pandas as pd
from envyaml import EnvYAML
from evo.core.trajectory import PoseTrajectory3D

from x_evaluate.evaluation_data import EvaluationDataSummary, GitInfo


def convert_to_evo_trajectory(df_poses, prefix="") -> PoseTrajectory3D:
    xyz_est = df_poses[[prefix + 'p_x', prefix + 'p_y', prefix + 'p_z']].to_numpy()
    wxyz_est = df_poses[[prefix + 'q_w', prefix + 'q_x', prefix + 'q_y', prefix + 'q_z']].to_numpy()
    return PoseTrajectory3D(xyz_est, wxyz_est, df_poses[['t']].to_numpy())


def read_evaluation_pickle(file) -> EvaluationDataSummary:
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


def timestamp_to_rosbag_time(timestamps, df_rt):
    return np.interp(timestamps, df_rt['ts_real'], df_rt['t_sim'])


def timestamp_to_rosbag_time_zero(timestamps, df_rt):
    return timestamp_to_rosbag_time(timestamps, df_rt) - df_rt['t_sim'][0]


def timestamp_to_real_time(timestamps, df_rt):
    return np.interp(timestamps, df_rt['ts_real'], df_rt['t_real'])


def envyaml_to_archive_dict(eval_config: EnvYAML) -> Dict:
    conf = eval_config.export()
    # remove environment keys to protect privacy
    for k in os.environ.keys():
        conf.pop(k)
    return conf


def get_git_info(path) -> GitInfo:
    x = git.Repo(path)
    return GitInfo(branch=x.active_branch.name,
                   last_commit=x.head.object.hexsha,
                   files_changed=len(x.index.diff(None)) > 0)


def name_to_identifier(name):
    return name.lower().replace(' ', '_')


def convert_to_tracks_txt(df_tracks: pd.DataFrame, out_file):
    tracks = df_tracks.loc[df_tracks.update_type != 'Lost', ['id', 'patch_t_current', 'center_x', 'center_y']]
    np.savetxt(out_file, tracks)
    return df_tracks


def read_output_files(output_folder, gt_available):
    df_poses = pd.read_csv(os.path.join(output_folder, "pose.csv"), delimiter=";")
    df_features = pd.read_csv(os.path.join(output_folder, "features.csv"), delimiter=";")
    df_resources = pd.read_csv(os.path.join(output_folder, "resource.csv"), delimiter=";")
    df_groundtruth = None
    if gt_available:
        df_groundtruth = pd.read_csv(os.path.join(output_folder, "gt.csv"), delimiter=";")
    df_realtime = pd.read_csv(os.path.join(output_folder, "realtime.csv"), delimiter=";")

    # profiling_json = read_json_file(output_folder)
    return df_groundtruth, df_poses, df_realtime, df_features, df_resources


def read_eklt_output_files(output_folder):
    df_events = pd.read_csv(os.path.join(output_folder, "events.csv"), delimiter=";")
    df_optimizations = pd.read_csv(os.path.join(output_folder, "optimizations.csv"), delimiter=";")
    df_tracks = pd.read_csv(os.path.join(output_folder, "tracks.csv"), delimiter=";")
    return df_events, df_optimizations, df_tracks
