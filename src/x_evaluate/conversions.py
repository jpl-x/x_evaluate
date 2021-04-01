from evo.core.trajectory import PoseTrajectory3D


def convert_to_evo_trajectory(df_poses, prefix="") -> PoseTrajectory3D:
    xyz_est = df_poses[[prefix + 'p_x', prefix + 'p_y', prefix + 'p_z']].to_numpy()
    wxyz_est = df_poses[[prefix + 'q_w', prefix + 'q_x', prefix + 'q_y', prefix + 'q_z']].to_numpy()
    return PoseTrajectory3D(xyz_est, wxyz_est, df_poses[['t']].to_numpy())
