import copy
from typing import List, Tuple

import numpy as np
from evo.core.trajectory import PoseTrajectory3D
from evo.core import lie_algebra as lie
from evo.core import sync

from x_evaluate.evaluation_data import AlignmentType


def get_best_yaw(C):
    '''
    maximize trace(Rz(theta) * C)
    '''
    assert C.shape == (3, 3)

    A = C[0, 1] - C[1, 0]
    B = C[0, 0] + C[1, 1]
    theta = np.pi / 2 - np.arctan2(B, A)

    return theta


def rot_z(theta):

    R = np.identity(3)
    R[0, 0] = np.cos(theta)
    R[0, 1] = -np.sin(theta)
    R[1, 0] = np.sin(theta)
    R[1, 1] = np.cos(theta)

    return R


def rpg_get_alignment_umeyama(ground_truth: PoseTrajectory3D, trajectory_estimate: PoseTrajectory3D, known_scale=False, yaw_only=False):
    """Implementation of the paper: S. Umeyama, Least-Squares Estimation
    of Transformation Parameters Between Two Point Patterns,
    IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.

    model = s * R * data + t

    Input:
    model -- first trajectory (nx3), numpy array type
    data -- second trajectory (nx3), numpy array type

    Output:
    s -- scale factor (scalar)
    R -- rotation matrix (3x3)
    t -- translation vector (3x1)
    t_error -- translational error per point (1xn)

    """

    # substract mean

    mu_M = ground_truth.positions_xyz.mean(0)
    mu_D = trajectory_estimate.positions_xyz.mean(0)
    model_zerocentered = ground_truth.positions_xyz - mu_M
    data_zerocentered = trajectory_estimate.positions_xyz - mu_D
    n = np.shape(ground_truth.positions_xyz)[0]

    # correlation
    C = 1.0/n*np.dot(model_zerocentered.transpose(), data_zerocentered)
    sigma2 = 1.0/n*np.multiply(data_zerocentered, data_zerocentered).sum()
    U_svd, D_svd, V_svd = np.linalg.linalg.svd(C)
    D_svd = np.diag(D_svd)
    V_svd = np.transpose(V_svd)

    S = np.eye(3)
    if(np.linalg.det(U_svd)*np.linalg.det(V_svd) < 0):
        S[2, 2] = -1

    if yaw_only:
        rot_C = np.dot(data_zerocentered.transpose(), model_zerocentered)
        theta = get_best_yaw(rot_C)
        R = rot_z(theta)
    else:
        R = np.dot(U_svd, np.dot(S, np.transpose(V_svd)))

    if known_scale:
        s = 1
    else:
        s = 1.0/sigma2*np.trace(np.dot(D_svd, S))

    t = mu_M-s*np.dot(R, mu_D)

    return s, R, t


def rpg_align(ground_truth, trajectory_estimate, alignment_type: AlignmentType = AlignmentType.Disabled,
              use_subtrajectory=False, t_left=3, t_right=8):

    if use_subtrajectory:
        gt = copy.deepcopy(ground_truth)
        gt.reduce_to_time_range(t_left, t_right)
        est = copy.deepcopy(ground_truth)
        est.reduce_to_time_range(t_left, t_right)
    else:
        gt = ground_truth
        est = trajectory_estimate
    if alignment_type == AlignmentType.PosYaw:
        s, r, t = rpg_get_alignment_umeyama(gt, est, known_scale=True, yaw_only=True)
    elif alignment_type == AlignmentType.SE3:
        s, r, t = rpg_get_alignment_umeyama(gt, est, known_scale=True, yaw_only=False)
    elif alignment_type == AlignmentType.SIM3:
        s, r, t = rpg_get_alignment_umeyama(gt, est, known_scale=False, yaw_only=False)
    elif alignment_type == AlignmentType.Disabled:
        s = 1
        r = np.identity(3)
        t = np.zeros((3, ))
    else:
        raise ValueError(F"Invalid alignment type {alignment_type}")

    trajectory_estimate.scale(s)
    trajectory_estimate.transform(lie.se3(r, t))


def get_split_distances_at_percentages(ground_truth: PoseTrajectory3D, percentages):
    split_distances = [ground_truth.distances[-1] * p for p in percentages]
    split_distances = np.round(split_distances, 2)
    return split_distances


def get_split_distances_on_equal_parts(ground_truth: PoseTrajectory3D, num_parts=5):
    percentages = [i / num_parts for i in range(1, num_parts)]
    return get_split_distances_at_percentages(ground_truth, percentages)


def get_split_distances_equispaced(ground_truth: PoseTrajectory3D, step_size=5):
    distance = ground_truth.distances[-1]
    return np.arange(0, distance, step_size)


def rpg_sub_trajectories(ground_truth: PoseTrajectory3D, trajectory_estimate: PoseTrajectory3D, split_distances,
                         max_diff=0.01) -> List[Tuple[PoseTrajectory3D, PoseTrajectory3D, float]]:

    split_points = [-1] * len(split_distances)

    for i in range(len(split_distances)):
        split_points[i] = np.argmin(np.abs(ground_truth.distances-split_distances[i]))

    split_times = [ground_truth.timestamps[index] for index in split_points]

    split_times.insert(0, ground_truth.timestamps[0])
    split_distances.append(ground_truth.distances[-1])
    split_times.append(ground_truth.timestamps[-1])

    trajectory_tuples = []

    for i in range(len(split_distances)):
        sub_ground_truth = copy.deepcopy(ground_truth)
        sub_ground_truth.reduce_to_time_range(split_times[i], split_times[i+1])

        sub_trajectory = copy.deepcopy(trajectory_estimate)
        sub_trajectory.reduce_to_time_range(split_times[i], split_times[i+1])

        sub_ground_truth, sub_trajectory = sync.associate_trajectories(sub_ground_truth, sub_trajectory, max_diff)

        trajectory_tuples.append((sub_ground_truth, sub_trajectory, np.round(split_distances[i], 2)))

    return trajectory_tuples


