import numpy as np


def vec_to_skew_mat(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def quat_to_rot_vel_mat(quat_wxyz):
    h_mat = np.empty((3, 4))
    h_mat[:, 0] = -quat_wxyz[1:]
    h_mat[:, 1:] = vec_to_skew_mat(quat_wxyz[1:]) + np.eye(3) *  quat_wxyz[0]

    return h_mat * 2


def quat_left_mat(quat_wxyz):
    mat = np.empty((4, 4))
    mat[:, 0] = quat_wxyz
    mat[0, 1:] = -quat_wxyz[1:]
    mat[1:, 1:] = vec_to_skew_mat(quat_wxyz[1:]) + np.eye(3) * quat_wxyz[0]
    return mat


def quat_mul(q0, q1):
    return np.matmul(quat_left_mat(q0), q1)


def quat_inv(quat_wxyz):
    quat_wxyz[1:] *= -1
    return quat_wxyz
