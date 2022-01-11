import evo.core
import numpy as np
from evo.core.transformations import quaternion_matrix
import evo.core.trajectory


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


def quaternion_sequence_to_rotational_velocity(t_wxyz):
    # q_wxyz_from = q_wxyz_from[::5, :]
    # q_wxyz_to = q_wxyz_to[::5, :]
    # time_diff = time_diff[::5]
    q_wxyz_from = t_wxyz[:-1, 1:]
    q_wxyz_to = t_wxyz[1:, 1:]
    time_diff = t_wxyz[1:, 0] - t_wxyz[:-1, 0]

    assert np.allclose(np.linalg.norm(q_wxyz_from, axis=1), 1)

    # q_wxyz_from_inv = np.apply_along_axis(quaternion_inverse, 1, q_wxyz_from)
    # quat_from_mat = np.apply_along_axis(quat_left_mat, 1, q_wxyz_from_inv)

    # calc_angular_speed()

    # calc_angular_speed()
    # np.apply_along_axis(calc_angular_speed, 1, q_wxyz_from, q_wxyz_to, t_wxyz[:-1, 0], t_wxyz[1:, 0])
    # q_diff = np.matmul(quat_from_mat, q_wxyz_to[:, :, None]).squeeze(-1)
    # q_dot = q_diff
    # q_dot = np.empty_like(q_diff)
    #
    # for i in range(len(q_dot)):
    #     theta_half = np.arccos(q_diff[i, 0])
    #     n = np.array([1, 0, 0])
    #     if np.abs(np.sin(theta_half)) > 1e-5:
    #         n = q_diff[i, 1:] / np.sin(theta_half)
    #
    #     q_dot[i, :] = theta_half * n
    #     # s = Slerp([0, time_diff[i]], R.from_quat([[1.0, 0, 0, 0], q_diff[i, :]]))
    #     # q_dot[i, :] = s(1).as_quat()

    # https://fgiesen.wordpress.com/2012/08/24/quaternion-differentiation/


    # expm()

    # q_diff = q_wxyz_to - q_wxyz_from
    # q_diff[:, 0] /= time_diff
    # q_diff[:, 1] /= time_diff
    # q_diff[:, 2] /= time_diff
    # mapping_matrices = np.apply_along_axis(quat_to_rot_vel_mat, 1, q_wxyz_from)
    # omegas = np.matmul(mapping_matrices, q_dot[:, :, None]).squeeze(-1)
    # rot_vel = np.linalg.norm(omegas, axis=1)

    # rot_vel = rot_vel / time_diff
    #
    rot_vel = np.empty((len(q_wxyz_from), 3))
    for i, (q1, q2, delta_t) in enumerate(zip(q_wxyz_from, q_wxyz_to, time_diff)):
        R_1 = quaternion_matrix(q1)
        R_2 = quaternion_matrix(q2)

        R_dot = (R_2 - R_1) / delta_t

        omega_skew = R_1.T @ R_dot
        omega_skew = 1/2 * (omega_skew - omega_skew.T)
        omega = np.array([omega_skew[1, 2], omega_skew[0, 2], omega_skew[0, 1]])
        rot_vel[i, :] = np.linalg.norm(omega)
        # rot_vel[i] = calc_angular_speed(quaternion_matrix(q1), quaternion_matrix(q2), t1, t2)
    return rot_vel


def calculate_velocities(trajectory: evo.core.trajectory.Trajectory):
    time_diff = trajectory.timestamps[1:] - trajectory.timestamps[:-1]
    velocity = (trajectory.positions_xyz[1:, :] - trajectory.positions_xyz[:-1, :])
    velocity[:, 0] /= time_diff
    velocity[:, 1] /= time_diff
    velocity[:, 2] /= time_diff
    t_wxyz = np.hstack((trajectory.timestamps[:, np.newaxis], trajectory.orientations_quat_wxyz))
    rot_vel = quaternion_sequence_to_rotational_velocity(t_wxyz)
    return trajectory.timestamps[:-1], velocity, rot_vel


def moving_average(t, data, time_window=10):
    delta_t = t[1:] - t[:-1]

    # 3-sigma region of difference should be within 1% of our time window to be able to assume constant delta_t
    if 3*np.std(delta_t) > 0.01*time_window:
        raise NotImplemented("Moving average has not been implemented for variable delta_t timesteps")

    fixed_n = int(np.round(time_window / np.mean(delta_t)))

    data = moving_average_fixed_n(data, fixed_n)
    t_out = t[fixed_n - 1:]
    return t_out, data


def moving_average_fixed_n(data: np.array, n: int):
    cum_sum = np.cumsum(data)
    # difference gets partial sums (with index n-1 subtracting '0')
    cum_sum[n:] = cum_sum[n:] - cum_sum[:-n]
    return cum_sum[n - 1:] / n
