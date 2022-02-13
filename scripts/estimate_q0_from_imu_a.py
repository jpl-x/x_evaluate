import argparse
import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R
from rosbag import Bag

from x_evaluate.utils import get_ros_topic_name_from_msg_type, read_all_ros_msgs_from_topic_into_dict


def calculate_smallest_rotation_to_z_up_frame(gravity_observation: np.array, imu_acc_bias: np.array) -> R:
    gravity_target_frame = np.array([0, 0, 9.81])
    # bias = np.array([0.599, 0.008, 1.476])

    gravity_observation = gravity_observation - imu_acc_bias  # Bias is subtracted - see in x library: state.cpp:209

    xyzw = np.zeros(4)

    a = np.cross(gravity_observation, gravity_target_frame)
    xyzw[:3] = a

    b = np.sqrt((np.linalg.norm(gravity_observation) ** 2) * (np.linalg.norm(gravity_target_frame) ** 2)) + np.dot(gravity_observation, gravity_target_frame)
    xyzw[3] = b

    rot = R.from_quat(xyzw)
    return rot


class TestAttitudeEstimationFromImu(unittest.TestCase):

    def setUp(self):
        # generate 10 random rotations (always the same with seed 76419)
        self.random_rotations = R.random(10, 76419).as_matrix()
        np.random.seed(732094)
        self.random_biases = np.random.random((10, 3))
        self.expected_gravity = np.array([0, 0, 9.81])

    def general_test(self, gravity_observation, imu_acc_bias, expected_rotatation):
        rotation = calculate_smallest_rotation_to_z_up_frame(gravity_observation, imu_acc_bias)
        self.assertTrue(np.allclose(rotation.as_matrix(), expected_rotatation.as_matrix()))

    def general_test_with_biases(self, gravity_observation, expected_result):
        self.general_test(gravity_observation, [0, 0, 0], expected_result)
        for b in self.random_biases:
            self.general_test(gravity_observation + b, b, expected_result)

    def test_perfect_imu_measurement(self):
        self.general_test_with_biases(self.expected_gravity, R.identity())
        self.general_test(self.expected_gravity*2, [0, 0, 0], R.identity())

    def test_z_alignment(self):
        for random_rot in self.random_rotations:
            for bias in self.random_biases:
                rotated_observation = random_rot @ self.expected_gravity
                restult_rot = calculate_smallest_rotation_to_z_up_frame(rotated_observation + bias, bias)

                # scaling of observation must not influence result
                restult_rot_scaled = calculate_smallest_rotation_to_z_up_frame(rotated_observation*10 + bias*10,
                                                                               bias*10)
                rot_with_scaling_close = np.allclose(restult_rot.as_matrix(), restult_rot_scaled.as_matrix())

                z_up_body = rotated_observation / np.linalg.norm(rotated_observation)
                z_alignment_close = np.allclose(restult_rot.as_matrix() @ z_up_body, [0, 0, 1])
                self.assertTrue(z_alignment_close and rot_with_scaling_close)


# def test():
#     print("This should be true")
#     print()


def main():
    #
    # # RUN TESTS
    # unittest.main()

    parser = argparse.ArgumentParser(description="Reads first IMU measurements from bag and calculates the first "
                                                 "attitude that would bring us closest to a z-up gravity-down "
                                                 "reference frame. If available, compares results to GT in bag ")
    parser.add_argument('--input_bag', type=str, required=True)
    parser.add_argument("--ba_xyz", nargs=3, metavar=('ba_x', 'ba_y', 'ba_z'), help="my help message", type=float,
                        default=[0.0, 0.0, 0.0])

    args = parser.parse_args()

    input_bag = Bag(args.input_bag, 'r')

    # t_0 = input_bag.get_start_time()
    bias = np.array(args.ba_xyz)
    imu_topic = None
    gt_topic = None
    try:
        imu_topic = get_ros_topic_name_from_msg_type(input_bag, 'sensor_msgs/Imu')
    except LookupError:
        print("No IMU topic found in file...")
        exit(1)

    try:
        gt_topic = get_ros_topic_name_from_msg_type(input_bag, 'geometry_msgs/PoseStamped')
    except LookupError:
        pass  # ok for gt_topic to be non existent

    _, imu_message, t_imu = next(input_bag.read_messages([imu_topic]))

    imu_acc = np.array([imu_message.linear_acceleration.x, imu_message.linear_acceleration.y,
                        imu_message.linear_acceleration.z])

    R_hat = calculate_smallest_rotation_to_z_up_frame(imu_acc, args.ba_xyz)

    print(F"q0_hat in xyzw format: {list(R_hat.as_quat())}")
    print()
    print("For xVIO parameter file this means:")
    #q: [-0.329112656567, -0.0699724433216, 0.0214667765852, 0.941449889249]  # [w,x,y,z]
    print(F"q: {list(np.roll(R_hat.as_quat(), 1))}  # [w,x,y,z]")

    if gt_topic is not None:
        _, gt_message, t_gt = next(input_bag.read_messages([gt_topic]))
        print()
        print("Here is how well we did:")

        t_offset_in_ms = (t_gt.to_sec() - t_imu.to_sec()) * 1000

        print(F"GT minus IMU message times is: {t_offset_in_ms:.2f}ms")

        orientation = gt_message.pose.orientation

        R_gt = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])

        # rotations go from from inertial reference frame to body frame, we flip this around and compare the z-up
        # vectors in the body frame
        e_z_body = R_gt.inv().as_matrix() @ np.array([0, 0, 1])
        e_z_body_hat = R_hat.inv().as_matrix() @ np.array([0, 0, 1])
        angle = np.arccos(np.dot(e_z_body, e_z_body_hat))  # those are unit vectors, so norm = 1

        print(F"z-direction in body frame [GT]: {list(e_z_body)}")
        print(F"z-direction in body frame [EST]: {list(e_z_body)}")
        print(F"Z-direction vectors are {np.rad2deg(angle):.2f}° off")


if __name__ == '__main__':
    main()
