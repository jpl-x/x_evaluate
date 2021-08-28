import argparse
import os
import sys

import numpy as np
import rospy

from rosbag import Bag


def main():
    parser = argparse.ArgumentParser(description="Reads geometry_msgs/PoseStamped messages and calculates initial "
                                                 "state for XVIO filter")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--t_from', type=float, default=None)
    parser.add_argument('--no_average', dest='use_average', default=True, action='store_false')
    parser.add_argument('--t_avg', type=float, default=0.05, help="timespan on which to calculate average velocity [s]")

    args = parser.parse_args()

    input_bag = Bag(args.input, 'r')

    topic_info = input_bag.get_type_and_topic_info()

    pose_topics = [k for k, t in topic_info.topics.items() if t.msg_type == 'geometry_msgs/PoseStamped']

    if len(pose_topics) > 1:
        print(F"Warning, multiple pose topics found ({pose_topics}), taking first: '{pose_topics[0]}'")
    elif len(pose_topics) == 0:
        print("No geometry_msgs/PoseStamped found in bag")
        sys.exit()

    pose_topic = pose_topics[0]

    t_from = None
    if args.t_from is not None:
        t_from = rospy.Time.from_sec(args.t_from)

    pose_messages = []

    for topic, msg, t in input_bag.read_messages([pose_topic], start_time=t_from):
        if len(pose_messages) > 0 and msg.header.stamp.to_sec() - pose_messages[0].header.stamp.to_sec() >= 0.1:
            break
        pose_messages.append(msg)

    x = [p.pose.position.x for p in pose_messages]
    y = [p.pose.position.y for p in pose_messages]
    z = [p.pose.position.z for p in pose_messages]
    qx = [p.pose.orientation.x for p in pose_messages]
    qy = [p.pose.orientation.y for p in pose_messages]
    qz = [p.pose.orientation.z for p in pose_messages]
    qw = [p.pose.orientation.w for p in pose_messages]
    t = [p.header.stamp.to_sec() for p in pose_messages]

    xyz = np.array([x, y, z]).T

    wxyz = np.array([qw, qx, qy, qz]).T

    delta_t = t[1] - t[0]
    start_velocity = (xyz[1, :] - xyz[0, :]) / delta_t
    avg_velocity = np.mean((xyz[1:, :] - xyz[:-1, :]) / delta_t, axis=0)

    velocity = avg_velocity if args.use_average else start_velocity

    init_v = list(velocity)
    init_p = list(xyz[0, :])
    init_q = list(wxyz[0, :])

    print(F"      # initial state computed from {pose_topic} at {t[0]}s in {os.path.basename(args.input)}:")
    print(F"      p: {init_p}")
    print(F"      v: {init_v}")
    print(F"      q: {init_q} #[w,x,y,z]")


if __name__ == '__main__':
    main()
