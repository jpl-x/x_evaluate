import argparse
import os
import sys

import numpy as np
import rospy
from matplotlib import pyplot as plt

from rosbag import Bag
from x_evaluate.plots import PlotContext


def main():
    parser = argparse.ArgumentParser(description="Plots event rate from BAG")
    parser.add_argument('--input', type=str, required=True)

    args = parser.parse_args()

    input_bag = Bag(args.input, 'r')

    topic_info = input_bag.get_type_and_topic_info()

    event_topics = [k for k, t in topic_info.topics.items() if t.msg_type == 'dvs_msgs/EventArray']

    if len(event_topics) > 1:
        print(F"WARNING: multiple event topics found ({event_topics}), taking first: '{event_topics[0]}'")
    elif len(event_topics) == 0:
        print("No dvs_msgs/EventArray found in bag")
        sys.exit()

    event_topic = event_topics[0]

    event_array_messages = []

    for topic, msg, t in input_bag.read_messages([event_topic]):
        event_array_messages.append(msg)

    event_times = np.array([e.ts.to_sec() for ea in event_array_messages for e in ea.events])

    start = input_bag.get_start_time()
    end = input_bag.get_end_time()

    event_times -= start

    bins = np.arange(start, end, 1.0)
    events_per_sec, t = np.histogram(event_times, bins=bins)

    with PlotContext() as pc:
        ax = pc.get_axis()
        ax.set_title(F"Events per second")
        ax.plot(t[1:], events_per_sec)
        ax.set_xlabel("time")
        ax.set_ylabel("events/s")

    # block for visualization
    plt.show()


if __name__ == '__main__':
    main()
