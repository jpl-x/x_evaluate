import argparse

import numpy as np
from matplotlib import pyplot as plt

from rosbag import Bag
from x_evaluate.plots import PlotContext
from x_evaluate.utils import get_ros_topic_name_from_msg_type, read_all_ros_msgs_from_topic_into_dict


def main():
    parser = argparse.ArgumentParser(description="Plots event rate from BAG")
    parser.add_argument('--input', type=str, required=True)

    args = parser.parse_args()

    input_bag = Bag(args.input, 'r')

    event_topic = get_ros_topic_name_from_msg_type(input_bag, 'dvs_msgs/EventArray')
    event_array_messages = read_all_ros_msgs_from_topic_into_dict(event_topic, input_bag)

    event_times = np.array([e.ts.to_sec() for ea in event_array_messages.values() for e in ea.events])

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
