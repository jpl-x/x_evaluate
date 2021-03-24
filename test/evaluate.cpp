//
// Created by Florian Mahlknecht on 2021-03-15.
// Copyright (c) 2021 NASA / JPL. All rights reserved.

#include <gflags/gflags.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <x_vio_ros/parameter_loader.h>
#include <geometry_msgs/PoseStamped.h>
//#include <dvs_msgs/EventArray.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

DEFINE_string(input_bag, "", "filename of the bag to scan");
//DEFINE_string(events_topic, "/cam0/events", "topic in rosbag publishing dvs_msgs::EventArray");
DEFINE_string(image_topic, "/cam0/image_raw", "topic in rosbag publishing sensor_msgs::Image");
DEFINE_string(pose_topic, "", "(optional) topic publishing IMU pose ground truth as geometry_msgs::PoseStamped");
DEFINE_string(imu_topic, "/imu", "topic in rosbag publishing sensor_msgs::Imu");
DEFINE_string(params_file, "", "filename of the params.yaml to use");


int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::cerr << "Evaluation script is alive, trying to read from " << FLAGS_input_bag << std::endl;
  rosbag::Bag bag;
  bag.open(FLAGS_input_bag);  // BagMode is Read by default

  std::cerr << "Now reading from " << FLAGS_params_file << std::endl;

  // directly reads yaml file, without the need for a ROS master / ROS parameter server
  YAML::Node config = YAML::LoadFile(FLAGS_params_file);

  x::ParameterLoader l;
  x::Params params;

  auto success = l.loadXParamsWithYamlFile(config, params);

  std::cerr << "Reading config '" << FLAGS_params_file << "' was " << (success? "successful" : "failing") << std::endl;

  uint64_t counter_imu = 0, counter_image = 0, counter_events = 0, counter_pose = 0;

  for (rosbag::MessageInstance const &m : rosbag::View(bag))
  {
    if (m.getTopic() == FLAGS_imu_topic) {
      auto msg = m.instantiate<sensor_msgs::Imu>();
      ++counter_imu;
    } else if (m.getTopic() == FLAGS_image_topic) {
      auto msg = m.instantiate<sensor_msgs::Image>();
      ++counter_image;
//    } else if (m.getTopic() == FLAGS_events_topic) {
//      auto msg = m.instantiate<dvs_msgs::EventArray>();
//      ++counter_events;
    } else if (m.getTopic() == FLAGS_pose_topic) {
      auto msg = m.instantiate<geometry_msgs::PoseStamped>();
      ++counter_pose;
    }
  }

  std::cerr << "Processed " << counter_imu << " IMU, "
            << counter_image << " image, "
            << counter_events << " events and "
            << counter_pose << " pose messages" << std::endl;

  bag.close();
}