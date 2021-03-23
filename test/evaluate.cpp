//
// Created by Florian Mahlknecht on 2021-03-15.
// Copyright (c) 2021 NASA / JPL. All rights reserved.

#include <gflags/gflags.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <x_vio_ros/parameter_loader.h>
#include <type_traits>

DEFINE_string(input_bag, "", "filename of the bag to scan");
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

//
//  for(rosbag::MessageInstance const &m : rosbag::View(bag))
//  {
//    std_msgs::Int32::ConstPtr i = m.instantiate<std_msgs::Int32>();
//    if (i != nullptr)
//      std::cout << i->data << std::endl;
//  }

  bag.close();
}