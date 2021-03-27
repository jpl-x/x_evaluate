//
// Created by Florian Mahlknecht on 2021-03-15.
// Copyright (c) 2021 NASA / JPL. All rights reserved.

#include <gflags/gflags.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <easy/profiler.h>

#include <x_vio_ros/parameter_loader.h>
#include <x/vio/vio.h>
#include <x/common/csv_writer.h>

#include <geometry_msgs/PoseStamped.h>
//#include <dvs_msgs/EventArray.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <cv_bridge/cv_bridge.h>
#include <x_vio_ros/ros_utils.h>
#include <boost/progress.hpp>
#include <boost/circular_buffer.hpp>
#include <algorithm>

void compareWithClosestGTPose(const x::State &x_state,
                              const boost::circular_buffer<geometry_msgs::PoseStamped> &recent_gt_poses,
                              double &max_time_diff);

DEFINE_string(input_bag, "", "filename of the bag to scan");
//DEFINE_string(events_topic, "/cam0/events", "topic in rosbag publishing dvs_msgs::EventArray");
DEFINE_string(image_topic, "/cam0/image_raw", "topic in rosbag publishing sensor_msgs::Image");
DEFINE_string(pose_topic, "", "(optional) topic publishing IMU pose ground truth as geometry_msgs::PoseStamped");
DEFINE_string(imu_topic, "/imu", "topic in rosbag publishing sensor_msgs::Imu");
DEFINE_string(params_file, "", "filename of the params.yaml to use");


int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  x::CsvWriter<std::string, double, double> csv("test.csv", {"type", "x", "accuracy"});
  csv.addRow("IMU", 1.6, 0.000004*(-1));

  // directly reads yaml file, without the need for a ROS master / ROS parameter server
  YAML::Node config = YAML::LoadFile(FLAGS_params_file);
  x::ParameterLoader l;
  x::Params params;
  auto success = l.loadXParamsWithYamlFile(config, params);
  std::cerr << "Reading config '" << FLAGS_params_file << "' was " << (success? "successful" : "failing") << std::endl;

  if (!success)
    return 1;

  std::cerr << "Reading rosbag '" << FLAGS_input_bag << "'" << std::endl;
  rosbag::Bag bag;
  bag.open(FLAGS_input_bag);  // BagMode is Read by default


  x::VIO vio;
  vio.setUp(params);

  bool initialized = false;

  rosbag::View view(bag);

  std::cerr << "Initializing at time " << view.getBeginTime().toSec() << std::endl;
  vio.initAtTime(view.getBeginTime().toSec());

  std::cerr << "Processing rosbag from time " << view.getBeginTime() << " to " << view.getEndTime()
            << std::endl << std::endl;

  uint64_t counter_imu = 0, counter_image = 0, counter_events = 0, counter_pose = 0;
  bool filer_initialized = false;
  double max_time_diff = 0.0;
  x::State most_recent_state;
  boost::circular_buffer<geometry_msgs::PoseStamped> recent_gt_poses(10);
  boost::progress_display show_progress(view.size(), std::cerr);

  EASY_PROFILER_ENABLE;

  for (rosbag::MessageInstance const &m : view) {
    if (m.getTopic() == FLAGS_imu_topic) {
      auto msg = m.instantiate<sensor_msgs::Imu>();
      ++counter_imu;

      auto a_m = x::msgVector3ToEigen(msg->linear_acceleration);
      auto w_m = x::msgVector3ToEigen(msg->angular_velocity);

      most_recent_state = vio.processImu(msg->header.stamp.toSec(), msg->header.seq, w_m, a_m);

    } else if (m.getTopic() == FLAGS_image_topic) {
      auto msg = m.instantiate<sensor_msgs::Image>();
      ++counter_image;

      x::TiledImage image;
      if (!x::msgToTiledImage(params, msg, image))
        continue;
      x::TiledImage feature_img(image);
      most_recent_state = vio.processImageMeasurement(image.getTimestamp(), image.getFrameNumber(), image, feature_img);

//    } else if (m.getTopic() == FLAGS_events_topic) {
//      auto msg = m.instantiate<dvs_msgs::EventArray>();
//      ++counter_events;

    } else if (m.getTopic() == FLAGS_pose_topic) {
      auto msg = m.instantiate<geometry_msgs::PoseStamped>();
      ++counter_pose;
      recent_gt_poses.push_back(*msg);

      if (filer_initialized) {
        compareWithClosestGTPose(most_recent_state, recent_gt_poses, max_time_diff);
      }
    }

    if (!filer_initialized && vio.isInitialized()) {
      filer_initialized = true;
      auto count = show_progress.count();
      show_progress.restart(view.size());
      show_progress += count;
    }

    ++show_progress;
  }

  profiler::dumpBlocksToFile("test.prof");

  std::cerr << "Processed " << counter_imu << " IMU, "
            << counter_image << " image, "
            << counter_events << " events and "
            << counter_pose << " pose messages" << std::endl;

  std::cerr << "Maximum time difference in GT pose alignment: " << max_time_diff << "s" << std::endl;

  bag.close();
}

void compareWithClosestGTPose(const x::State &x_state,
                              const boost::circular_buffer<geometry_msgs::PoseStamped> &recent_gt_poses,
                              double& max_time_diff) {
  auto closest_pose = std::lower_bound(recent_gt_poses.begin(), recent_gt_poses.end(), x_state.getTime(),
                                       [](const geometry_msgs::PoseStamped& pose, double time) {
                                          return pose.header.stamp.toSec() < time;
                                       });

  auto pos = x_state.getPosition();

  auto pos_gt = x::msgVector3ToEigen(closest_pose->pose.position);

  auto error = pos - pos_gt;
  max_time_diff = std::max(max_time_diff, fabs(closest_pose->header.stamp.toSec() - x_state.getTime()));

//  std::cerr << "Time diff: " << closest_pose->header.stamp.toSec() - x_state.getTime() << std::endl;
//  std::cerr << "Pose error: " << error.x() << " " << error.y() << " " << error.z() << std::endl;
}
