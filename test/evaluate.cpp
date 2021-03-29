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
#include <x_vio_ros/ros_utils.h>
#include <boost/progress.hpp>


using PoseCsv = x::CsvWriter<std::string,
                             double,
                             double, double, double,
                             double, double, double, double>;

using GTCsv = x::CsvWriter<double,
                           double, double, double,
                           double, double, double, double>;

void addPose(PoseCsv& csv, const std::string& update_modality, const x::State& s) {
  csv.addRow(update_modality, s.getTime(),
             s.getPosition().x(), s.getPosition().y(), s.getPosition().z(),
             s.getOrientation().x(), s.getOrientation().y(), s.getOrientation().z(), s.getOrientation().w());
}



DEFINE_string(input_bag, "", "filename of the bag to scan");
//DEFINE_string(events_topic, "/cam0/events", "topic in rosbag publishing dvs_msgs::EventArray");
DEFINE_string(image_topic, "/cam0/image_raw", "topic in rosbag publishing sensor_msgs::Image");
DEFINE_string(pose_topic, "", "(optional) topic publishing IMU pose ground truth as geometry_msgs::PoseStamped");
DEFINE_string(imu_topic, "/imu", "topic in rosbag publishing sensor_msgs::Imu");
DEFINE_string(params_file, "", "filename of the params.yaml to use");


int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  PoseCsv pose_csv("pose.csv", {"update_modality", "t",
                                "estimated_p_x", "estimated_p_y", "estimated_p_z",
                                "estimated_q_x", "estimated_q_y", "estimated_q_z", "estimated_q_w"});

  GTCsv gt_csv("gt.csv", {"t_gt", "closest_gt_p_x", "closest_gt_p_y", "closest_gt_p_z",
                          "closest_gt_q_x", "closest_gt_q_y", "closest_gt_q_z", "closest_gt_q_w"});


  x::CsvWriter<double, double, double> rt_csv("realtime.csv", {"t_sim", "t_real", "rt_factor"});

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

  rosbag::View view(bag);

  std::cerr << "Initializing at time " << view.getBeginTime().toSec() << std::endl;
  vio.initAtTime(view.getBeginTime().toSec());

  std::cerr << "Processing rosbag from time " << view.getBeginTime() << " to " << view.getEndTime()
            << std::endl << std::endl;

  uint64_t counter_imu = 0, counter_image = 0, counter_events = 0, counter_pose = 0;
  bool filer_initialized = false;
  boost::progress_display show_progress(view.size(), std::cerr);

  double rt_factor_resolution = .5;
  double next_rt_factor = rt_factor_resolution;
  profiler::timestamp_t calculation_time = 0, last_calculation_time = 0;

  EASY_PROFILER_ENABLE;

  for (rosbag::MessageInstance const &m : view) {
    bool processing_useful_message = false;

    if (m.getTopic() == FLAGS_imu_topic) {
      processing_useful_message = true;
      EASY_BLOCK("IMU Message");
      auto msg = m.instantiate<sensor_msgs::Imu>();
      ++counter_imu;

      auto a_m = x::msgVector3ToEigen(msg->linear_acceleration);
      auto w_m = x::msgVector3ToEigen(msg->angular_velocity);

      auto state = vio.processImu(msg->header.stamp.toSec(), msg->header.seq, w_m, a_m);
      addPose(pose_csv, "IMU", state);

    } else if (m.getTopic() == FLAGS_image_topic) {
      processing_useful_message = true;
      EASY_BLOCK("Image Message", profiler::colors::Green);
      auto msg = m.instantiate<sensor_msgs::Image>();
      ++counter_image;

      x::TiledImage image;
      if (!x::msgToTiledImage(params, msg, image))
        continue;
      x::TiledImage feature_img(image);
      auto state = vio.processImageMeasurement(image.getTimestamp(), image.getFrameNumber(), image, feature_img);
      addPose(pose_csv, "Image", state);

//    } else if (m.getTopic() == FLAGS_events_topic) {
//      processing_useful_message = true;
//      auto msg = m.instantiate<dvs_msgs::EventArray>();
//      ++counter_events;

    } else if (m.getTopic() == FLAGS_pose_topic) {
      EASY_BLOCK("GT Message", profiler::colors::Yellow);
      auto p = m.instantiate<geometry_msgs::PoseStamped>();
      ++counter_pose;
      gt_csv.addRow(p->header.stamp.toSec(), p->pose.position.x, p->pose.position.y, p->pose.position.z,
                    p->pose.orientation.x, p->pose.orientation.y, p->pose.orientation.z, p->pose.orientation.w);
    }

    if (processing_useful_message && filer_initialized) {
      // frame time is the duration of the last ended "root" block
      auto duration_in_us  = profiler::this_thread_frameTime();
      calculation_time += duration_in_us;

      double rt_factor = std::numeric_limits<double>::quiet_NaN();

      if (m.getTime().toSec() >= next_rt_factor) {
        next_rt_factor += rt_factor_resolution;

        rt_factor = static_cast<double>(calculation_time-last_calculation_time) * 1e-6 / rt_factor_resolution;

        // reset calculation time
        last_calculation_time = calculation_time;
      }

      rt_csv.addRow(m.getTime().toSec(), calculation_time*1e-6, rt_factor);
    }

    if (!filer_initialized && vio.isInitialized()) {
      filer_initialized = true;
      next_rt_factor = m.getTime().toSec() + rt_factor_resolution;
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

  bag.close();
}


// // Former approach, for reference only --> do this in python
//void compareWithClosestGTPose(PoseCsv &csv, const std::string& update_modality, const x::State &x_state,
//                              const boost::circular_buffer<geometry_msgs::PoseStamped> &recent_gt_poses,
//                              double &max_time_diff) {
//  auto closest_pose = std::lower_bound(recent_gt_poses.begin(), recent_gt_poses.end(), x_state.getTime(),
//                                       [](const geometry_msgs::PoseStamped& pose, double time) {
//                                          return pose.header.stamp.toSec() < time;
//                                       });
//
//  max_time_diff = std::max(max_time_diff, fabs(closest_pose->header.stamp.toSec() - x_state.getTime()));
//
//  csv.addRow(update_modality, x_state.getTime(), closest_pose->header.stamp.toSec(),
//             x_state.getPosition().x(), x_state.getPosition().y(), x_state.getPosition().z(),
//             x_state.getOrientation().x(), x_state.getOrientation().y(), x_state.getOrientation().z(), x_state.getOrientation().w(),
//             closest_pose->pose.position.x, closest_pose->pose.position.y, closest_pose->pose.position.z,
//             closest_pose->pose.orientation.x, closest_pose->pose.orientation.y, closest_pose->pose.orientation.z, closest_pose->pose.orientation.w);
//
//  auto pos = x_state.getPosition();
//  auto pos_gt = x::msgVector3ToEigen(closest_pose->pose.position);
//  auto error = pos - pos_gt;
//
////  std::cerr << "Time diff: " << closest_pose->header.stamp.toSec() - x_state.getTime() << std::endl;
////  std::cerr << "Pose error: " << error.x() << " " << error.y() << " " << error.z() << std::endl;
//}
