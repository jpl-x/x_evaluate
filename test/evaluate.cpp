//
// Created by Florian Mahlknecht on 2021-03-15.
// Copyright (c) 2021 NASA / JPL. All rights reserved.

#include <gflags/gflags.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <tf2_msgs/TFMessage.h>
#include <iostream>
#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem>
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif
#include <memory>
#include <yaml-cpp/yaml.h>
#include <easy/profiler.h>
//#include <easy/converter/converter.h>

#include <x_vio_ros/parameter_loader.h>
#include <x/vio/vio.h>
#include <x/eklt/eklt_vio.h>
#include <x/events/e_vio.h>
#include <x/common/csv_writer.h>

#include <geometry_msgs/PoseStamped.h>
#include <dvs_msgs/EventArray.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <x_vio_ros/ros_utils.h>
#include <boost/progress.hpp>

#include <sys/resource.h>
#include <ctime>

enum class Frontend : int8_t {
  XVIO = 0,
  EKLT = 1,
  EVIO = 2,
};

std::map<std::string, Frontend> frontends {
  {"XVIO", Frontend::XVIO},
  {"EKLT", Frontend::EKLT},
  {"EVIO", Frontend::EVIO},
};


DEFINE_string(input_bag, "", "filename of the bag to scan");
DEFINE_string(events_topic, "", "topic in rosbag publishing dvs_msgs::EventArray");
DEFINE_string(image_topic, "/cam0/image_raw", "topic in rosbag publishing sensor_msgs::Image");
DEFINE_string(pose_topic, "", "(optional) topic publishing IMU pose ground truth as geometry_msgs::PoseStamped");
DEFINE_string(imu_topic, "/imu", "topic in rosbag publishing sensor_msgs::Imu");
DEFINE_string(params_file, "", "filename of the params.yaml to use");
DEFINE_string(output_folder, "", "folder where to write output files, is created if not existent");

static bool validateFrontend(const char* flagname, const std::string& value) {
  if (frontends.find(value) != frontends.end())
    return true;
  std::string possible_values;
  bool is_first = true;
  for (const auto& v : frontends) {
    if (is_first) {
      possible_values += v.first;
      is_first = false;
    } else {
      possible_values += ", " + v.first;
    }
  }
  std::cerr << "Invalid error for '" << flagname << "'. Possible values: " << possible_values << std::endl;
  return false;
}

DEFINE_string(frontend, "XVIO", "which frontend to use");
DEFINE_validator(frontend, &validateFrontend);
DEFINE_double(from, std::numeric_limits<double>::min(), "skip messages with timestamp lower than --form");
DEFINE_double(to, std::numeric_limits<double>::max(), "skip messages with timestamp bigger than --to");


using PoseCsv = x::CsvWriter<std::string,
                             double,
                             double, double, double,
                             double, double, double, double>;
using ImuBiasCsv = x::CsvWriter<double,
                             double, double, double,
                             double, double, double>;

using GTCsv = x::CsvWriter<double,
                           double, double, double,
                           double, double, double, double>;

void addPose(PoseCsv& csv, const std::string& update_modality, const x::State& s) {
  csv.addRow(update_modality, s.getTime(),
             s.getPosition().x(), s.getPosition().y(), s.getPosition().z(),
             s.getOrientation().x(), s.getOrientation().y(), s.getOrientation().z(), s.getOrientation().w());
}

void addImuBias(ImuBiasCsv& csv, const std::string& update_modality, const x::State& s) {
  csv.addRow(s.getTime(),
             s.getAccelerometerBias().x(), s.getAccelerometerBias().y(), s.getAccelerometerBias().z(),
             s.getGyroscopeBias().x(), s.getGyroscopeBias().y(), s.getGyroscopeBias().z());
}

char* get_time_string_in_utc() {
  std::time_t curr_time;
  curr_time = std::time(nullptr);
  tm *tm_gmt = std::gmtime(&curr_time);
  return std::asctime(tm_gmt);
}

// FIXME the usage of VioClass as template is a big hack, enabling fast testing of VIO + EKLTVIO, without code duplication
// FIXME a common interface in X library (baseclass) or a parameter "enable_eklt" in x_params might be a solution
template <typename VioClass>
int evaluate() {
  {

    if (FLAGS_output_folder.empty()) {
      std::cerr << "ERROR: No output folder specified, provide --output_folder" << std::endl;
      return 1;
    }

    // directly reads yaml file, without the need for a ROS master / ROS parameter server
    YAML::Node config = YAML::LoadFile(FLAGS_params_file);
    x::ParameterLoader l;
    x::Params params;
    auto success = l.loadXParamsWithYamlFile(config, params);
    std::cerr << "Reading config '" << FLAGS_params_file << "' was " << (success ? "successful" : "failing")
              << std::endl;

    if (!success)
      return 1;

    fs::path output_path(FLAGS_output_folder);

    fs::create_directories(output_path);
    fs::copy(FLAGS_params_file, output_path / "params.yaml", fs::copy_options::overwrite_existing);


    PoseCsv pose_csv(output_path / "pose.csv", {"update_modality", "t",
                                                "estimated_p_x", "estimated_p_y", "estimated_p_z",
                                                "estimated_q_x", "estimated_q_y", "estimated_q_z", "estimated_q_w"});
    ImuBiasCsv imu_bias_csv(output_path / "imu_bias.csv", {"t", "b_a_x", "b_a_y", "b_a_z", "b_w_x", "b_w_y", "b_w_z"});


    std::unique_ptr<GTCsv> gt_csv(nullptr);

    if (!FLAGS_pose_topic.empty())
      gt_csv.reset(new GTCsv(output_path / "gt.csv", {"t", "p_x", "p_y", "p_z", "q_x", "q_y", "q_z", "q_w"}));

    x::CsvWriter<double, double, profiler::timestamp_t, std::string, profiler::timestamp_t> rt_csv(
      output_path / "realtime.csv", {"t_sim", "t_real", "ts_real", "processing_type", "process_time_in_us"});

    x::CsvWriter<profiler::timestamp_t, double, double, double, size_t, size_t> resource_csv(output_path / "resource.csv",
                   {"ts", "cpu_usage", "cpu_user_mode_usage", "cpu_kernel_mode_usage", "memory_usage_in_bytes", "debug_memory_in_bytes"});

    x::XVioPerformanceLoggerPtr xvio_logger = std::make_shared<x::XVioPerformanceLogger>(output_path);

    x::EkltPerformanceLoggerPtr  eklt_logger;
    if constexpr (std::is_same<VioClass, x::EKLTVIO>::value) {
      eklt_logger = std::make_shared<x::EkltPerformanceLogger>(output_path);
    }

    std::cerr << "Reading rosbag '" << FLAGS_input_bag << "'" << std::endl;
    rosbag::Bag bag;
    bag.open(FLAGS_input_bag);  // BagMode is Read by default


    VioClass vio;
    if constexpr (std::is_same<VioClass, x::EKLTVIO>::value) {

      vio.setUp(params, xvio_logger, eklt_logger);
    } else {
      vio.setUp(params, xvio_logger);
    }

    auto from = ros::TIME_MIN;
    auto to = ros::TIME_MAX;

    // if initialized differently from default values
    if (FLAGS_from > std::numeric_limits<double>::min())
      from = ros::Time(FLAGS_from);
    if (FLAGS_to < std::numeric_limits<double>::max())
      to = ros::Time(FLAGS_to);

    rosbag::View view(bag, from, to);

    std::cerr << "Initializing at time " << view.getBeginTime().toSec() << std::endl;
    vio.initAtTime(view.getBeginTime().toSec());

    std::cerr << "Processing rosbag from time " << view.getBeginTime() << " to " << view.getEndTime()
              << std::endl << std::endl;

    uint64_t counter_imu = 0, counter_image = 0, counter_events = 0, counter_pose = 0;
    bool filer_initialized = false;
    x::State state;
    auto t_0 = std::numeric_limits<double>::infinity();
    auto t_last_flush = std::numeric_limits<double>::infinity();
    boost::progress_display show_progress(view.size(), std::cerr);

    profiler::timestamp_t calculation_time = 0, last_rusage_check = 0;

    struct timeval rusage_walltime;
    gettimeofday(&rusage_walltime, nullptr);

    struct rusage prev_rusage;
    getrusage(RUSAGE_SELF, &prev_rusage);


    EASY_PROFILER_ENABLE;
    EASY_MAIN_THREAD;

    for (rosbag::MessageInstance const &m : view) {

      std::string process_type;

      auto start = profiler::now();

      if (m.getTopic() == FLAGS_imu_topic) {
        EASY_BLOCK("IMU Message", profiler::colors::Red);
        process_type = "IMU";
        auto msg = m.instantiate<sensor_msgs::Imu>();
        ++counter_imu;

        auto a_m = x::msgVector3ToEigen(msg->linear_acceleration);
        auto w_m = x::msgVector3ToEigen(msg->angular_velocity);

        state = vio.processImu(msg->header.stamp.toSec(), msg->header.seq, w_m, a_m);
        EASY_END_BLOCK;

      } else if (m.getTopic() == FLAGS_image_topic) {
        EASY_BLOCK("Image Message", profiler::colors::Green);
        process_type = "Image";
        auto msg = m.instantiate<sensor_msgs::Image>();
        ++counter_image;

        x::TiledImage image;
        if (!x::msgToTiledImage(params, msg, image))
          continue;
        x::TiledImage feature_img(image);
        state = vio.processImageMeasurement(image.getTimestamp(), image.getFrameNumber(), image, feature_img);
        EASY_END_BLOCK;

      } else if (!FLAGS_events_topic.empty() && m.getTopic() == FLAGS_events_topic) {

        // this constexpr if is necessary, since VIO.processEventsMeasurement(...) would not compile with same arguments
        if constexpr (!std::is_same<VioClass, x::VIO>::value) {
          EASY_BLOCK("Events Message", profiler::colors::Blue);
          process_type = "Events";
          auto msg = m.instantiate<dvs_msgs::EventArray>();
          ++counter_events;

          x::EventArray::Ptr x_events = x::msgToEvents(msg);

          if constexpr (std::is_same<VioClass, x::EKLTVIO>::value) {
            x::TiledImage tracker_img, feature_img;

            state = vio.processEventsMeasurement(x_events, tracker_img, feature_img);
          } else if constexpr (std::is_same<VioClass, x::EVIO>::value) {
            // Initialize plain image to plot features on
            cv::Mat event_img(x_events->height,
                              x_events->width,
                              CV_32F,
                              cv::Scalar(0.0));
            state = vio.processEventsMeasurement(x_events, event_img);
          }
          EASY_END_BLOCK;
        }
      } else if (!FLAGS_pose_topic.empty() && m.getTopic() == FLAGS_pose_topic) {
        EASY_BLOCK("GT Message");
        if (m.isType<geometry_msgs::PoseStamped>()) {
          auto p = m.instantiate<geometry_msgs::PoseStamped>();
          ++counter_pose;
          gt_csv->addRow(p->header.stamp.toSec(), p->pose.position.x, p->pose.position.y, p->pose.position.z,
                         p->pose.orientation.x, p->pose.orientation.y, p->pose.orientation.z, p->pose.orientation.w);
        } else if (m.isType<tf2_msgs::TFMessage>()) {
          auto tf = m.instantiate<tf2_msgs::TFMessage>();
          for (const auto & p : tf->transforms) {
            ++counter_pose;
            gt_csv->addRow(p.header.stamp.toSec(), p.transform.translation.x, p.transform.translation.y, p.transform.translation.z,
                           p.transform.rotation.x, p.transform.rotation.y, p.transform.rotation.z, p.transform.rotation.w);
          }

        } else {
          std::cerr << "WARNING: unable to type of GT message: " << m.getTopic() << std::endl;
        }
        EASY_END_BLOCK;
      }

      // stop here --> all the rest is not considered
      auto stop = profiler::now();

      if (m.getTime().toSec() < t_0)
        t_0 = m.getTime().toSec();

      if (m.getTime().toSec() < t_last_flush)  // initialization
        t_last_flush = m.getTime().toSec();

      if (m.getTime().toSec() - t_last_flush > 5.0) {
        t_last_flush = m.getTime().toSec();
        x::DebugMemoryMonitor::instance().flush_all();
      }

      // profile 1s only to avoid huge files that are not handleable anymore
      if (m.getTime().toSec() - t_0 > 1.0)
        EASY_PROFILER_DISABLE;

      if (calculation_time - last_rusage_check >= 1000000) {
        last_rusage_check = calculation_time;
        struct timeval rusage_walltime_new;
        gettimeofday(&rusage_walltime_new, nullptr);

        double walltime_sec_passed = (rusage_walltime_new.tv_sec + rusage_walltime_new.tv_usec * 1e-6) -
                                   (rusage_walltime.tv_sec + rusage_walltime.tv_usec * 1e-6);

        struct rusage cur_rusage;
        getrusage(RUSAGE_SELF, &cur_rusage);

        double cpu_time_usr = (cur_rusage.ru_utime.tv_sec + cur_rusage.ru_utime.tv_usec * 1e-6) -
                              (prev_rusage.ru_utime.tv_sec + prev_rusage.ru_utime.tv_usec * 1e-6);
        double cpu_time_sys = (cur_rusage.ru_stime.tv_sec + cur_rusage.ru_stime.tv_usec * 1e-6) -
                              (prev_rusage.ru_stime.tv_sec + prev_rusage.ru_stime.tv_usec * 1e-6);

//        std::cout << "timings passed: WT: " << walltime_sec_passed
//                  << " USR: " << cpu_time_usr
//                  << " SYS: " << cpu_time_sys << std::endl;

        double cpu_usage = 100 * (cpu_time_sys + cpu_time_usr) / walltime_sec_passed;
        double cpu_usage_usr = 100 * cpu_time_usr / walltime_sec_passed;
        double cpu_usage_sys = 100 * cpu_time_sys / walltime_sec_passed;

        size_t mem_usage_in_bytes = cur_rusage.ru_maxrss * 1024L;
        size_t mem_usage_debug = x::DebugMemoryMonitor::instance().memory_usage_in_bytes();

        resource_csv.addRow(profiler::now(), cpu_usage, cpu_usage_usr, cpu_usage_sys, mem_usage_in_bytes, mem_usage_debug);

        rusage_walltime = rusage_walltime_new;
        prev_rusage = cur_rusage;
      }

      if (!filer_initialized && vio.isInitialized()) {
        filer_initialized = true;
//        auto count = show_progress.count();
//        show_progress.restart(view.size());
//        show_progress += count;
      }

      if (!process_type.empty() && filer_initialized) {
        auto duration_in_us = profiler::toMicroseconds(stop - start);
        calculation_time += duration_in_us;

        addPose(pose_csv, process_type, state);
        addImuBias(imu_bias_csv, process_type, state);
        rt_csv.addRow(m.getTime().toSec(), calculation_time * 1e-6, profiler::now(), process_type, duration_in_us);
      }

      ++show_progress;
    }

    profiler::dumpBlocksToFile((output_path / "profiling.prof").c_str());
//    JsonExporter je;
//    je.convert((output_path / "profiling.prof").c_str(), (output_path / "profiling.json").c_str());

    std::cerr << "Processed " << counter_imu << " IMU, "
              << counter_image << " image, "
              << counter_events << " event and "
              << counter_pose << " pose messages" << std::endl;

    std::cerr << "Writing outputs to folder " << output_path << std::endl;

    // manually flush as workaround for memory corruption in EKLT node
    x::DebugMemoryMonitor::instance().flush_all();

    bag.close();

    // destructor calls (--> CSV flushing) happening here
  }
  std::cerr << "Evaluation completed " << get_time_string_in_utc();
  std::cerr << "Good bye!" << std::endl;
  return 0;
}


int main(int argc, char **argv) {

  std::cerr << "Running " << argv[0] << " " << get_time_string_in_utc() << std::endl;

  google::ParseCommandLineFlags(&argc, &argv, true);

  switch(frontends[FLAGS_frontend]) {
    case Frontend::XVIO:
      return evaluate<x::VIO>();
    case Frontend::EKLT:
      return evaluate<x::EKLTVIO>();
    case Frontend::EVIO:
      return evaluate<x::EVIO>();
    default:
      std::cerr << "Invalid frontend type, unable to evaluate" << std::endl;
      return 1;
  }
}
