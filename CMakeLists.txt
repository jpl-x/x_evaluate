
cmake_minimum_required(VERSION 2.8.3)
cmake_policy(SET CMP0048 NEW)  # avoids CMAKE_PROJECT_VERSION warning
project(x_evaluate)

set (CMAKE_BUILD_TYPE Release)

if (CMAKE_BUILD_TYPE MATCHES Release)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# build with catkin build x_vio_ros --cmake-args -DMY_DEBUG=1
if (MY_DEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Og")
endif()

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Og -fno-omit-frame-pointer -fsanitize=address") # REMOVE ME LATER
#set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -g -Og -fno-omit-frame-pointer -fsanitize=address") # REMOVE ME LATER

find_package(catkin REQUIRED COMPONENTS
  x
  cv_bridge
  rosbag
  dvs_msgs
  sensor_msgs
  std_msgs
  gflags_catkin
  easy_profiler_catkin
)
find_package(yaml-cpp REQUIRED) # used in evaluation executable
catkin_python_setup()

catkin_package()

# Set build flags, depending on the architecture
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall")


include_directories (
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${easy_profiler_catkin_INCLUDE_DIRS}
)


set (FS_LIBRARY "")
if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
  set(FS_LIBRARY stdc++fs)
endif()

# add generic evaluation program as executable to be a able to call it from python
add_executable(evaluate ${SOURCE} test/evaluate.cpp)
target_link_libraries(evaluate
        ${OpenCV_LIBRARIES}
        ${catkin_LIBRARIES}
        ${rostest_LIBRARIES}
        ${YAML_CPP_LIBRARIES}
        ${easy_profiler_catkin_LIBRARIES}
        ${FS_LIBRARY}
)

target_compile_definitions(evaluate PUBLIC -DUSING_EASY_PROFILER)

catkin_install_python(PROGRAMS test/evaluate.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

