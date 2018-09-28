cmake_minimum_required(VERSION 3.1)
project(stereo_dnn_ros_viz)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  message_filters
  sensor_msgs
)

find_package(OpenCV 3.3.1 REQUIRED
  CONFIG
  PATHS /usr/local /usr
  NO_DEFAULT_PATH
)

catkin_package()

###########
## Build ##
###########

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

include_directories(
  #include
  ${catkin_INCLUDE_DIRS}
)

file(GLOB stereo_dnn_ros_viz_sources src/*.cpp)

## Declare a C++ executable
add_executable(stereo_dnn_ros_viz_node ${stereo_dnn_ros_viz_sources})

## Specify libraries to link a library or executable target against
target_link_libraries(stereo_dnn_ros_viz_node
  ${catkin_LIBRARIES}
  opencv_core
  opencv_imgproc
  opencv_highgui
)
