cmake_minimum_required(VERSION 3.1)
project(image_pub)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  camera_info_manager
)
find_package(OpenCV REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs camera_info_manager
  DEPENDS        OpenCV
)

###########
## Build ##
###########

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

include_directories(
  ${catkin_INCLUDE_DIRS}
)

file(GLOB image_pub_sources src/*.cpp)

add_executable(image_pub_node ${image_pub_sources})

target_link_libraries(image_pub_node
  ${catkin_LIBRARIES}
  opencv_core
  opencv_imgproc
  opencv_highgui
)
