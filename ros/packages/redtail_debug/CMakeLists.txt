cmake_minimum_required(VERSION 3.1)
project(redtail_debug)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  cmake_modules
  tf
  eigen_conversions
)
find_package(Eigen3 REQUIRED)

catkin_package(
  CATKIN_DEPENDS 
    roscpp
    std_msgs
    geometry_msgs
    cmake_modules
    tf
    eigen_conversions
  DEPENDS EIGEN3
)

###########
## Build ##
###########

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

include_directories(
  ${catkin_INCLUDE_DIRS}
)

file(GLOB redtail_debug_sources src/*.cpp)

add_executable(redtail_debug_node ${redtail_debug_sources})

target_link_libraries(redtail_debug_node
  ${catkin_LIBRARIES}
  ${tf_LIBRARIES}
)
