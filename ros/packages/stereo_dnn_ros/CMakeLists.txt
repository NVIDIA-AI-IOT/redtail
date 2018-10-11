cmake_minimum_required(VERSION 3.1)
project(stereo_dnn_ros)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  message_filters
  sensor_msgs
)

## System dependencies are found with CMake's conventions
#find_package(Boost REQUIRED COMPONENTS system)
find_package(CUDA)
# OpenCV 3.3.1 is default on Jetson starting JetPack 3.2. Might be required to install in simulation environment.
# PATHS is required as ROS Kinetic installs its own version of OpenCV 3.3.1 without CUDA support.
find_package(OpenCV 3.3.1 REQUIRED
  CONFIG
  PATHS /usr/local /usr
  NO_DEFAULT_PATH
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES stereo_dnn_ros
#  CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(
  CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS}; 
    -O3
)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_53,code=compute_53)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_62,code=sm_62)

set(stereo_dnn_lib_dir    ${CMAKE_SOURCE_DIR}/stereoDNN)
set(stereo_dnn_sample_dir ${stereo_dnn_lib_dir}/sample_app)

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  #include
  ${catkin_INCLUDE_DIRS}
  ${stereo_dnn_lib_dir}/lib
  ${stereo_dnn_sample_dir}
)

## Locations of library files.
link_directories(
  ${stereo_dnn_lib_dir}/build/lib
)

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(stereo_dnn_ros ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

file(GLOB stereo_dnn_ros_sources src/*.cpp)

## Declare a C++ executable
cuda_add_executable(stereo_dnn_ros_node ${stereo_dnn_ros_sources})

# Add StereoDNN sample networks.
set(stereo_dnn_sample_dir ${CMAKE_SOURCE_DIR}/stereoDNN/sample_app)
target_sources(stereo_dnn_ros_node PRIVATE 
  ${stereo_dnn_sample_dir}/nvsmall_1025x321_net.cpp
  ${stereo_dnn_sample_dir}/nvtiny_513x161_net.cpp
  ${stereo_dnn_sample_dir}/resnet18_1025x321_net.cpp
  ${stereo_dnn_sample_dir}/resnet18_2D_513x257_net.cpp
)

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(stereo_dnn_ros_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(stereo_dnn_ros_node
  ${catkin_LIBRARIES}
  nvstereo_inference_debug
  ${CUDA_LIBRARIES}
  cudnn
  nvinfer
  opencv_core
  opencv_imgproc
  opencv_highgui
)
