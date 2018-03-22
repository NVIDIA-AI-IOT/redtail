cmake_minimum_required(VERSION 3.1)
project(caffe_ros)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
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
  INCLUDE_DIRS include
#  LIBRARIES caffe_ros
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
    -O3 -gencode arch=compute_53,code=sm_53
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(caffe_ros ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
file(GLOB caffe_ros_sources src/*.cpp)

cuda_add_executable(caffe_ros_node ${caffe_ros_sources})

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(caffe_ros_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(caffe_ros_node
  ${catkin_LIBRARIES}
  nvcaffe_parser
  nvinfer
  opencv_core
  opencv_imgproc
  opencv_highgui
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS caffe_ros caffe_ros_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  add_rostest_gtest(${PROJECT_NAME}_tests tests/tests_basic.launch tests/tests.cpp)
  target_link_libraries(${PROJECT_NAME}_tests 
    ${catkin_LIBRARIES}
    opencv_core
    opencv_imgproc
    opencv_highgui
  )

  ## Add gtest based cpp test target and link libraries
  #catkin_add_gtest(${PROJECT_NAME}_test test/test_caffe_ros.cpp)
  #if(TARGET ${PROJECT_NAME}-test)
  #  target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
  #endif()
endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
