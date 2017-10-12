#!/bin/bash
#

# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

# Stop in case of any error.
set -e

source /opt/ros/kinetic/setup.bash

# Create catkin workspace.
mkdir -p ${CATKIN_WS}/src
cd ${CATKIN_WS}/src
catkin_init_workspace
cd ..
catkin_make
source devel/setup.bash

# Install gscam ROS package and its dependencies.
cd ${HOME}
git clone https://github.com/ros-perception/image_common.git
ln -s ${HOME}/image_common/camera_calibration_parsers ${CATKIN_WS}/src/
ln -s ${HOME}/image_common/camera_info_manager ${CATKIN_WS}/src/
ln -s ${HOME}/image_common/image_transport ${CATKIN_WS}/src/

cd ${HOME}
git clone https://github.com/ros-drivers/gscam.git
cd gscam
# At present, master branch does not support GStreamer 1.0 so need to switch to gstreamer-1-0-support branch.
git checkout gstreamer-1-0-support
# Currently you get a build error in gscam (‘scoped_ptr’ in namespace ‘boost’ does not name a template type).
sed -i '9i\#include <boost/scoped_ptr.hpp>' ./include/gscam/gscam_nodelet.h
# Create symlink to catkin workspace.
ln -s ${HOME}/gscam ${CATKIN_WS}/src/

cd ${CATKIN_WS}
catkin_make -DGSTREAMER_VERSION_1_x=On

# angles package used in px4 controller.
cd ${HOME}
git clone https://github.com/ros/angles.git
# Create symlink to catkin workspace.
ln -s ${HOME}/angles/angles ${CATKIN_WS}/src/

# Not getting Redtail sources to give an option to use mapped volume when creating a container.
# caffe_ros ROS node might need to be built with this:
# catkin_make caffe_ros_node -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF
# otherwise it might fail duing linking with: cannot find -lopencv_dep_cudart

echo 'source ${CATKIN_WS}/devel/setup.bash' >> ${HOME}/.bashrc
