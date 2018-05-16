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

# Install gscam ROS package.
cd ${HOME}
apt-get install -y ros-kinetic-camera-info-manager ros-kinetic-camera-calibration-parsers ros-kinetic-image-transport
git clone https://github.com/ros-drivers/gscam.git
# Create symlink to catkin workspace.
ln -s ${HOME}/gscam ${CATKIN_WS}/src/

cd ${CATKIN_WS}
catkin_make -DGSTREAMER_VERSION_1_x=On

# angles package used in px4 controller.
cd ${HOME}
apt-get install -y ros-kinetic-angles

# Not getting Redtail sources to give an option to use mapped volume when creating a container.
# caffe_ros ROS node might need to be built with this:
# catkin_make caffe_ros_node -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF
# otherwise it might fail duing linking with: cannot find -lopencv_dep_cudart

echo 'source ${CATKIN_WS}/devel/setup.bash' >> ${HOME}/.bashrc
