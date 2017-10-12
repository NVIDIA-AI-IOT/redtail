#!/bin/bash
#

# Stop in case of any error
set -e

source /opt/ros/indigo/setup.bash

WDIR=`pwd`

# Create catkin workspace
mkdir -p simulation/ros_catkin_ws/src
cd simulation/ros_catkin_ws/src
catkin_init_workspace
cd ..
catkin_make
source devel/setup.bash

# Get packages
cd $WDIR/simulation/ros_catkin_ws/src

git clone https://github.com/ros-drivers/driver_common
git clone https://github.com/erlerobot/ardupilot_sitl_gazebo_plugin
git clone https://github.com/erlerobot/rotors_simulator -b sonar_plugin
git clone https://github.com/PX4/mav_comm.git
git clone https://github.com/ethz-asl/glog_catkin.git
git clone https://github.com/catkin/catkin_simple.git
git clone https://github.com/erlerobot/mavros.git
git clone https://github.com/ros-simulation/gazebo_ros_pkgs.git -b indigo-devel
git clone https://github.com/erlerobot/gazebo_python_examples
#git clone https://github.com/erlerobot/gazebo_cpp_examples
#git clone https://github.com/tu-darmstadt-ros-pkg/hector_gazebo/                

# Make packages
cd ..
catkin_make --pkg mav_msgs mavros_msgs gazebo_msgs
source devel/setup.bash
catkin_make -j `nproc`

# Gazebo models
cd $WDIR
mkdir -p .gazebo/models
cd .gazebo/models
git clone https://github.com/erlerobot/erle_gazebo_models
