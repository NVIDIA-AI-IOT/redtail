FROM ubuntu:14.04

# Build with:
# docker build -t erle-sim:indigo -f Dockerfile.indigo .

ENV HOME /root

COPY scripts/* $HOME/scripts/
RUN chmod +x -R $HOME/scripts/*
RUN chown -R root:root $HOME/scripts/*

WORKDIR $HOME
# Add repos:
#   ROS
#   Gazebo
#   DRC
RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key 0xB01FA116   && \
    sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -sc) main" > /etc/apt/sources.list.d/gazebo-stable.list' && \
    wget http://packages.osrfoundation.org/gazebo.key -O - | apt-key add -              && \
    sh -c 'echo "deb http://packages.osrfoundation.org/drc/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/drc-latest.list' && \
    wget http://packages.osrfoundation.org/drc.key -O - | apt-key add -                 && \
    apt-get update

# Install required components first.
RUN apt-get install -y --no-install-recommends  \
    gawk make git curl cmake g++                \
    python-dev python-pip python-matplotlib     \ 
    python-serial python-wxgtk2.8 python-scipy  \ 
    python-opencv python-numpy python-pyparsing \
    python-setuptools ccache realpath           \
    libopencv-dev libxml2-dev libxslt1-dev      \
    lsb-release unzip libgl1-mesa-dri

# Python packages and MAVProxy
RUN pip install wheel   && \
    pip install future  && \
    pip install pymavlink catkin_pkg --upgrade  && \
    pip install MAVProxy==1.5.2

# ArUco
RUN wget "https://downloads.sourceforge.net/project/aruco/OldVersions/aruco-1.3.0.tgz?ts=1494024608&use_mirror=cytranet" -O aruco-1.3.0.tgz && \
    tar -xvzf aruco-1.3.0.tgz  && \
    cd aruco-1.3.0/            && \
    mkdir build && cd build    && \
    cmake ..                   && \
    make -j `nproc`            && \
    make install

# APM
WORKDIR $HOME
RUN mkdir -p simulation && \
    cd simulation       && \
    git clone https://github.com/erlerobot/ardupilot -b gazebo

# ROS
# Install and init ROS base and MAVROS
RUN apt-get install -y ros-indigo-ros-base ros-indigo-mavros ros-indigo-mavros-extras && \
    rosdep init     && \
    rosdep update   && \
    echo "source /opt/ros/indigo/setup.bash" >> $HOME/.bashrc

# Install ROS packages
RUN apt-get install -y --no-install-recommends python-rosinstall \
    ros-indigo-octomap-msgs    \
    ros-indigo-joy             \
    ros-indigo-geodesy         \
    ros-indigo-octomap-ros     \
    ros-indigo-mavlink         \
    ros-indigo-control-toolbox \
    ros-indigo-transmission-interface \
    ros-indigo-joint-limits-interface \
    ros-indigo-image-transport      \
    ros-indigo-cv-bridge            \
    ros-indigo-angles               \
    ros-indigo-polled-camera        \
    ros-indigo-camera-info-manager  \
    ros-indigo-controller-manager

# Install Gazebo
WORKDIR $HOME
RUN apt-get install -y --no-install-recommends gazebo7 libgazebo7-dev drcsim7

# Create ROS workspace
WORKDIR $HOME
RUN bash $HOME/scripts/init_workspace.bash

WORKDIR $HOME/simulation
