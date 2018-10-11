FROM nvidia/cudagl:9.0-devel-ubuntu16.04

# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

# Build with:
# docker build -t nvidia-redtail-sim:kinetic-v2 --build-arg TENSORRT_TAR_FILE=<TensorRT_tar_name> -f Dockerfile.kinetic .

ENV HOME /root

ARG TENSORRT_TAR_FILE

WORKDIR ${HOME}

RUN apt-get update && apt-get -y  --no-install-recommends install software-properties-common

# cuDNN version must match the one used by TensorRT.
# TRT 4.0 is compiled with cuDNN 7.1.

RUN apt-get update && apt-get -y --no-install-recommends install \
        ant \
        bzip2 \
        ca-certificates \
        ccache \
        cmake \
        curl \
        genromfs \
        git \
        gosu \
        iproute \
        iputils-ping \
        less \
        lcov \
        libcudnn7=7.1.4.18-1+cuda9.0 \
        libcudnn7-dev=7.1.4.18-1+cuda9.0 \
        libeigen3-dev \
        libopencv-dev \
        make \
        nano \
        net-tools \
        ninja-build \
        openjdk-8-jdk \
        patch \
        pkg-config \
        protobuf-compiler \
        python-argparse \
        python-dev \
        python-empy \
        python-numpy \
        python-pip \
        python-serial \
        python-software-properties \
        rsync \
        s3cmd \
        software-properties-common \
        sudo \
        unzip \
        xsltproc \
        wget \
        zip \
    && apt-get -y autoremove \
    && apt-get clean autoclean \
    # pip
    && pip install setuptools wheel \
    && pip install 'matplotlib==2.2.2' --force-reinstall \
    # coveralls code coverage reporting
    && pip install cpp-coveralls \
    # jinja template generation
    && pip install jinja2 \
    # cleanup
    && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/* \
    # Add Fast-RTPS
    && cd /opt && curl http://www.eprosima.com/index.php/component/ars/repository/eprosima-fast-rtps/eprosima-fast-rtps-1-5-0/eprosima_fastrtps-1-5-0-linux-tar-gz?format=raw | tar xz eProsima_FastRTPS-1.5.0-Linux/share/fastrtps/fastrtpsgen.jar eProsima_FastRTPS-1.5.0-Linux/bin/fastrtpsgen \
    && ln -s /opt/eProsima_FastRTPS-1.5.0-Linux/bin/fastrtpsgen /usr/local/bin/fastrtpsgen \
    && mkdir -p /usr/local/share/fastrtps && ln -s /opt/eProsima_FastRTPS-1.5.0-Linux/share/fastrtps/fastrtpsgen.jar /usr/local/share/fastrtps/fastrtpsgen.jar

# GStreamer
RUN apt-get -y --no-install-recommends install \
    gstreamer1.0-plugins-base        \
    gstreamer1.0-plugins-bad         \
    gstreamer1.0-plugins-ugly        \
    gstreamer1.0-plugins-base-apps   \
    gstreamer1.0-plugins-good        \
    gstreamer1.0-tools               \
    libgstreamer1.0-dev              \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libyaml-cpp-dev                  \
    v4l-utils

# Gazebo
WORKDIR ${HOME}
RUN wget --quiet http://packages.osrfoundation.org/gazebo.key -O - | apt-key add - \
    && sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable xenial main" > /etc/apt/sources.list.d/gazebo-stable.list' \
    && apt-get update && apt-get -y --no-install-recommends install \
        gazebo7         \
        libgazebo7-dev  \
    # px4tools
    && pip install px4tools \
    # dronekit latest
    && git clone https://github.com/dronekit/dronekit-python.git                         \
    && (cd dronekit-python && pip install -r requirements.txt) && rm -rf dronekit-python \
    # pymavlink latest
    && git clone https://github.com/ArduPilot/pymavlink.git && cd pymavlink                             \
    && git clone git://github.com/mavlink/mavlink.git && ln -s ${PWD}/mavlink/message_definitions ../   \
    && pip install . && cd .. && rm -rf pymavlink && rm -rf message_definitions

# PX4 firmware
WORKDIR ${HOME}
RUN mkdir ./px4/ && cd ./px4/                                        \
    && git clone https://github.com/PX4/Firmware.git && cd Firmware/ \
    && git checkout v1.4.4

# ROS Kinetic
WORKDIR ${HOME}
RUN apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116 \
    && sh -c 'echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list' \
    && sh -c 'echo "deb http://packages.ros.org/ros-shadow-fixed/ubuntu/ xenial main" > /etc/apt/sources.list.d/ros-shadow.list' \
    && apt-get update && apt-get -y --no-install-recommends install \
        ros-kinetic-gazebo-ros-pkgs \
        ros-kinetic-mavros          \
        ros-kinetic-mavros-extras   \
        ros-kinetic-ros-base        \
        ros-kinetic-joy             \
        ros-kinetic-rviz            \
    && apt-get -y autoremove        \
    && apt-get clean autoclean      \
    && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

# Initialize ROS
RUN geographiclib-get-geoids egm96-5 \
    && rosdep init                   \
    && rosdep update

RUN echo 'source /opt/ros/kinetic/setup.bash' >> ${HOME}/.bashrc

# Install OpenCV with CUDA support.
# REVIEW alexeyk: JetPack 3.2 comes with OpenCV 3.3.1 _without_ CUDA support.
WORKDIR ${HOME}
RUN git clone https://github.com/opencv/opencv.git && cd opencv \
    && git checkout 3.3.1                   \
    && mkdir build && cd build              \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE    \
        -D CMAKE_INSTALL_PREFIX=/usr/local  \
        -D WITH_CUDA=OFF                     \
        -D WITH_OPENCL=OFF                  \
        -D ENABLE_FAST_MATH=1               \
        -D CUDA_FAST_MATH=1                 \
        -D WITH_CUBLAS=1                    \
        -D BUILD_DOCS=OFF                   \
        -D BUILD_PERF_TESTS=OFF             \
        -D BUILD_TESTS=OFF                  \
        ..                                  \
    && make -j `nproc`                      \
    && make install                         \
    && cd ${HOME} && rm -rf ./opencv/

# Install TensorRT
WORKDIR ${HOME}
COPY ${TENSORRT_TAR_FILE} ${HOME}

ENV TENSORRT_BASE_DIR /opt/tensorrt
# REVIEW alexeyk: lift to a build argument.
ENV TENSORRT_VER      4.0.1.6
ENV TENSORRT_DIR      ${TENSORRT_BASE_DIR}/TensorRT-${TENSORRT_VER}

RUN mkdir ${TENSORRT_BASE_DIR} \
    && tar -xf ${TENSORRT_TAR_FILE} -C ${TENSORRT_BASE_DIR} \
    && echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:${TENSORRT_DIR}/lib" >> ${HOME}/.bashrc \
    && echo "export LIBRARY_PATH=\${LIBRARY_PATH}:${TENSORRT_DIR}/lib" >> ${HOME}/.bashrc       \
    && echo "export CPATH=\${CPATH}:${TENSORRT_DIR}/include" >> ${HOME}/.bashrc

# Setup catkin workspace
ENV CATKIN_WS ${HOME}/ws
COPY ./scripts/init_workspace.sh ${HOME}
RUN ${HOME}/init_workspace.sh

# To be run by a user after creating a container.
COPY ./scripts/build_redtail.sh ${HOME}

ENV CCACHE_CPP2=1
ENV CCACHE_MAXSIZE=1G
ENV DISPLAY :0
#ENV PATH "/usr/lib/ccache:$PATH"
ENV TERM=xterm
# Some QT-Apps/Gazebo don't not show controls without this
ENV QT_X11_NO_MITSHM 1

# SITL UDP PORTS
EXPOSE 14556/udp
EXPOSE 14557/udp
