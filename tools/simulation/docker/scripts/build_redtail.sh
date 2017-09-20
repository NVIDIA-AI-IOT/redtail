#!/bin/bash

# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

REDTAIL_PATH=$1
if [[ -z "${REDTAIL_PATH}" ]]; then
    echo "First argument is missing."
    echo "Usage  : build_redtail.sh <full_path_to_redtail>"
    echo "Example: build_redtail.sh /data/src/redtail"
    exit 1
fi

cd ${CATKIN_WS}

if [[ ! -L "${CATKIN_WS}/src/caffe_ros" ]]; then
    ln -s ${REDTAIL_PATH}/ros/packages/caffe_ros ${CATKIN_WS}/src/
    catkin_make caffe_ros_node -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF
fi

if [[ ! -L "${CATKIN_WS}/src/image_pub" ]]; then
    ln -s ${REDTAIL_PATH}/ros/packages/image_pub ${CATKIN_WS}/src/
fi

if [[ ! -L "${CATKIN_WS}/src/px4_controller" ]]; then
    ln -s ${REDTAIL_PATH}/ros/packages/px4_controller ${CATKIN_WS}/src/
fi

catkin_make
