#!/bin/bash

# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

PX4_NAME=$1
if [[ -z "${PX4_NAME}" ]]; then
    PX4_NAME=px4-ros
fi

HOST_DATA_DIR=$2
if [[ -z "${HOST_DATA_DIR}" ]]; then
    HOST_DATA_DIR=/data/
fi

CONTAINER_DATA_DIR=$3
if [[ -z "${CONTAINER_DATA_DIR}" ]]; then
    CONTAINER_DATA_DIR=/data/
fi

NVIDIA_DOCKER_VOLUME=$4
if [[ -n "${NVIDIA_DOCKER_VOLUME}" ]]; then
    NVIDIA_DOCKER_VOLUME_PARAM="-v ${NVIDIA_DOCKER_VOLUME}:/usr/local/nvidia:ro"
fi

echo "Container name    : ${PX4_NAME}"
echo "Host data dir     : ${HOST_DATA_DIR}"
echo "Container data dir: ${CONTAINER_DATA_DIR}"
echo "NVIDIA Docker vol : ${NVIDIA_DOCKER_VOLUME}"
PX4_ID=`docker ps -aqf "name=^/${PX4_NAME}$"`
if [ -z "${PX4_ID}" ]; then
    echo "Creating new px4 container."
    xhost +
    docker run -it --privileged --network=host -v ${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}:rw ${NVIDIA_DOCKER_VOLUME_PARAM} -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix${DISPLAY} -p 14556:14556/udp --name=${PX4_NAME} px4io/px4-dev-ros:v1.0 bash
else
    echo "Found px4 container: ${PX4_ID}."
    # Check if the container is already running and start if necessary.
    if [ -z `docker ps -qf "name=^/${PX4_NAME}$"` ]; then
        xhost +local:${PX4_ID}
        echo "Starting and attaching to ${PX4_NAME} container..."
        docker start ${PX4_ID}
        docker attach ${PX4_ID}
    else
        echo "Found running ${PX4_NAME} container, attaching bash..."
        docker exec -it ${PX4_ID} bash
    fi
fi

