#!/bin/bash

# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

REDTAIL_NAME=$1
if [[ -z "${REDTAIL_NAME}" ]]; then
    REDTAIL_NAME=redtail-sim-v2
fi

HOST_DATA_DIR=$2
if [[ -z "${HOST_DATA_DIR}" ]]; then
    HOST_DATA_DIR=/data/
fi

CONTAINER_DATA_DIR=$3
if [[ -z "${CONTAINER_DATA_DIR}" ]]; then
    CONTAINER_DATA_DIR=/data/
fi

echo "Container name    : ${REDTAIL_NAME}"
echo "Host data dir     : ${HOST_DATA_DIR}"
echo "Container data dir: ${CONTAINER_DATA_DIR}"
REDTAIL_ID=`docker ps -aqf "name=^/${REDTAIL_NAME}$"`
if [ -z "${REDTAIL_ID}" ]; then
    echo "Creating new redtail container."
    xhost +
    nvidia-docker run -it --privileged --network=host -v ${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}:rw -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix${DISPLAY} -p 14556:14556/udp --name=${REDTAIL_NAME} nvidia-redtail-sim:kinetic-v2 bash
else
    echo "Found redtail container: ${REDTAIL_ID}."
    # Check if the container is already running and start if necessary.
    if [ -z `docker ps -qf "name=^/${REDTAIL_NAME}$"` ]; then
        xhost +local:${REDTAIL_ID}
        echo "Starting and attaching to ${REDTAIL_NAME} container..."
        docker start ${REDTAIL_ID}
        docker attach ${REDTAIL_ID}
    else
        echo "Found running ${REDTAIL_NAME} container, attaching bash..."
        docker exec -it ${REDTAIL_ID} bash
    fi
fi

