#!/bin/bash

# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

TENSORRT_TAR_FILE=$1
if [[ -z "${TENSORRT_TAR_FILE}" ]]; then
    echo "First argument is missing."
    echo "Usage  : build_redtail_image.sh <full_path_to_TensorRT_tar_file>"
    echo "Example: build_redtail_image.sh /data/downloads/NVIDIA/TensorRT-2.1.2.x86_64.cuda-8.0-14-04.tar.bz2"
    exit 1
fi

# Copy the file to Docker context first.
cp ${TENSORRT_TAR_FILE} .
# Build the image.
docker build -t nvidia-redtail-sim:kinetic --build-arg TENSORRT_TAR_FILE=`basename ${TENSORRT_TAR_FILE}` -f Dockerfile.kinetic .
