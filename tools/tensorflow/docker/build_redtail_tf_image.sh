#!/bin/bash

# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

# Image tag suffix, e.g. 1.5.
IMAGE_TAG_SUFFIX=$2
if [[ -z "${IMAGE_TAG_SUFFIX}" ]]; then
    IMAGE_TAG_SUFFIX="1.5"
fi
echo "Using ${IMAGE_TAG_SUFFIX} image suffix."

# Build the image.
docker build -t nvidia-redtail-tf:${IMAGE_TAG_SUFFIX} -f tf1.5.Dockerfile .
