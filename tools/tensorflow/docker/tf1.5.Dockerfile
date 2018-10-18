FROM nvidia/cudagl:9.0-devel-ubuntu16.04

# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

# Build with:
# docker build -t nvidia-redtail-tf:1.5 -f tf1.5.Dockerfile .

ENV HOME /root

WORKDIR ${HOME}

RUN apt-get update && apt-get -y  --no-install-recommends install software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test

# REVIEW alexeyk: libcudnn7=7.0.5.15 fixes the problem with TF 1.5 built against cuDNN 7.0 while recent version of the image has 7.1.
# https://github.com/tensorflow/tensorflow/issues/17566

RUN apt-get update && apt-get -y --no-install-recommends install \
        curl \
        git \
        gosu \
        libgtk2.0-dev \
        libjpeg-dev \
        libpng-dev \
        iproute \
        iputils-ping \
        less \
        libasound2 \
        libx11-xcb-dev \
        libxss1 \
        libcudnn7=7.0.5.15-1+cuda9.0 libcudnn7-dev=7.0.5.15-1+cuda9.0 \
        mc \
        nano \
        net-tools \
        patch \
        pkg-config \
        protobuf-compiler \
        rsync \
        sudo \
        unzip \
        wget \
        zip \
    && apt-get -y autoremove \
    && apt-get clean autoclean \
    # cleanup
    && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

# Anaconda
WORKDIR ${HOME}
RUN wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.0.1-Linux-x86_64.sh \
    && chmod +x ./Anaconda3-5.0.0.1-Linux-x86_64.sh \
    && ./Anaconda3-5.0.0.1-Linux-x86_64.sh -b

ENV PATH ${HOME}/anaconda3/bin:${PATH}

# TensorFlow conda environment setup.
COPY ./scripts/install_tensorflow.sh ${HOME}
RUN ${HOME}/install_tensorflow.sh

WORKDIR ${HOME}
