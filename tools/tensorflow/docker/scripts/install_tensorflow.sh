#!/bin/bash

# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

# Stop in case of any error.
set -e

wget https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.5.0-cp35-cp35m-linux_x86_64.whl

conda create -n tensorflow python=3.5 anaconda
source activate tensorflow
conda install -y -c menpo opencv3
pip install --ignore-installed --upgrade tensorflow_gpu-1.5.0-cp35-cp35m-linux_x86_64.whl
pip install plyfile

echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/cuda/extras/CUPTI/lib64/" >> ${HOME}/.bashrc

echo "source activate tensorflow" >> ${HOME}/.bashrc

