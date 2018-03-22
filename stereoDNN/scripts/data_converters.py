# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

"""
Contains data conversion routines used in other scripts.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def nhwc_to_nchw(src):
    """Converts tensor from NHWC to NCHW format. Usually
    used for converting input/output tensors.
    NHWC format is commonly used by TensorFlow 
    while NCHW - TensorRT/cuDNN."""
    assert len(src.shape) == 4, "Expected 4D tensor."
    return np.transpose(src, [0, 3, 1, 2])

def rsck_to_kcrs(src):
    """Converts tensor from RSCK to KCRS format. Usually
    used for converting convolution filter tensors.
    cuDNN notation is used for dimensions where RS are
    spatial dimensions (height and width), C - number
    of input channels and K - number of output channels.
    """
    assert len(src.shape) == 4, "Expected 4D tensor."
    return np.transpose(src, [3, 2, 0, 1])

def ndhwc_to_ndchw(src):
    """Converts tensor from NDHWC to NDCHW format. Usually
    used for converting input/output tensors.
    NDHWC format is commonly used by TensorFlow 
    while NDCHW - TensorRT/cuDNN."""
    assert len(src.shape) == 5, "Expected 5D tensor."
    return np.transpose(src, [0, 1, 4, 2, 3])

def ndhwc_to_ncdhw(src):
    """Converts tensor from NDHWC to NCDHW format. Usually
    used for converting input/output tensors.
    NDHWC format is commonly used by TensorFlow 
    while NCDHW - TensorRT/cuDNN. This particular format
    is used to convert input of 3D convolution from TensorFlow
    to cuDNN format."""
    assert len(src.shape) == 5, "Expected 5D tensor."
    return np.transpose(src, [0, 4, 1, 2, 3])

def vrsck_to_kvcrs(src):
    """Converts tensor from VRSCK to KVCRS format. Usually
    used for converting convolution filter tensors.
    cuDNN notation is used for dimensions where RS are
    spatial dimensions (height and width), V - depth
    dimension, C - number of input channels 
    and K - number of output channels.
    """
    assert len(src.shape) == 5, "Expected 5D tensor."
    return np.transpose(src, [4, 0, 3, 1, 2])

