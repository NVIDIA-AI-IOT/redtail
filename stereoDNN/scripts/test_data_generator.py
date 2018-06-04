# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

"""
Generates data for nvstereo_inference library tests.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import struct
import time

import warnings
# Ignore 'FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated' warning.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning, module='h5py')
    import tensorflow as tf

from data_converters import *

parser = argparse.ArgumentParser(description='Stereo DNN TensorRT C++ test data generator')

parser.add_argument('--data_dir', type=str, help='directory path to store generated test data', required=True)

args = parser.parse_args()

def get_rand(shape):
    return np.random.randn(*shape).astype(np.float32)

def write_bin(src, filename):
    with open(filename, 'wb') as w:
        w.write(struct.pack('<i', len(src.shape)))
        for d in src.shape:
            w.write(struct.pack('<i', d))
        src.reshape(-1).tofile(w)

def create_elu_plugin_data(data_dir):
    # Create data for ELU plugin.
    print("---")
    print("Creating data for ELU plugin...")
    print("---")
    # Basic test.
    np.random.seed(1)
    dims   = [1, 2, 4, 3] # NHWC
    input  = 10 * get_rand(dims) - 5
    output = tf.nn.elu(input).eval()
    write_bin(nhwc_to_nchw(input),  os.path.join(data_dir, 'elu_i_01.bin'))
    write_bin(nhwc_to_nchw(output), os.path.join(data_dir, 'elu_o_01.bin'))

    # 4D input/batch size 2.
    np.random.seed(1)
    dims   = [2, 2, 3, 4, 3] # NDHWC
    input  = 10 * get_rand(dims) - 5
    output = tf.nn.elu(input).eval()
    write_bin(ndhwc_to_ndchw(input),  os.path.join(data_dir, 'elu_i_02.bin'))
    write_bin(ndhwc_to_ndchw(output), os.path.join(data_dir, 'elu_o_02.bin'))

def create_conv3d_plugin_data(data_dir):
    print("---")
    print("Creating data for Conv3D plugin...")
    print("---")
    # The most basic test: single output/feature map, no padding/unit strides.
    np.random.seed(1)
    x = get_rand([1, 1, 3, 3, 3]) # NDHWC
    w = get_rand([1, 3, 3, 3, 1]) # VRSCK
    s = [1, 1, 1, 1, 1]
    y = tf.nn.conv3d(x, w, s, padding='VALID').eval()
    write_bin(ndhwc_to_ndchw(x), os.path.join(data_dir, 'conv3d_01_x.bin'))
    write_bin(vrsck_to_kvcrs(w), os.path.join(data_dir, 'conv3d_01_w.bin'))
    write_bin(ndhwc_to_ndchw(y), os.path.join(data_dir, 'conv3d_01_y.bin'))

    # Testing HW padding and strides (D == 1, K == 1).
    np.random.seed(1)
    x = get_rand([1, 1, 5, 5, 3]) # NDHWC
    w = get_rand([1, 3, 3, 3, 1]) # VRSCK
    s = [1, 1, 2, 2, 1]
    y = tf.nn.conv3d(x, w, s, padding='SAME').eval()
    write_bin(ndhwc_to_ndchw(x), os.path.join(data_dir, 'conv3d_02_x.bin'))
    write_bin(vrsck_to_kvcrs(w), os.path.join(data_dir, 'conv3d_02_w.bin'))
    write_bin(ndhwc_to_ndchw(y), os.path.join(data_dir, 'conv3d_02_y.bin'))

    # Testing HW strides with DHW padding.
    np.random.seed(1)
    x = get_rand([1, 2, 3, 3, 3]) # NDHWC
    w = get_rand([2, 3, 3, 3, 1]) # VRSCK
    s = [1, 1, 2, 2, 1]
    y = tf.nn.conv3d(x, w, s, padding='SAME').eval()
    write_bin(ndhwc_to_ndchw(x), os.path.join(data_dir, 'conv3d_03_x.bin'))
    write_bin(vrsck_to_kvcrs(w), os.path.join(data_dir, 'conv3d_03_w.bin'))
    write_bin(ndhwc_to_ndchw(y), os.path.join(data_dir, 'conv3d_03_y.bin'))

    # Testing unit stride and DHW padding (padding is symmetrical in D dim) and multiple feature maps.
    # Note: similar to conv3D_1 block in NVSmall.
    np.random.seed(1)
    x = get_rand([1, 8, 9, 9, 3]) # NDHWC
    w = get_rand([3, 3, 3, 3, 4]) # VRSCK
    s = [1, 1, 1, 1, 1]
    y = tf.nn.conv3d(x, w, s, padding='SAME').eval()
    write_bin(ndhwc_to_ndchw(x), os.path.join(data_dir, 'conv3d_04_x.bin'))
    write_bin(vrsck_to_kvcrs(w), os.path.join(data_dir, 'conv3d_04_w.bin'))
    write_bin(ndhwc_to_ndchw(y), os.path.join(data_dir, 'conv3d_04_y.bin'))
    #write_bin(np.transpose(y, [0, 4, 1, 2, 3]), os.path.join(data_dir, 'conv3d_04_y.bin'))

    # Testing DHW strides with DHW padding (padding is non-symmetrical in D dim) and multiple feature maps.
    # Note: similar to conv3D_3ds block in NVSmall.
    np.random.seed(1)
    x = get_rand([1, 8, 9, 9, 3]) # NDHWC
    w = get_rand([3, 3, 3, 3, 4]) # VRSCK
    s = [1, 2, 2, 2, 1]
    y = tf.nn.conv3d(x, w, s, padding='SAME').eval()
    write_bin(ndhwc_to_ndchw(x), os.path.join(data_dir, 'conv3d_05_x.bin'))
    write_bin(vrsck_to_kvcrs(w), os.path.join(data_dir, 'conv3d_05_w.bin'))
    write_bin(ndhwc_to_ndchw(y), os.path.join(data_dir, 'conv3d_05_y.bin'))

    # Testing DHW strides with DHW padding (padding is non-symmetrical in D dim) and multiple feature maps
    # as well as bias and ELU activation.
    # Note: similar to full conv3D_3ds block in NVSmall.
    np.random.seed(1)
    x = get_rand([1, 8, 9, 9, 3]) # NDHWC
    w = get_rand([3, 3, 3, 3, 6]) # VRSCK
    b = get_rand([6])             # K
    s = [1, 2, 2, 2, 1]
    y   = tf.nn.conv3d(x, w, s, padding='SAME')
    y_b = tf.nn.bias_add(y, b)
    y_a = tf.nn.elu(y_b).eval()
    write_bin(ndhwc_to_ndchw(x),   os.path.join(data_dir, 'conv3d_06_x.bin'))
    write_bin(vrsck_to_kvcrs(w),   os.path.join(data_dir, 'conv3d_06_w.bin'))
    write_bin(b,                   os.path.join(data_dir, 'conv3d_06_b.bin'))
    write_bin(ndhwc_to_ndchw(y_a), os.path.join(data_dir, 'conv3d_06_y.bin'))

    # Testing end output padding in D dimension.
    # Note: similar to full conv3D_3ds->conv3D_4 blocks in NVSmall.
    np.random.seed(1)
    x   = get_rand([1, 8, 9, 9, 3]) # NDHWC
    w   = get_rand([3, 3, 3, 3, 3]) # VRSCK
    s_1 = [1, 1, 1, 1, 1]
    s_2 = [1, 2, 2, 2, 1]
    # First convo has non-symmetrical D padding.
    y_1 = tf.nn.conv3d(x,   w, s_1, padding='SAME')
    y_2 = tf.nn.conv3d(y_1, w, s_2, padding='SAME').eval()
    write_bin(ndhwc_to_ndchw(x),   os.path.join(data_dir, 'conv3d_07_x.bin'))
    write_bin(vrsck_to_kvcrs(w),   os.path.join(data_dir, 'conv3d_07_w.bin'))
    write_bin(ndhwc_to_ndchw(y_2), os.path.join(data_dir, 'conv3d_07_y.bin'))

def create_conv3d_tran_plugin_data(data_dir):
    print("---")
    print("Creating data for Conv3dTranspose plugin...")
    print("---")
    # The most basic test: single input/feature map, no padding/unit strides.
    np.random.seed(1)
    y = get_rand([1, 1, 1, 1, 1]) # NDHWK
    w = get_rand([1, 3, 3, 3, 1]) # VRSCK
    s          = [1, 1, 1, 1, 1]
    x_shape    = [1, 1, 3, 3, 3]  # NDHWC
    x = tf.nn.conv3d_transpose(y, w, output_shape=x_shape, strides=s, padding='VALID').eval()
    write_bin(ndhwc_to_ndchw(y), os.path.join(data_dir, 'conv3d_tran_01_y.bin'))
    write_bin(vrsck_to_kvcrs(w), os.path.join(data_dir, 'conv3d_tran_01_w.bin'))
    write_bin(ndhwc_to_ndchw(x), os.path.join(data_dir, 'conv3d_tran_01_x.bin'))

    # Testing HW padding and strides (D == 1, K == 1).
    np.random.seed(1)
    y = get_rand([1, 1, 3, 3, 1]) # NDHWK
    w = get_rand([1, 3, 3, 3, 1]) # VRSCK
    s          = [1, 1, 2, 2, 1]
    x_shape    = [1, 1, 5, 5, 3]  # NDHWC
    x = tf.nn.conv3d_transpose(y, w, output_shape=x_shape, strides=s, padding='SAME').eval()
    write_bin(ndhwc_to_ndchw(y), os.path.join(data_dir, 'conv3d_tran_02_y.bin'))
    write_bin(vrsck_to_kvcrs(w), os.path.join(data_dir, 'conv3d_tran_02_w.bin'))
    write_bin(ndhwc_to_ndchw(x), os.path.join(data_dir, 'conv3d_tran_02_x.bin'))

    # Testing DHW padding and strides.
    # Note: similar to deconv3D_1 convo in NVSmall.
    np.random.seed(1)
    y = get_rand([1, 4, 5, 5, 8]) # NDHWK
    w = get_rand([3, 3, 3, 4, 8]) # VRSCK
    s          = [1, 2, 2, 2, 1]
    x_shape    = [1, 8, 9, 9, 4]  # NDHWC
    x = tf.nn.conv3d_transpose(y, w, output_shape=x_shape, strides=s, padding='SAME').eval()
    write_bin(ndhwc_to_ncdhw(y), os.path.join(data_dir, 'conv3d_tran_03_y.bin'))
    write_bin(vrsck_to_kvcrs(w), os.path.join(data_dir, 'conv3d_tran_03_w.bin'))
    write_bin(ndhwc_to_ndchw(x), os.path.join(data_dir, 'conv3d_tran_03_x.bin'))

    # Testing DHW padding and strides with bias and ELU.
    # Note: similar to deconv3D_1 convo in NVSmall.
    np.random.seed(1)
    y = get_rand([1, 4, 5, 5, 8]) # NDHWK
    w = get_rand([3, 3, 3, 4, 8]) # VRSCK
    b = get_rand([4])             # C
    s          = [1, 2, 2, 2, 1]
    x_shape    = [1, 8, 9, 9, 4]  # NDHWC
    x   = tf.nn.conv3d_transpose(y, w, output_shape=x_shape, strides=s, padding='SAME')
    x_b = tf.nn.bias_add(x, b)
    x_a = tf.nn.elu(x_b).eval()
    write_bin(ndhwc_to_ncdhw(y),   os.path.join(data_dir, 'conv3d_tran_04_y.bin'))
    write_bin(vrsck_to_kvcrs(w),   os.path.join(data_dir, 'conv3d_tran_04_w.bin'))
    write_bin(ndhwc_to_ndchw(x_a), os.path.join(data_dir, 'conv3d_tran_04_x.bin'))
    write_bin(b,                   os.path.join(data_dir, 'conv3d_tran_04_b.bin'))

    # Testing DHW padding and strides in 2 subsequent deconvolutions.
    # Note: similar to deconv3D_1->deconv3D_2 convo in NVSmall.
    np.random.seed(1)
    # First convo^T.
    y  = get_rand([1,  4,  5,  5, 16]) # NDHWK
    w1 = get_rand([3,  3,  3,  8, 16]) # VRSCK
    s1          = [1,  2,  2,  2,  1]
    x1_shape    = [1,  8,  9,  9,  8]  # NDHWC
    # Second convo^T.
    w2 = get_rand([3,  3,  3,  4,  8]) # VRSCK
    s2          = [1,  2,  2,  2,  1]
    x2_shape    = [1, 16, 17, 17,  4]  # NDHWC
    x1 = tf.nn.conv3d_transpose(y,  w1, output_shape=x1_shape, strides=s1, padding='SAME')
    x2 = tf.nn.conv3d_transpose(x1, w2, output_shape=x2_shape, strides=s2, padding='SAME').eval()
    write_bin(ndhwc_to_ncdhw(y),  os.path.join(data_dir, 'conv3d_tran_05_y.bin'))
    write_bin(vrsck_to_kvcrs(w1), os.path.join(data_dir, 'conv3d_tran_05_w1.bin'))
    write_bin(vrsck_to_kvcrs(w2), os.path.join(data_dir, 'conv3d_tran_05_w2.bin'))
    write_bin(ndhwc_to_ndchw(x2), os.path.join(data_dir, 'conv3d_tran_05_x.bin'))

def create_cost_volume_data(data_dir):
    def cost_volume(left, right, max_disp):
        height = int(left.shape[1])
        width  = int(left.shape[2])
        depth  = int(left.shape[3])

        right_padded = tf.pad(right, [[0, 0], [0, 0], [max_disp - 1, 0], [0,0]], "CONSTANT")
        right_disp   = tf.extract_image_patches(right_padded, [1, height, width, 1], [1, 1, 1, 1], [1, 1, 1, 1], padding="VALID")
        right_disp   = tf.squeeze(right_disp, axis=1)
        disp_dim     = int(right_disp.shape[1])
        right_disp   = tf.reshape(right_disp, [-1, disp_dim, height, width, depth])
        right_disp   = tf.reverse(right_disp, [1])
        
        left_disp = tf.expand_dims(left, axis=1)
        left_disp = tf.tile(left_disp, [1, disp_dim, 1, 1, 1])
        
        cost_volume = tf.concat([left_disp, right_disp], axis=4)
    
        return cost_volume

    def corr_cost_volume_left(left, right, max_disp):
        height = int(left.shape[1])
        width  = int(left.shape[2])
        depth  = int(left.shape[3])

        right_padded = tf.pad(right, [[0, 0], [0, 0], [max_disp - 1, 0], [0,0]], "CONSTANT")
        right_disp   = tf.extract_image_patches(right_padded, [1, height, width, 1], [1, 1, 1, 1], [1, 1, 1, 1], padding="VALID")
        right_disp   = tf.squeeze(right_disp, axis=1)
        disp_dim     = int(right_disp.shape[1])
        right_disp   = tf.reshape(right_disp, [-1, disp_dim, height, width, depth])
        right_disp   = tf.reverse(right_disp, [1])
        
        left_disp = tf.expand_dims(left, axis=1)
        left_disp = tf.tile(left_disp, [1, disp_dim, 1, 1, 1])
        
        corr_cost_volume = tf.reduce_sum(tf.multiply(left_disp, right_disp), axis=4, keep_dims=True)
    
        return corr_cost_volume

    print("---")
    print("Creating data for CostVolume plugin...")
    print("---")

    # Basic test: 6x6x4 input, max_disp == 2, output is 2x6x6x8.
    np.random.seed(1)
    in_shape = [1, 6, 6, 4] # NHWC
    left  = get_rand(in_shape) 
    right = get_rand(in_shape)
    max_disp = 2
    cost_vol = cost_volume(left, right, max_disp).eval()
    write_bin(nhwc_to_nchw(left),       os.path.join(data_dir, 'cost_vol_01_l.bin'))
    write_bin(nhwc_to_nchw(right),      os.path.join(data_dir, 'cost_vol_01_r.bin'))
    write_bin(ndhwc_to_ndchw(cost_vol), os.path.join(data_dir, 'cost_vol_01_cv.bin'))

    # More realistic test: 33x32x8 input, max_disp == 8, output is 8x33x32x16.
    np.random.seed(1)
    in_shape = [1, 32, 33, 8] # NHWC
    #in_shape = [1, 128, 257, 32] # NHWC
    left  = get_rand(in_shape) 
    right = get_rand(in_shape)
    max_disp = 12
    #max_disp = 24
    cost_vol = cost_volume(left, right, max_disp).eval()
    write_bin(nhwc_to_nchw(left),       os.path.join(data_dir, 'cost_vol_02_l.bin'))
    write_bin(nhwc_to_nchw(right),      os.path.join(data_dir, 'cost_vol_02_r.bin'))
    write_bin(ndhwc_to_ndchw(cost_vol), os.path.join(data_dir, 'cost_vol_02_cv.bin'))

    # Correlation basic test: 6x6x4 input, max_disp == 2, output is 6x6x2.
    np.random.seed(1)
    in_shape = [1, 6, 6, 4] # NHWC
    left  = get_rand(in_shape) 
    right = get_rand(in_shape)
    max_disp = 2
    cost_vol = corr_cost_volume_left(left, right, max_disp).eval()
    write_bin(nhwc_to_nchw(left),       os.path.join(data_dir, 'corr_cost_vol_01_l.bin'))
    write_bin(nhwc_to_nchw(right),      os.path.join(data_dir, 'corr_cost_vol_01_r.bin'))
    write_bin(ndhwc_to_ndchw(cost_vol), os.path.join(data_dir, 'corr_cost_vol_01_cv.bin'))

def create_softargmax_data(data_dir):
    def softargmin(volume):
        input_depth = int(volume.shape[1])
        index_steps = tf.constant(np.reshape(np.array(range(input_depth)),(1,input_depth,1,1,1)), dtype=tf.float32)
        prob_volume = tf.nn.softmax(tf.multiply(volume, -1.0), dim=1)
        softargmax_result = tf.reduce_sum(tf.multiply(prob_volume, index_steps), axis=1)
    
        return softargmax_result
        
    def softargmax(volume):
        input_depth = int(volume.shape[1])
        index_steps = tf.constant(np.reshape(np.array(range(input_depth)),(1,input_depth,1,1,1)), dtype=tf.float32)
        prob_volume = tf.nn.softmax(volume, dim=1)
        softargmax_result = tf.reduce_sum(tf.multiply(prob_volume, index_steps), axis=1)
            
        return softargmax_result
        
    print("---")
    print("Creating data for Softargmax plugin...")
    print("---")

    # Basic argmin test: 4x5x5x1 input, output is 5x5x1.
    np.random.seed(1)
    x = get_rand([1, 4, 5, 7, 1]) # NDHWC
    y = softargmin(x).eval()
    write_bin(ndhwc_to_ndchw(x), os.path.join(data_dir, 'softargmax_01_x.bin'))
    write_bin(nhwc_to_nchw(y),   os.path.join(data_dir, 'softargmax_01_y.bin'))

    # Large argmin test: 2x12x33x65x1 input, output is 2x33x65x1.
    np.random.seed(1)
    x = get_rand([2, 12, 33, 65, 1]) # NDHWC
    y = softargmin(x).eval()
    write_bin(ndhwc_to_ndchw(x), os.path.join(data_dir, 'softargmax_02_x.bin'))
    write_bin(nhwc_to_nchw(y),   os.path.join(data_dir, 'softargmax_02_y.bin'))

    # Basic argmax test: 4x5x5x1 input, output is 5x5x1.
    np.random.seed(1)
    x = get_rand([1, 4, 5, 7, 1]) # NDHWC
    y = softargmax(x).eval()
    write_bin(ndhwc_to_ndchw(x), os.path.join(data_dir, 'softargmax_03_x.bin'))
    write_bin(nhwc_to_nchw(y),   os.path.join(data_dir, 'softargmax_03_y.bin'))

def main():
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    create_elu_plugin_data(args.data_dir)
    create_conv3d_plugin_data(args.data_dir)
    create_conv3d_tran_plugin_data(args.data_dir)
    create_cost_volume_data(args.data_dir)
    create_softargmax_data(args.data_dir)

    print("Done.")

if __name__ == '__main__':
    main()
