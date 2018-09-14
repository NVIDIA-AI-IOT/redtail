# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

"""
Generates TensorRT code for NVSmall model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# ResNet-18
# python ./model_builder.py --model_type resnet18 --net_name ResNet18_1025x321 --checkpoint_path=../models/ResNet-18/TensorFlow/model-inference-1025x321-0 --weights_file=../models/ResNet-18/TensorRT/trt_weights.bin --cpp_file=../sample_app/resnet18_1025x321_net.cpp

def write_residual_block(input, name, path, builder):
    cur = builder.write_2d_convolution(input, name + '_conv1', os.path.join(path, 'res_conv1'))
    cur = builder.write_act(cur, cur + '_act')
    cur = builder.write_2d_convolution(cur, name + '_conv2', os.path.join(path, 'res_conv2'))
    cur = builder.write_add_tensors(cur, input, cur + '_add')
    cur = builder.write_act(cur, cur + '_act')
    return cur

def write_2d_encoder(builder):
    sides = ['left', 'right']
    builder.write_input(sides[0])
    builder.write_input(sides[1])
    layer_inp = [sides[0] + '_scale', sides[1] + '_scale']
    builder.write_scale(sides[0], layer_inp[0])
    builder.write_scale(sides[1], layer_inp[1])
    # conv1
    for i in range(len(sides)):
        cur = builder.write_2d_convolution(layer_inp[i], '{}_{}'.format(sides[i], 'conv1'), os.path.join('model/encoder2D', 'conv1'))
        cur = builder.write_act(cur, cur + '_act')
        layer_inp[i] = cur
    # resblock 1 - 8
    for l in ['resblock1', 'resblock2', 'resblock3', 'resblock4', 'resblock5', 'resblock6', 'resblock7', 'resblock8']:
        for i in range(len(sides)):
            cur = '{}_{}'.format(sides[i], l)
            layer_inp[i] =  write_residual_block(layer_inp[i], cur, os.path.join('model/encoder2D', l), builder)
    # encoder2D_out
    left  = builder.write_2d_convolution(layer_inp[0], sides[0] + '_encoder2D_out', 'model/encoder2D/encoder2D_out')
    right = builder.write_2d_convolution(layer_inp[1], sides[1] + '_encoder2D_out', 'model/encoder2D/encoder2D_out')
    return left, right

def create(builder):
    def write_3d_encoder(input):
        input = 'cost_vol'
        for l in ['conv3D_1a', 'conv3D_1b', 'conv3D_1ds',
                  'conv3D_2a', 'conv3D_2b', 'conv3D_2ds',
                  'conv3D_3a', 'conv3D_3b', 'conv3D_3ds',
                  'conv3D_4a', 'conv3D_4b', 'conv3D_4ds',
                  'conv3D_5a', 'conv3D_5b']:
            # Pad after each D stride == 2.
            if l in ['conv3D_1ds', 'conv3D_2ds', 'conv3D_3ds', 'conv3D_4ds']:
                input = builder.write_conv3d_pad(input, l + '_pad')
            input = builder.write_3d_convolution(input, l, 'model/encoder3D')
            # No transpose for conv3D_5b as it goes directly to decoder.
            if l != 'conv3D_5b':
                input = builder.write_conv3d_transform(input, l + '_tran')
            input = builder.write_act(input, l + '_act')
        return input

    def write_3d_decoder(input):
        cur = input
        for i in range(1, 5):
            l = 'deconv3D_{}'.format(i)
            cur = builder.write_3d_convolution_transpose(cur, l, 'model/decoder3D')
            cur = builder.write_add_tensors(cur, 'conv3D_{}b_act'.format(5 - i), l + '_add_skip')
            cur = builder.write_act(cur, l + '_act')
            cur = builder.write_conv3d_transform(cur, l + '_transform')
        # deconv3D_5
        cur  = builder.write_3d_convolution_transpose(cur, 'deconv3D_5', 'model/decoder3D')
        return cur

    builder.write_header()
    builder.do_indent()
    left, right = write_2d_encoder(builder)
    cur = builder.write_cost_vol(left, right, 'cost_vol', 'model/cost_vol/cost_volume_left')
    cur = write_3d_encoder(cur)
    cur = write_3d_decoder(cur)
    # Softargmax
    cur = builder.write_softargmax(cur, 'disp', is_argmin=True)
    builder.write_output(cur)
    builder.write_footer()
