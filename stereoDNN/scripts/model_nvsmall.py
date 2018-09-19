# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

"""
Generates TensorRT code for NVSmall model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Create inference model:
# NVSmall:
# python ./model_builder.py --model_type nvsmall --net_name NVTiny513x161 --checkpoint_path=../models/NVTiny/TensorFlow/model-inference-513x161-0 --weights_file=../models/NVTiny/TensorRT/trt_weights.bin --cpp_file=../sample_app/nvtiny_513x161_net.cpp
# NVTiny:
# python ./model_builder.py --model_type nvsmall --net_name NVTiny513x161 --checkpoint_path=../models/NVTiny/TensorFlow/model-inference-513x161-0 --weights_file=../models/NVTiny/TensorRT/trt_weights.bin --cpp_file=../sample_app/nvtiny_513x161_net.cpp
def create(builder):
    def write_2d_encoder():
        sides = ['left', 'right']
        builder.write_input(sides[0])
        builder.write_input(sides[1])
        layer_inp = [sides[0] + '_scale', sides[1] + '_scale']
        builder.write_scale(sides[0], layer_inp[0])
        builder.write_scale(sides[1], layer_inp[1])
        for l in ['conv1', 'conv2', 'conv3', 'conv4']:
            for i in range(len(sides)):
                cur = '{}_{}'.format(sides[i], l)
                builder.write_2d_convolution(layer_inp[i], cur, os.path.join('model/encoder2D', l))
                builder.write_act(cur, cur + '_act')
                layer_inp[i] = cur + '_act'
        left  = builder.write_2d_convolution(layer_inp[0], sides[0] + '_conv5', 'model/encoder2D/conv5')
        right = builder.write_2d_convolution(layer_inp[1], sides[1] + '_conv5', 'model/encoder2D/conv5')
        return left, right

    def write_3d_encoder(input):
        input = 'cost_vol'
        for l in ['conv3D_1', 'conv3D_2', 'conv3D_3ds', 'conv3D_4', 'conv3D_5', 'conv3D_6ds', 'conv3D_7', 'conv3D_8']:
            # Pad after each D stride == 2.
            if l in ['conv3D_3ds', 'conv3D_6ds']:
                input = builder.write_conv3d_pad(input, l + '_pad')
            input = builder.write_3d_convolution(input, l, 'model/encoder3D')
            # No transpose for conv3D_8 as it goes directly to decoder.
            if l != 'conv3D_8':
                input = builder.write_conv3d_transform(input, l + '_tran')
            input = builder.write_act(input, l + '_act')
        return input

    def write_3d_decoder(input):
        # deconv3D_1
        cur  = builder.write_3d_convolution_transpose(input, 'deconv3D_1', 'model/decoder3D')
        cur  = builder.write_add_tensors(cur, 'conv3D_5_act', 'deconv3D_1_add_skip')
        cur  = builder.write_act(cur, 'deconv3D_1_act')
        # deconv3D_2
        cur  = builder.write_conv3d_transform(cur, 'deconv3D_1_transform')
        cur  = builder.write_3d_convolution_transpose(cur, 'deconv3D_2', 'model/decoder3D')
        cur  = builder.write_add_tensors(cur, 'conv3D_2_act', 'deconv3D_2_add_skip')
        cur  = builder.write_act(cur, 'deconv3D_2_act')
        # deconv3D_3
        cur  = builder.write_conv3d_transform(cur, 'deconv3D_2_transform')
        cur  = builder.write_3d_convolution_transpose(cur, 'deconv3D_3', 'model/decoder3D')
        return cur

    builder.write_header()
    builder.do_indent()
    left, right = write_2d_encoder()
    cur = builder.write_cost_vol(left, right, 'cost_vol', 'model/cost_vol/cost_volume_left')
    cur = write_3d_encoder(cur)
    cur = write_3d_decoder(cur)
    # Softargmax
    cur = builder.write_softargmax(cur, 'disp', is_argmin=True)
    builder.write_output(cur)
    builder.write_footer()
