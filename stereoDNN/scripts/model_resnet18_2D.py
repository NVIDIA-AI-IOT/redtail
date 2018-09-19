# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

"""
Generates TensorRT code for NVSmall model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import model_resnet18

# Create inference model:
# python ./model_builder.py --model_type resnet18_2D --net_name ResNet18_2D_513x257 --checkpoint_path=../models/ResNet-18_2D/TensorFlow/model-inference-513x257-0 --weights_file=../models/ResNet-18_2D/TensorRT/trt_weights.bin --cpp_file=../sample_app/resnet18_2D_513x257_net.cpp
def create(builder):
    def write_2d_bneck_encoder(input):
        input = 'concat'
        for l in ['conv2D_1', 'conv2D_2', 'conv2D_3ds', 'conv2D_4', 'conv2D_5', 'conv2D_6ds', 'conv2D_7', 'conv2D_8']:
            input = builder.write_2d_convolution(input, l, os.path.join('model/bneck_encoder2D', l))
            input = builder.write_act(input, l + '_act')
        return input

    def write_2d_bneck_decoder(input):
        cur = input
        for item in [('deconv2D_1', 'conv2D_5_act'), ('deconv2D_2', 'conv2D_2_act'), ('deconv2D_3', '')]:
            l, skip = item
            cur  = builder.write_2d_convolution_transpose(cur, l, os.path.join('model/bneck_decoder2D', l))
            if skip != '':
                cur  = builder.write_add_tensors(cur, skip, l + '_add_skip')
                cur  = builder.write_act(cur, l + '_act')
        return cur

    builder.write_header()
    builder.do_indent()
    left, right = model_resnet18.write_2d_encoder(builder)
    cur = builder.write_cost_vol(left, right, 'cost_vol', 'model/cost_vol/cost_volume_left', is_corr=True)
    # Softargmax
    cur = builder.write_softargmax(cur, 'softargmax', is_argmin=False)
    # Concat with features.
    cur = builder.write_concat_tensors('left_conv1_act', cur, 'concat')
    cur = write_2d_bneck_encoder(cur)
    cur = write_2d_bneck_decoder(cur)
    cur = builder.write_sigmoid(cur, 'disp')
    builder.write_output(cur)
    builder.write_footer()
