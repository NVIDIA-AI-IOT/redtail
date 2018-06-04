# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

"""
Generates TensorRT C++ API code from TensorFlow model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import warnings
# Ignore 'FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated' warning.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning, module='h5py')
    import tensorflow as tf

import tensorrt_model_builder
import model_nvsmall
import model_resnet18
import model_resnet18_2D

def check_model_type(src):
    supported_types = ['nvsmall', 'resnet18', 'resnet18_2D']
    if src in supported_types:
        return src
    else:
        raise argparse.ArgumentTypeError('Invalid model type {}. Supported: {}'.format(src, ', '.join(supported_types)))

def check_data_type(src):
    if src == 'fp32' or src == 'fp16':
        return src
    else:
        raise argparse.ArgumentTypeError('Invalid data type {}. Supported: fp32, fp16'.format(src))

parser = argparse.ArgumentParser(description='Stereo DNN TensorRT C++ code generator')

parser.add_argument('--model_type',      type=check_model_type, help='model type, currently supported: nvsmall', required=True)
parser.add_argument('--net_name',        type=str, help='network name to use in C++ code generation',  required=True)
parser.add_argument('--checkpoint_path', type=str, help='path to checkpoint file (without extension)', required=True)
parser.add_argument('--weights_file',    type=str, help='path to generated weights file',              required=True)
parser.add_argument('--cpp_file',        type=str, help='path to generated TensorRT C++ model file',   required=True)
parser.add_argument('--data_type',       type=check_data_type, help='model data type, supported: fp32, fp16', default='fp32')

args = parser.parse_args()

def read_model(model_path, session):
    print('Reading model...')
    saver = tf.train.import_meta_graph(model_path + '.meta')
    print('Loaded graph meta.')
    saver.restore(session, model_path)
    print('Loaded weights.')
    print('Done reading model.')

def main():
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    model = read_model(args.checkpoint_path, sess)

    with open(args.weights_file, 'wb') as weights_w:
        with open(args.cpp_file, 'w') as cpp_w:
            builder = tensorrt_model_builder.TrtModelBuilder(model, args.net_name, cpp_w, weights_w, args.data_type)
            if args.model_type == 'nvsmall':
                model_nvsmall.create(builder)
            elif args.model_type == 'resnet18':
                model_resnet18.create(builder)
            elif args.model_type == 'resnet18_2D':
                model_resnet18_2D.create(builder)
            else:
                # Should never happen, yeah.
                assert False, 'Not supported.'
    print('Done.')

if __name__ == '__main__':
    main()
