# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

"""
Generates TensorRT C++ API code from TensorFlow model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import textwrap as tw
import time
import struct
import sys

import warnings
# Ignore 'FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated' warning.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning, module='h5py')
    import tensorflow as tf

from data_converters import *

class TrtModelBuilder(object):
    def __init__(self, model, net_name, code_writer, weights_writer, data_type, act='elu'):
        self.default_indent = 4
        self.cur_indent     = 0
        self.max_line_width = 160
        self.indent = tw.TextWrapper(initial_indent=' '*self.cur_indent,
                                     subsequent_indent=' '*(self.cur_indent + self.default_indent),
                                     width=self.max_line_width, break_long_words=False)
        self.model          = model 
        self.net_name       = net_name
        self.code_writer    = code_writer
        self.weights_writer = weights_writer
        self.data_type      = data_type
        self.act            = act
        self.has_srelu_weights = False  

    def _indent_lines(self, src):
        src = src.split('\n')
        for i in range(len(src)):
            src[i] = self.indent.fill(src[i])
        return '\n'.join(src)

    def do_indent(self):
        self.cur_indent = self.cur_indent + self.default_indent
        self.indent.initial_indent = ' '*self.cur_indent

    def _write_weights(self, name, src):
        # Write name as null-terminated string.
        self.weights_writer.write(struct.pack('%ds'%len(name), name.encode()))
        self.weights_writer.write(struct.pack('B', 0))
        # Weight count and data.
        src_flat = np.reshape(src, -1)
        src_flat = src_flat.astype(np.float16) if self.data_type == 'fp16' else src_flat.astype(np.float32)
        self.weights_writer.write(struct.pack('<I', len(src_flat)))
        src_flat.tofile(self.weights_writer)

    def write_header(self):
        code = """\
// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

//-------------------------------------------------------------------
// !!! This file was automatically generated. Do not edit. !!!
//-------------------------------------------------------------------

#include <NvInfer.h>
#include <cassert>
#include <string>
#include <unordered_map>
#include "redtail_tensorrt_plugins.h"

namespace redtail {{ namespace tensorrt
{{

using namespace nvinfer1;

using weight_map = std::unordered_map<std::string, Weights>;

INetworkDefinition* create{0}Network(IBuilder& builder, IPluginContainer& plugin_factory,
                                     DimsCHW img_dims, const weight_map& weights, DataType data_type,
                                     ILogger& log)
{{
    INetworkDefinition* network = builder.createNetwork();
    assert(network != nullptr);
"""
        self.code_writer.write(code.format(self.net_name))

    def write_footer(self):
        code = """\
    return network;
}

} } // namespace
"""
        self.code_writer.write(code)

    def write_input(self, name):
        code = """\
// Input tensor.
auto {0} = network->addInput("{0}", DataType::kFLOAT, img_dims);
assert({0} != nullptr);

"""
        code = code.format(name)
        self.code_writer.write(self._indent_lines(code))

    def write_output(self, output):
        code = """\
auto {0}_out = {0}->getOutput(0);
{0}_out->setName("{0}");
network->markOutput(*{0}_out);

    """
        self.code_writer.write(self._indent_lines(code.format(output)))

    def write_scale(self, input, name):
        # Write code.
        code = """\
// Scaling op.
auto {1} = network->addScale(*{0}, ScaleMode::kUNIFORM,
                             weights.at("{1}_shift"), weights.at("{1}_scale"), weights.at("{1}_power"));
assert({1} != nullptr);
{1}->setName("{1}");

"""
        code = code.format(input, name)
        self.code_writer.write(self._indent_lines(code))
        # REVIEW alexeyk: default for now.
        self._write_weights(name + '_shift', [0.0])
        self._write_weights(name + '_scale', [1.0])
        self._write_weights(name + '_power', [1.0])

    # TensorFlow padding computation, taken from:
    # https://www.tensorflow.org/api_guides/python/nn#Convolution
    def _compute_tf_padding(self, in_dim, kern_dim, stride_dim):
        if in_dim % stride_dim == 0:
            pad_along = max(kern_dim - stride_dim, 0)
        else:
            pad_along = max(kern_dim - (in_dim % stride_dim), 0)
        pad_start = pad_along // 2
        pad_end   = pad_along - pad_start
        return pad_start, pad_end

    def write_2d_convolution(self, input, name, op_path):
        # Convolution code.
        conv_code = """\
// {1} convolution op.
auto {1} = network->addConvolution(*{0}->getOutput(0), {2}, DimsHW {{{3}, {4}}},
                                   weights.at("{1}_k"), weights.at("{1}_b"));
assert({1} != nullptr);
{1}->setName("{1}");
{1}->setStride( DimsHW {{{5}, {6}}});
{1}->setPadding(DimsHW {{{7}, {8}}});

"""
        # Padding code.
        pad_code = """\
// {1} padding op.
auto {1}_pad = network->addPadding(*{0}->getOutput(0), DimsHW {{{2}, {3}}}, DimsHW {{{4}, {5}}});
assert({1}_pad != nullptr);

"""
        g = tf.get_default_graph()
        assert g is not None, "No default graph set"
        # Get and check convolution operation.
        conv_op = g.get_operation_by_name(os.path.join(op_path, 'Conv2D'))
        # Assuming input/output of 2D convolution are in NHWC format.
        assert conv_op.type == 'Conv2D', 'Expected Conv2D operation but got {}.'.format(conv_op.type)
        assert conv_op.get_attr('data_format').decode() == 'NHWC', 'Only NHWC format is currently supported for 2D convolutions.'
        assert len(conv_op.inputs)  == 2, 'Convolution expected to have 2 inputs.'
        assert len(conv_op.outputs) == 1, 'Convolution expected to have only one output.'
        # Get and check bias operation
        bias_op = g.get_operation_by_name(os.path.join(op_path, 'BiasAdd'))
        assert bias_op.type == 'BiasAdd', 'Expected BiasAdd operation but got {}.'.format(bias_op.type)
        assert bias_op.get_attr('data_format').decode() == 'NHWC', 'Only NHWC format is currently supported for 2D convolution biases.'
        assert len(bias_op.inputs)  == 2, 'BiasAdd expected to have 2 inputs.'
        assert len(bias_op.outputs) == 1, 'BiasAdd expected to have only one output.'

        # Get weights, stride and padding.
        kernel_weights = conv_op.inputs[1].eval()
        kernel_shape   = kernel_weights.shape
        assert len(kernel_shape) == 4, 'Convolution kernel weights tensor expected to have rank 4.'
        # Weights in TF are in RSCK format (in TensorRT speak).
        kh, kw, kc, kk = kernel_shape
        strides = conv_op.get_attr('strides')
        assert len(strides) == 4, 'Convolution strides tensor expected to have length 4.'
        # Compute padding.
        in_shape  = conv_op.inputs[0].shape.as_list()
        out_shape = conv_op.outputs[0].shape.as_list()
        pad_top,  pad_bottom = self._compute_tf_padding(in_shape[1], kh, strides[1])
        pad_left, pad_right  = self._compute_tf_padding(in_shape[2], kw, strides[2])
        # If padding is symetrical - use TensorRT convolution padding
        # otherwise need to explicitly pad the input.
        trt_conv_pad_h = 0
        trt_conv_pad_w = 0
        code = ''
        conv_input = input
        if pad_top == pad_bottom and pad_left == pad_right:
            trt_conv_pad_h = pad_top
            trt_conv_pad_w = pad_left
        else:
            assert False, 'Not supported at the moment due to bug (#3199) in TRT.'
            p_b = pad_bottom - pad_top
            p_r = pad_right  - pad_left
            assert 0 <= p_b and p_b <= 1, 'Bottom pad should not be greater than top pad by more than 1.'
            assert 0 <= p_r and p_r <= 1, 'Right pad should not be greater than left pad by more than 1.'
            # Write padding layer.
            code = pad_code.format(input, name, pad_top, pad_left, pad_bottom, pad_right)
            conv_input = name + '_pad'
        # Write code.
        code += conv_code.format(conv_input, name, kk, kh, kw, strides[1], strides[2], trt_conv_pad_h, trt_conv_pad_w)
        self.code_writer.write(self._indent_lines(code))
        # TRT requires kernel weights to be in KCRS format while TensorFlow uses RSCK.
        kernel_weights = rsck_to_kcrs(kernel_weights)
        # Write kernel weights.
        self._write_weights(name + '_k', kernel_weights)
        # Write bias weights.
        bias_weights = bias_op.inputs[1].eval()
        bias_shape   = bias_weights.shape
        assert len(bias_shape) == 1, 'Convolution bias weights tensor expected to have rank 1.'
        assert bias_shape[0] == kernel_shape[3], 'Convolution bias size does not match convolution output channels.'
        self._write_weights(name + '_b', bias_weights)
        return name

    def write_2d_convolution_transpose(self, input, name, op_path):
        # Convolution code.
        code = """\
// {1} transposed convolution op.
auto {1} = network->addDeconvolution(*{0}->getOutput(0), {2}, DimsHW {{{3}, {4}}},
                                     weights.at("{1}_k"), weights.at("{1}_b"));
assert({1} != nullptr);
{1}->setName("{1}");
{1}->setStride( DimsHW {{{5}, {6}}});
{1}->setPadding(DimsHW {{{7}, {8}}});

"""
        g = tf.get_default_graph()
        assert g is not None, "No default graph set"
        # Get and check transposed convolution operation.
        conv_op = g.get_operation_by_name(os.path.join(op_path, 'conv2d_transpose'))
        # Assuming input/output of 2D convolution are in NHWC format.
        assert conv_op.type == 'Conv2DBackpropInput', 'Expected Conv2DBackpropInput operation but got {}.'.format(conv_op.type)
        assert conv_op.get_attr('data_format').decode() == 'NHWC', 'Only NHWC format is currently supported for 2D transposed convolutions.'
        assert len(conv_op.inputs)  == 3, 'Transposed convolution expected to have 3 inputs.'
        assert len(conv_op.outputs) == 1, 'Transposed convolution expected to have only one output.'
        # Get and check bias operation
        bias_op = g.get_operation_by_name(os.path.join(op_path, 'BiasAdd'))
        assert bias_op.type == 'BiasAdd', 'Expected BiasAdd operation but got {}.'.format(bias_op.type)
        assert bias_op.get_attr('data_format').decode() == 'NHWC', 'Only NHWC format is currently supported for 2D transposed convolution biases.'
        assert len(bias_op.inputs)  == 2, 'BiasAdd expected to have 2 inputs.'
        assert len(bias_op.outputs) == 1, 'BiasAdd expected to have only one output.'

        # Get weights, stride and padding.
        kernel_weights = conv_op.inputs[1].eval()
        kernel_shape   = kernel_weights.shape
        assert len(kernel_shape) == 4, '2D transposed convolution kernel weights tensor expected to have rank 4.'
        # Weights in TF are in RSCK format.
        kh, kw, kc, kk = kernel_shape
        strides = conv_op.get_attr('strides')
        assert len(strides) == 4, '2D transposed convolution strides tensor expected to have length 4.'
        # Compute padding. 
        # Note that padding is with respect to convolution input, that is transposed convolution output.
        conv_in_shape  = np.array(conv_op.outputs[0].shape.as_list())
        pad_h_start, pad_h_end = self._compute_tf_padding(conv_in_shape[1], kh, strides[1])
        pad_w_start, pad_w_end = self._compute_tf_padding(conv_in_shape[2], kw, strides[2])
        # cuDNN limitations...
        assert pad_h_start == pad_h_end, 'Only symmetrical padding is currently supported for H dimension.'
        assert pad_w_start == pad_w_end, 'Only symmetrical padding is currently supported for W dimension.'
        # Write code.
        code = code.format(input, name, kc, kh, kw, strides[1], strides[2], pad_h_start, pad_w_start)
        self.code_writer.write(self._indent_lines(code))
        # Convert and write weights.
        kernel_weights = rsck_to_kcrs(kernel_weights)
        # Write kernel weights.
        self._write_weights(name + '_k', kernel_weights)
        # Write bias weights.
        bias_weights = bias_op.inputs[1].eval()
        bias_shape   = bias_weights.shape
        assert len(bias_shape) == 1, '2D transposed convolution bias weights tensor expected to have rank 1.'
        assert bias_shape[0] == kc, '2D transposed convolution bias size does not match convolution input channels.'
        self._write_weights(name + '_b', bias_weights)
        
        return name

    def write_3d_convolution(self, input, name, op_path):
        # Write code.
        code = """\
// {1} 3D convolution op.
auto {1} = addConv3D(plugin_factory, *network, *{0}->getOutput(0),
                     Conv3DType::kTensorFlow, {{5, {{{k_dims}}}}},
                     Dims{{3, {{{stride_dims}}}}}, Dims{{3, {{{pad_start_dims}}}}}, Dims{{3, {{{pad_end_dims}}}}},
                     weights.at("{1}_k"), weights.at("{1}_b"),
                     "{1}");
assert({1} != nullptr);
{1}->setName("{1}");

"""
        g = tf.get_default_graph()
        assert g is not None, "No default graph set"
        # Get and check convolution operation.
        conv_op = g.get_operation_by_name(os.path.join(op_path, os.path.join(name, 'Conv3D')))
        # Assuming input/output of 3D convolution are in NHWC format.
        assert conv_op.type == 'Conv3D', 'Expected Conv3D operation but got {}.'.format(conv_op.type)
        assert conv_op.get_attr('data_format').decode() == 'NDHWC', 'Only NDHWC format is currently supported for 3D convolutions.'
        assert len(conv_op.inputs)  == 2, 'Convolution expected to have 2 inputs.'
        assert len(conv_op.outputs) == 1, 'Convolution expected to have only one output.'
        # Get and check bias operation
        bias_op = g.get_operation_by_name(os.path.join(op_path, os.path.join(name, 'BiasAdd')))
        assert bias_op.type == 'BiasAdd', 'Expected BiasAdd operation but got {}.'.format(bias_op.type)
        # REVIEW alexeyk: is this a bug in our model code? Should the bias tensor be NDCHW dim as well?
        assert bias_op.get_attr('data_format').decode() == 'NHWC', 'Only NHWC format is currently supported for 3D convolution biases.'
        assert len(bias_op.inputs)  == 2, 'BiasAdd expected to have 2 inputs.'
        assert len(bias_op.outputs) == 1, 'BiasAdd expected to have only one output.'

        # Get weights, stride and padding.
        kernel_weights = conv_op.inputs[1].eval()
        kernel_shape   = kernel_weights.shape
        assert len(kernel_shape) == 5, '3D convolution kernel weights tensor expected to have rank 5.'
        # Weights in TF are in VRSCK format (in TensorRT speak).
        kd, kh, kw, kc, kk = kernel_shape
        strides = conv_op.get_attr('strides')
        assert len(strides) == 5, '3D convolution strides tensor expected to have length 5.'
        # Compute padding.
        in_shape  = conv_op.inputs[0].shape.as_list()
        out_shape = conv_op.outputs[0].shape.as_list()
        pad_c_start, pad_c_end = self._compute_tf_padding(in_shape[1], kd, strides[1])
        pad_h_start, pad_h_end = self._compute_tf_padding(in_shape[2], kh, strides[2])
        pad_w_start, pad_w_end = self._compute_tf_padding(in_shape[3], kw, strides[3])
        # cuDNN limitations...
        assert pad_h_start == pad_h_end, 'Only symmetrical padding is currently supported for H dimension.'
        assert pad_w_start == pad_w_end, 'Only symmetrical padding is currently supported for W dimension.'
        # REVIEW alexeyk: C padding is done by padding plugin now, not by conv3d plugin.
        p_c = pad_c_end - pad_c_start
        assert 0 <= p_c and p_c <= 1, 'Depth end pad should not be greater than start pad by more than 1.'

        code = code.format(input, name,
                           k_dims         ='{}, {}, {}, {}, {}'.format(kk, kd, kc, kh, kw),
                           stride_dims    ='{}, {}, {}'.format(strides[1], strides[2], strides[3]),
                           pad_start_dims ='{}, {}, {}'.format(pad_c_start, pad_h_start, pad_w_start),
                           pad_end_dims   ='{}, {}, {}'.format(pad_c_end, pad_h_end, pad_w_end))
        self.code_writer.write(self._indent_lines(code))
        # TRT requires kernel weights to be in KVCRS format while TensorFlow uses VRSCK.
        kernel_weights = vrsck_to_kvcrs(kernel_weights)
        # Write kernel weights.
        self._write_weights(name + '_k', kernel_weights)
        # Write bias weights.
        bias_weights = bias_op.inputs[1].eval()
        bias_shape   = bias_weights.shape
        assert len(bias_shape) == 1, '3D convolution bias weights tensor expected to have rank 1.'
        # REVIEW alexeyk: should really assert against D?
        assert bias_shape[0] == kernel_shape[4], '3D convolution bias size does not match convolution output channels.'
        self._write_weights(name + '_b', bias_weights)
        return name

    def write_3d_convolution_transpose(self, input, name, op_path):
        # Write code.
        code = """\
// {1} 3D transposed convolution op.
Dims {1}_out_dims{{4, {{{out_dims}}}}};
auto {1} = addConv3DTranspose(plugin_factory, *network, *{0}->getOutput(0),
                              Conv3DType::kTensorFlow, {{5, {{{k_dims}}}}}, {1}_out_dims,
                              Dims{{3, {{{stride_dims}}}}}, Dims{{3, {{{pad_start_dims}}}}}, Dims{{3, {{{pad_end_dims}}}}},
                              weights.at("{1}_k"), weights.at("{1}_b"),
                              "{1}");
assert({1} != nullptr);
{1}->setName("{1}");

"""
        slice_code = """\
// {0} output slice op.
auto {0}_slice_layer = addSlice(plugin_factory, *network, *{0}->getOutput(0),
                                {0}_out_dims,
                                {{4, {{0, 0, 0, 0}}}},
                                {{4, {{{0}_out_dims.d[0] - 1, {0}_out_dims.d[1], {0}_out_dims.d[2], {0}_out_dims.d[3]}}}},
                                "{0}_slice");
assert({0}_slice_layer != nullptr);
{0}_slice_layer->setName("{0}_slice_layer");

        """
        g = tf.get_default_graph()
        assert g is not None, "No default graph set"
        # Get and check tran convolution operation.
        conv_op = g.get_operation_by_name(os.path.join(op_path, os.path.join(name, 'conv3d_transpose')))
        # Assuming input/output of 3D convolution are in NHWC format.
        assert conv_op.type == 'Conv3DBackpropInputV2', 'Expected Conv3DBackpropInputV2 operation but got {}.'.format(conv_op.type)
        assert conv_op.get_attr('data_format').decode() == 'NDHWC', 'Only NDHWC format is currently supported for 3D transposed convolutions.'
        assert len(conv_op.inputs)  == 3, 'Transposed convolution expected to have 3 inputs.'
        assert len(conv_op.outputs) == 1, 'Transposed convolution expected to have only one output.'
        # Get and check bias operation
        bias_op = g.get_operation_by_name(os.path.join(op_path, os.path.join(name, 'BiasAdd')))
        assert bias_op.type == 'BiasAdd', 'Expected BiasAdd operation but got {}.'.format(bias_op.type)
        # REVIEW alexeyk: is this a bug in our model code? Should the bias tensor be NDCHW dim as well?
        assert bias_op.get_attr('data_format').decode() == 'NHWC', 'Only NHWC format is currently supported for 3D transposed convolution biases.'
        assert len(bias_op.inputs)  == 2, 'BiasAdd expected to have 2 inputs.'
        assert len(bias_op.outputs) == 1, 'BiasAdd expected to have only one output.'

        # Get weights, stride and padding.
        kernel_weights = conv_op.inputs[1].eval()
        kernel_shape   = kernel_weights.shape
        assert len(kernel_shape) == 5, '3D transposed convolution kernel weights tensor expected to have rank 5.'
        # Weights in TF are in VRSCK format.
        kd, kh, kw, kc, kk = kernel_shape
        strides = conv_op.get_attr('strides')
        assert len(strides) == 5, '3D transposed convolution strides tensor expected to have length 5.'
        # Compute padding. 
        # Note that padding is with respect to convolution input, that is transposed convolution output.
        conv_out_shape = conv_op.inputs[0].shape.as_list()
        conv_in_shape  = np.array(conv_op.outputs[0].shape.as_list())
        pad_c_start, pad_c_end = self._compute_tf_padding(conv_in_shape[1], kd, strides[1])
        pad_h_start, pad_h_end = self._compute_tf_padding(conv_in_shape[2], kh, strides[2])
        pad_w_start, pad_w_end = self._compute_tf_padding(conv_in_shape[3], kw, strides[3])
        # cuDNN limitations...
        assert pad_h_start == pad_h_end, 'Only symmetrical padding is currently supported for H dimension.'
        assert pad_w_start == pad_w_end, 'Only symmetrical padding is currently supported for W dimension.'
        p_c = pad_c_end - pad_c_start
        assert 0 <= p_c and p_c <= 1, 'Depth end pad should not be greater than start pad by more than 1.'
        # This is a special case for 3D transposed convolution:
        # Do not pad in C dimension in cuDNN but rather increase corresponding output (i.e. convo input) dimension.
        # In such configuration, this transposed convolution should be followed by slicing plugin.
        if pad_c_end != pad_c_start:
            conv_in_shape[1] += 1
            pad_c_start       = 0
            pad_c_end         = 0
        code = code.format(input, name,
                           k_dims         ='{}, {}, {}, {}, {}'.format(kk, kd, kc, kh, kw),
                           out_dims       ='{}, {}, {}, {}'.format(*(conv_in_shape[1:])[[0, 3, 1, 2]]),
                           stride_dims    ='{}, {}, {}'.format(strides[1], strides[2], strides[3]),
                           pad_start_dims ='{}, {}, {}'.format(pad_c_start, pad_h_start, pad_w_start),
                           pad_end_dims   ='{}, {}, {}'.format(pad_c_end,   pad_h_end,   pad_w_end))
        self.code_writer.write(self._indent_lines(code))
        out_name = name
        # Write slice layer in case of asymmetric C padding.
        if p_c != 0:
            self.code_writer.write(self._indent_lines(slice_code.format(name)))
            out_name += '_slice_layer'
        # TRT requires kernel weights to be in KVCRS format while TensorFlow uses VRSCK.
        kernel_weights = vrsck_to_kvcrs(kernel_weights)
        # Write kernel weights.
        self._write_weights(name + '_k', kernel_weights)
        # Write bias weights.
        bias_weights = bias_op.inputs[1].eval()
        bias_shape   = bias_weights.shape
        assert len(bias_shape) == 1, '3D transposed convolution bias weights tensor expected to have rank 1.'
        # REVIEW alexeyk: should really assert against D?
        assert bias_shape[0] == kernel_shape[3], '3D transposed convolution bias size does not match convolution output channels.'
        self._write_weights(name + '_b', bias_weights)
        return out_name

    def write_act(self, input, name):
        if self.act == 'elu':
            return self.write_elu(input, name)
        elif self.act == 'srelu':
            return self.write_srelu(input, name)
        else:
            assert False, 'Not supported activation: {}'.format(self.act)

    def write_elu(self, input, name):
        # Write code.
        code = """\
// {1} ELU activation op.
auto {1} = addElu(plugin_factory, *network, *{0}->getOutput(0), data_type, "{1}");
assert({1} != nullptr);
{1}->setName("{1}");

"""
        code = code.format(input, name)
        self.code_writer.write(self._indent_lines(code))
        return name

    def write_srelu(self, input, name):
        # Write code.
        code = """\
// {1} SReLU activation op.
auto {1}_shift_up = network->addScale(*{0}->getOutput(0), ScaleMode::kUNIFORM,
                                      weights.at("srelu_shift_up"), weights.at("srelu_shift_scale"), weights.at("srelu_shift_power"));

assert({1}_shift_up != nullptr);
{1}_shift_up->setName("{1}_shift_up");

auto {1}_relu = network->addActivation(*{1}_shift_up->getOutput(0), ActivationType::kRELU);
//auto {1}_relu = network->addActivation(*{0}->getOutput(0), ActivationType::kRELU);
assert({1}_relu != nullptr);
{1}_relu->setName("{1}_relu");

//auto {1} = {1}_relu;
auto {1} = network->addScale(*{1}_relu->getOutput(0), ScaleMode::kUNIFORM,
                             weights.at("srelu_shift_down"), weights.at("srelu_shift_scale"), weights.at("srelu_shift_power"));
assert({1} != nullptr);
{1}->setName("{1}");

"""
        code = code.format(input, name)
        self.code_writer.write(self._indent_lines(code))
        # Write SReLU biases only once.
        if not self.has_srelu_weights:
            self._write_weights('srelu_shift_up',    [1.0])
            self._write_weights('srelu_shift_down',  [-1.0])
            self._write_weights('srelu_shift_scale', [1.0])
            self._write_weights('srelu_shift_power', [1.0])
            self.has_srelu_weights = True

        return name

    def write_sigmoid(self, input, name):
        # Write code.
        code = """\
// {1} sigmoid activation op.
auto {1} = network->addActivation(*{0}->getOutput(0), ActivationType::kSIGMOID);
assert({1} != nullptr);
{1}->setName("{1}");

"""
        code = code.format(input, name)
        self.code_writer.write(self._indent_lines(code))
        return name

    def write_cost_vol(self, left_input, right_input, name, op_path, is_corr=False):
        # Write code.
        code = """\
// {2} cost volume op.
auto {2} = addCostVolume(plugin_factory, *network, *{0}->getOutput(0), *{1}->getOutput(0),
                         {3}, {4}, data_type, "{2}");
assert({2} != nullptr);
{2}->setName("{2}");

"""
        g = tf.get_default_graph()
        assert g is not None, "No default graph set"
        last_op   = g.get_operation_by_name(os.path.join(op_path, 'concat' if not is_corr else 'Sum'))
        out_shape = last_op.outputs[0].shape
        assert len(out_shape) == 5, 'Cost volume output tensor expected to have rank 5 but got {}.'.format(len(out_shape))
        # Max disparity is the second outermost dimension in the output.
        max_disp = out_shape[1]
        code = code.format(left_input, right_input, name, 
                           'CostVolumeType::kDefault' if not is_corr else 'CostVolumeType::kCorrelation',
                           max_disp)
        self.code_writer.write(self._indent_lines(code))
        return name

    def write_conv3d_pad(self, input, name):
        # Write code.
        code = """\
// {1} padding op.
auto {1} = addPad(plugin_factory, *network, *{0}->getOutput(0), {{0, 0, 0, 0}}, {{1, 0, 0, 0}}, "{1}");
assert({1} != nullptr);
{1}->setName("{1}");

"""
        code = code.format(input, name)
        self.code_writer.write(self._indent_lines(code))
        return name

    def write_conv3d_transform(self, input, name):
        code = """\
// Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
auto {1} = addTransform(plugin_factory, *network, *{0}->getOutput(0), {{1, 0, 2, 3}}, "{1}_transform");
assert({1} != nullptr);
{1}->setName("{1}");

"""
        code = code.format(input, name)
        self.code_writer.write(self._indent_lines(code))
        return name

    def write_add_tensors(self, t1, t2, name):
        # Write code.
        code = """\
// {2} tensor add op.
auto {2} = network->addElementWise(*({0}->getOutput(0)), *({1}->getOutput(0)), ElementWiseOperation::kSUM);
assert({2} != nullptr);
{2}->setName("{2}");

"""
        code = code.format(t1, t2, name)
        self.code_writer.write(self._indent_lines(code))
        return name

    def write_softargmax(self, input, name, is_argmin):
        code = """\
// Softargmax.
auto {1} = addSoftargmax(plugin_factory, *network, *{0}->getOutput(0), {2}, data_type, "{1}_softargmax");
assert({1} != nullptr);
{1}->setName("{1}");

"""
        code = code.format(input, name,
                           'SoftargmaxType::kMin' if is_argmin else 'SoftargmaxType::kMax' )
        self.code_writer.write(self._indent_lines(code))
        return name

    def write_concat_tensors(self, t1, t2, name):
        # Write code.
        code = """\
// {2} tensor concat op.
ITensor* {2}_inputs[] = {{{0}->getOutput(0), {1}->getOutput(0)}};
auto {2} = network->addConcatenation({2}_inputs, 2);
assert({2} != nullptr);
{2}->setName("{2}");

"""
        code = code.format(t1, t2, name)
        self.code_writer.write(self._indent_lines(code))
        return name
