# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

import sys

def writeDataLayer(f, netName):
    lines = [
"""#
# Auto-generated file. Any changes made to the file will be lost if the script is re-run.
#
""",
'name: "' + netName + '"',
"""
layer {
  name: "train-data"
  type: "Data"
  top: "data"
  top: "label"
  data_param {
    batch_size: 128
  }
  include {
    stage: "train"
  }
}
layer {
  name: "val-data"
  type: "Data"
  top: "data"
  top: "label"
  data_param {
    batch_size: 32
  }
  include {
    stage: "val"
  }
}
layer {
  name: "data_aug"
  type: "Python"
  bottom: "data"
  bottom: "label"
  top: "data"
  top: "label"
  python_param {
    module: "digits_python_layers"
    layer: "TrailAugLayer"
    param_str: "{'debug': False, 'hflip3': True, 'blurProb': 0.1, 'contrastRadius': 0.2, 'brightnessRadius': 0.2, 'saturationRadius': 0.3, 'sharpnessRadius': 0.3, 'scaleMin': 0.9, 'scaleMax': 1.2, 'rotateAngle': 15, 'numThreads': 32}"
  }
  include { stage: "train" }
}
layer {
  name: "sub_mean"
  type: "Scale"
  bottom: "data"
  top: "sub_mean"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  scale_param {
    filler {
      value: 0.00390625
    }
    bias_term: true
    bias_filler {
      value: -0.5
    }
  }
}
"""
    ]
    f.writelines(lines)

def writeLastPoolLayer(f, srcName):
    """
    Writes last average pooling layer.
    As TensorRT currently does not support global pooling, use hardcoded kernel size
    assuming input it 320x180
    """
    f.write(
"""
layer {
  bottom: "%s"
  top: "pool_avg"
  name: "pool_avg"
  type: "Pooling"
  pooling_param {
    kernel_w: 10
    kernel_h: 6
    stride: 1
    pool: AVE
  }
}
""" % srcName
    )

def writeOutputLayers(f, lastLayer):
    text = """
layer {
  name: "fc3"
  type: "InnerProduct"
  bottom: "%(name)s"
  top: "fc3"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 1.0
    decay_mult: 0.0
  }
  inner_product_param {
    num_output: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "fc3"
  bottom: "label"
  top: "accuracy"
  include {
    stage: "val"
  }
}
layer {
  name: "cee_loss"
  type: "Python"
  bottom: "fc3"
  bottom: "label"
  top: "cee_loss"
  python_param {
    module: "digits_python_layers"
    layer: "CrossEntropySoftmaxWithEntropyLossLayer"
    param_str: "{ 'entScale': 0.01, 'pScale': 0.0001, 'label_eps': 0.01 }"
  }
  loss_weight: 1
  exclude {
    stage: "deploy"
  }
}
layer {
  name: "softmax"
  type: "Softmax"
  bottom: "fc3"
  top: "softmax"
  include {
    stage: "deploy"
  }
}
"""
    f.write(text % {'name': lastLayer})

def firstConvLayer(f):
    f.write("""
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "sub_mean"
  top: "conv1"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 1.0
    decay_mult: 0.0
  }
  convolution_param {
    num_output: 64
    kernel_size: 7
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
layer {
  name: "conv1_srelu1_1"
  type: "Scale"
  bottom: "conv1"
  top: "conv1"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  scale_param {
    filler {
      value: 1
    }
    bias_term: true
    bias_filler {
      value: 1.0
    }
  }
}
layer {
  name: "conv1_srelu1_2"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
layer {
  name: "conv1_srelu1_3"
  type: "Scale"
  bottom: "conv1"
  top: "conv1"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  scale_param {
    filler {
      value: 1
    }
    bias_term: true
    bias_filler {
      value: -1.0
    }
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
""")

def writeConvLayer(f, srcName, numMap, name, kernelSize=3, stride=1, pad=1):
    f.write(
"""
layer {
  name: "%(name)s"
  type: "Convolution"
  bottom: "%(srcName)s"
  top: "%(name)s"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 1.0
    decay_mult: 0.0
  }
  convolution_param {
    num_output: %(numMap)d
    kernel_size: %(kernelSize)d
    stride: %(stride)d
    pad: %(pad)d
    weight_filler {
      type: "xavier"
    }
    bias_term: true
  }
}
""" % {'srcName': srcName, 'numMap': numMap, 'name': name, 'kernelSize': kernelSize, 'stride': stride, 'pad': pad})

def writeSReLULayer(f, name):
    f.write(
"""
layer {
  name: "%(name)s_srelu_1"
  type: "Scale"
  bottom: "%(name)s"
  top: "%(name)s"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  scale_param {
    filler {
      value: 1
    }
    bias_term: true
    bias_filler {
      value: 1.0
    }
  }
}
layer {
  name: "%(name)s_srelu_2"
  type: "ReLU"
  bottom: "%(name)s"
  top: "%(name)s"
}
layer {
  name: "%(name)s_srelu_3"
  type: "Scale"
  bottom: "%(name)s"
  top: "%(name)s"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  scale_param {
    filler {
      value: 1
    }
    bias_term: true
    bias_filler {
      value: -1.0
    }
  }
}
""" % {'name': name})

def writeSumLayer(f, name1, name2, name):
    f.write(
"""
layer {
  name: "%(name)s_sum"
  type: "Eltwise"
  bottom: "%(name1)s"
  bottom: "%(name2)s"
  top: "%(name)s"
  eltwise_param { operation: SUM }
}
""" % {'name1': name1, 'name2': name2, 'name': name})

def writeBlock2(f, srcName, numMap, name):
    # Write first 3x3 conv SReLU layer.
    curName = name + '_1'
    writeConvLayer(f, srcName, numMap, curName)
    writeSReLULayer(f, curName)
    # Write second 3x3 conv (no SReLU!) layer.
    prevName = curName
    curName = name + '_2'
    writeConvLayer(f, prevName, numMap, curName)
    # Write sum.
    writeSumLayer(f, srcName, curName, name)
    # Write final SReLU.
    writeSReLULayer(f, name)

def writeBlock2Inc(f, srcName, numMap, name):
    # Write first 3x3 conv SReLU layer.
    curName = name + '_1'
    writeConvLayer(f, srcName, numMap, curName)
    writeSReLULayer(f, curName)
    # Write second 3x3 conv (no SReLU!) layer with stride=2.
    prevName = curName
    curName = name + '_2'
    writeConvLayer(f, prevName, numMap, curName, stride=2)
    # Write projection layer.
    projName = name + '_proj'
    writeConvLayer(f, srcName, numMap, projName, kernelSize=1, stride=2, pad=0)
    # Write sum.
    writeSumLayer(f, projName, curName, name)
    # Write final SReLU.
    writeSReLULayer(f, name)

def writeResNetSections(f, sections, srcName, srcNumMap):
    for isec in range(0, len(sections)):
        numMap    = sections[isec][0]
        numBlocks = sections[isec][1]
        for iblock in range(0, numBlocks):
            curName = 'res{0}_{1}'.format(isec + 1, iblock + 1)
            if numMap == srcNumMap:
                writeBlock2(   f, srcName, numMap, curName)
            else:
                writeBlock2Inc(f, srcName, numMap, curName)
            srcNumMap = numMap
            srcName   = curName
    return srcName

if __name__ == '__main__':
    netName = 'SReLU ResNet-18'
    prototxtFile = './srelu-resnet-18.prototxt'
    if len(sys.argv) > 1:
        netName = sys.argv[1]
    if len(sys.argv) > 2:
        prototxtFile = sys.argv[2]
    resNet18Sections = [[64, 2], [128, 2], [256, 2], [512, 2]]
    with open(prototxtFile, 'w') as f:
        writeDataLayer(f, netName)
        firstConvLayer(f)
        lastLayer = writeResNetSections(f, resNet18Sections, 'pool1', 64)
        writeLastPoolLayer(f, lastLayer)
        # Write output, loss, softmax etc layers.
        writeOutputLayers(f, 'pool_avg')