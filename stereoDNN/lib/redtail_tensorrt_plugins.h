// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#ifndef REDTAIL_TENSORRT_PLUGINS_H
#define REDTAIL_TENSORRT_PLUGINS_H

#include <memory>
#include <NvInfer.h>

namespace redtail { namespace tensorrt
{

using namespace nvinfer1;

// -----------------------------------------------------------------
// 3D convolution type used to setup convolution descriptors.
// -----------------------------------------------------------------
enum class Conv3DType
{
    kCuDnn      = 0,
    kTensorFlow = 1
};

// -----------------------------------------------------------------
// Plugin container/factory.
// TensorRT does not manage plugins and requires a plugin lifetime
// to be the same as any TRT engine.
// Each plugin create* function returns naked pointer as expected by TRT,
// with IPluginContainer managing the plugin's lifetime.
// -----------------------------------------------------------------
class IPluginContainer
{
public:
    virtual ~IPluginContainer() = default;

    // ELU plugin.
    virtual IPlugin* createEluPlugin(DataType data_type, std::string name) = 0;

    // Cost volume plugin.
    virtual IPlugin* createCostVolumePlugin(int max_disparity, std::string name) = 0;

    // 3D convolution.
    virtual IPlugin* createConv3DPlugin(Conv3DType conv_type, Dims kernel_dims,
                                        Dims stride_dims, Dims pad_start_dims, Dims pad_end_dims,
                                        Weights kernel_weights, Weights bias_weights,
                                        std::string name) = 0;

    virtual IPlugin* createConv3DTransposePlugin(Conv3DType conv_type, Dims kernel_dims, Dims out_dims,
                                                 Dims stride_dims, Dims pad_start_dims, Dims pad_end_dims,
                                                 Weights kernel_weights, Weights bias_weights,
                                                 std::string name) = 0;

    virtual IPlugin* createTransformPlugin(Permutation permutation, std::string name) = 0;

    virtual IPlugin* createPaddingPlugin(DimsNCHW pad_start, DimsNCHW pad_end,
                                         std::string name) = 0;

    virtual IPlugin* createSlicePlugin(Dims dims, Dims slice_start, Dims slice_end,
                                       std::string name) = 0;

    virtual IPlugin* createSoftargmaxPlugin(std::string name) = 0;

    static std::unique_ptr<IPluginContainer> create(ILogger& log);
};

// -----------------------------------------------------------------
// Plugins helper functions.
// -----------------------------------------------------------------
ILayer* addElu(IPluginContainer& plugin_factory, INetworkDefinition& network, ITensor& input,
               DataType data_type, const std::string& name);

ILayer* addCostVolume(IPluginContainer& plugin_factory, INetworkDefinition& network,
                      ITensor& left_input, ITensor& right_input, int max_disparity,
                      const std::string& name);

ILayer* addConv3D(IPluginContainer& plugin_factory, INetworkDefinition& network, ITensor& input,
                  Conv3DType conv_type, Dims kernel_dims, Dims stride_dims,
                  Dims pad_start_dims, Dims pad_end_dims,
                  Weights kernel_weights, Weights bias_weights,
                  const std::string& name);

ILayer* addConv3DTranspose(IPluginContainer& plugin_factory, INetworkDefinition& network, ITensor& input,
                           Conv3DType conv_type, Dims kernel_dims, Dims out_dims,
                           Dims stride_dims, Dims pad_start_dims, Dims pad_end_dims,
                           Weights kernel_weights, Weights bias_weights,
                           const std::string& name);

ILayer* addSlice(IPluginContainer& plugin_factory, INetworkDefinition& network, ITensor& input,
                 Dims dims, Dims slice_start, Dims slice_end,
                 const std::string& name);

ILayer* addTransform(IPluginContainer& plugin_factory, INetworkDefinition& network, ITensor& input,
                     Permutation permutation,
                     const std::string& name);

ILayer* addPad(IPluginContainer& plugin_factory, INetworkDefinition& network, ITensor& input,
               DimsNCHW pad_start, DimsNCHW pad_end,
               const std::string& name);

ILayer* addSoftargmax(IPluginContainer& plugin_factory, INetworkDefinition& network, ITensor& input,
                      const std::string& name);

} }

#endif // REDTAIL_TENSORRT_PLUGINS_H
