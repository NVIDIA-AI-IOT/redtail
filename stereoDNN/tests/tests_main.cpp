// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "redtail_tensorrt_plugins.h"
#include "internal_utils.h"

using namespace nvinfer1;
using namespace redtail::tensorrt;

using FloatVec = std::vector<float>;

namespace testing
{
 namespace internal
 {
  enum GTestColor {
      COLOR_DEFAULT,
      COLOR_RED,
      COLOR_GREEN,
      COLOR_YELLOW
  };

  extern void ColoredPrintf(GTestColor color, const char* fmt, ...);
 }
}

// -----------------------------------------------------------------
// TensorRT logger.
// -----------------------------------------------------------------
class Logger : public nvinfer1::ILogger
{
public:
    Logger(ILogger::Severity max_log_level) :
        max_log_level_(max_log_level)
    {
    }

    void log(nvinfer1::ILogger::Severity severity, const char* msg) override
    {
        if (severity > max_log_level_)
            return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "TRT INTERNAL_ERROR: "; break;
            case Severity::kERROR:          std::cerr << "TRT ERROR: "; break;
            case Severity::kWARNING:        std::cerr << "TRT WARNING: "; break;
            case Severity::kINFO:           std::cerr << "TRT INFO: "; break;
            default:                        std::cerr << "TRT UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }

private:
    Severity max_log_level_;
};

static std::unique_ptr<Logger> g_logger;

std::string g_data_dir("");

// -----------------------------------------------------------------
// Helper struct to store network input information.
// -----------------------------------------------------------------
struct NetInput
{
    std::string name;
    Dims        dims;
    FloatVec    data;
};

IPluginLayer* addPlugin(INetworkDefinition& network, ITensor* const* inputs, int num_inputs, IPlugin* plugin)
{
    auto plugin_ext   = dynamic_cast<IPluginExt*>(plugin);
    auto plugin_layer = plugin_ext != nullptr 
                        ? network.addPluginExt(inputs, num_inputs, *plugin_ext)
                        : network.addPlugin(inputs, num_inputs, *plugin);
    return plugin_layer;
}

// -----------------------------------------------------------------
// Main test driver function. Creates all required TensorRT components
// to run TRT plugin. The plugin is created by the provided factory.
// Returns the output of the plugin.
// -----------------------------------------------------------------
template<typename FactoryOp, typename PostProcOp>
FloatVec runPlugin(int batch_size, const std::vector<NetInput>& inputs, Dims out_dims,
                   FactoryOp factory_op, PostProcOp post_proc_op,
                   IPluginContainer& factory, DataType data_type = DataType::kFLOAT)
{
    // REVIEW alexeyk: assuming single output for now.
    const char* output_name = "output";

    // Assuming that input tensors are in FP32 format, TRT will do the necessary conversion.
    // Though the code below supports FP16 input tensors, our sample code uses FP32 only.
    DataType input_data_type = DataType::kFLOAT;

    IBuilder* builder = createInferBuilder(*g_logger);
    // Note: must use EXPECT_* as ASSERT_ contains return statement.
    EXPECT_NE(builder, nullptr);

    INetworkDefinition* network = builder->createNetwork();
    EXPECT_NE(network, nullptr);

    // Add inputs.
    std::vector<ITensor*> plugin_inputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        // Must have at least rank 2 and same batch size.
        EXPECT_GT(inputs[i].dims.nbDims, 1);
        EXPECT_EQ(batch_size, inputs[i].dims.d[0]);
        // Get input dims without batch index dim.
        Dims in_dims;
        in_dims.nbDims = inputs[i].dims.nbDims - 1;
        std::copy(inputs[i].dims.d + 1, inputs[i].dims.d + inputs[i].dims.nbDims, in_dims.d);
        // addInput currently supports only 1D-3D input.
        if (in_dims.nbDims <= 3)
        {
            auto input = network->addInput(inputs[i].name.c_str(), input_data_type,
                                           DimsCHW(in_dims.d[0], in_dims.d[1], in_dims.d[2]));
            EXPECT_NE(input, nullptr);
            plugin_inputs[i] = input;
        }
        else if (in_dims.nbDims == 4)
        {
            // Create input with flattened dims.
            EXPECT_EQ(DimsUtils::getTensorSize(in_dims), (int)DimsUtils::getTensorSize(in_dims));
            DimsCHW flat_dims{1, 1, (int)DimsUtils::getTensorSize(in_dims)};
            auto input = network->addInput(inputs[i].name.c_str(), input_data_type, flat_dims);
            EXPECT_NE(input, nullptr);
            // Add reshape layer to restore original dims.
            auto shuf = network->addShuffle(*input);
            EXPECT_NE(shuf, nullptr);
            shuf->setReshapeDimensions(DimsNCHW(in_dims.d[0], in_dims.d[1], in_dims.d[2], in_dims.d[3]));
            plugin_inputs[i] = shuf->getOutput(0);
        }
        else
            assert(false); // TRT does not support input tensors with rank > 4.
    }

    // Create plugin. The factory method can create additional layers/plugins.
    IPlugin* plugin = factory_op();
    EXPECT_NE(plugin, nullptr);
    // Add the plugin to the network.
    auto plugin_layer = addPlugin(*network, &plugin_inputs[0], inputs.size(), plugin);
    EXPECT_NE(plugin_layer, nullptr);

    ILayer* out_layer = post_proc_op(network, plugin_layer, factory);

    // Setup network output.
    auto out_layer_out = out_layer->getOutput(0);
    EXPECT_NE(out_layer_out, nullptr);
    out_layer_out->setName(output_name);
    network->markOutput(*out_layer_out);

    // Build the engine.
    // builder->setMinFindIterations(2);
    // builder->setAverageFindIterations(2);

    builder->setMaxBatchSize(batch_size);
    // "ought to be enough for anybody."
    builder->setMaxWorkspaceSize(1024 * 1024 * 1024);

    builder->setHalf2Mode(data_type == DataType::kHALF);

    auto engine = builder->buildCudaEngine(*network);
    EXPECT_NE(engine, nullptr);
    // Network and builder can be destroyed right after network is built.
    // This follows the behavior in real (non-test) code.
    builder->destroy();
    network->destroy();

    // Setup input and output buffers.
    std::vector<void*> buffers(inputs.size() + 1);
    EXPECT_EQ(engine->getNbBindings(), buffers.size());

    size_t elt_size_bytes = input_data_type == DataType::kHALF ? 2 : 4;

    for (size_t i = 0; i < inputs.size(); i++)
    {
        // Expecting binding indices to match inputs.
        EXPECT_EQ(engine->getBindingIndex(inputs[i].name.c_str()), i);
        // Allocate and copy.
        CHECKL(cudaMalloc(&buffers[i], inputs[i].data.size() * elt_size_bytes), *g_logger);
        // Do FP32 -> FP16 of input if necessary.
        if (input_data_type == DataType::kFLOAT)
            CHECKL(cudaMemcpy(buffers[i], inputs[i].data.data(), inputs[i].data.size() * elt_size_bytes, cudaMemcpyHostToDevice), *g_logger);
        else
        {
            cv::Mat dst;
            cv::convertFp16(cv::Mat(inputs[i].data), dst);
            CHECKL(cudaMemcpy(buffers[i], dst.data, inputs[i].data.size() * elt_size_bytes, cudaMemcpyHostToDevice), *g_logger);
        }
    }

    int    out_idx  = engine->getBindingIndex(output_name);
    size_t out_size = DimsUtils::getTensorSize(out_dims);
    EXPECT_EQ(out_idx, buffers.size() - 1);
    CHECKL(cudaMalloc(&buffers[out_idx], out_size * elt_size_bytes), *g_logger);

    // Create the context.
    IExecutionContext *context = engine->createExecutionContext();
    EXPECT_NE(context, nullptr);

    // Run (finally).
    auto host_start = std::chrono::high_resolution_clock::now();
    auto res = context->execute(batch_size, buffers.data());
    auto host_end   = std::chrono::high_resolution_clock::now();
    EXPECT_TRUE(res);
    auto host_elapsed_ms = std::chrono::duration<float, std::milli>(host_end - host_start).count();
    auto msg = std::string("Host execution time: ") + std::to_string(host_elapsed_ms);
    g_logger->log(ILogger::Severity::kINFO, msg.c_str());

    // Copy results back to host.
    FloatVec out_h(out_size);
    // Do FP32 -> FP16 of input if necessary.
    if (input_data_type == DataType::kFLOAT)
    {
        auto out_h_p = const_cast<float*>(out_h.data());
        CHECKL(cudaMemcpy(out_h_p, buffers[out_idx], out_h.size() * sizeof(float), cudaMemcpyDeviceToHost), *g_logger);
    }
    else if (input_data_type == DataType::kHALF)
    {
        std::vector<uint16_t> out_h_16(out_size);
        auto out_h_p = const_cast<uint16_t*>(out_h_16.data());
        CHECKL(cudaMemcpy(out_h_p, buffers[out_idx], out_h.size() * elt_size_bytes, cudaMemcpyDeviceToHost), *g_logger);
        cv::Mat dst;
        cv::convertFp16(cv::Mat(out_size, 1, CV_16S, out_h_p), dst);
        std::copy((float*)dst.data, (float*)dst.data + out_size, out_h.begin());
    }
    else
        ADD_FAILURE();
    // Clean up.
    for (size_t i = 0; i < buffers.size(); i++)
        CHECKL(cudaFree(buffers[i]), *g_logger);
    context->destroy();
    engine->destroy();

    return out_h;
}

// -----------------------------------------------------------------
// Reads the binary file produced by Python test_data_generator.py script.
// The format of the file:
// - number of dimensions (int32)
// - dimensions (int32[])
// - data (float32[])
// -----------------------------------------------------------------
FloatVec readBinaryFile(const std::string& filename, Dims& dims)
{
    std::ifstream file(filename, std::ios::binary);
    EXPECT_TRUE(file.is_open());

    int32_t dim_count;
    file.read(reinterpret_cast<char*>(&dim_count), sizeof(int32_t));
    EXPECT_GE(dim_count, 1);
    EXPECT_LE(dim_count, sizeof(dims.d) / sizeof(dims.d[0]));

    dims.nbDims = dim_count;
    file.read(reinterpret_cast<char*>(dims.d), dim_count * sizeof(int32_t));

    FloatVec res(DimsUtils::getTensorSize(dims));
    file.read(reinterpret_cast<char*>(&res[0]), res.size() * sizeof(float));
    return res;
}

// -----------------------------------------------------------------
// ELU plugin tests.
// -----------------------------------------------------------------
TEST(EluPluginTests, Basic)
{
    Dims in_dims;
    Dims out_dims;
    FloatVec in  = readBinaryFile(g_data_dir + "elu_i_01.bin", in_dims);
    FloatVec out = readBinaryFile(g_data_dir + "elu_o_01.bin", out_dims);
    ASSERT_EQ(in_dims.nbDims, out_dims.nbDims);
    ASSERT_EQ(in_dims.nbDims, 4);
    ASSERT_EQ(in_dims.d[0],   1);

    auto factory = IPluginContainer::create(*g_logger);
    auto actual  = runPlugin(1, {{"input", in_dims, in}}, out_dims,
                             [&] { return factory->createEluPlugin(DataType::kFLOAT, "ELU"); },
                             [] (INetworkDefinition*, ILayer* plugin, IPluginContainer&) { return plugin; },
                             *factory);

    ASSERT_EQ(out.size(), actual.size());
    for (size_t i = 0; i < actual.size(); i++)
        EXPECT_FLOAT_EQ(out[i], actual[i]) << "Vectors 'actual' and 'out' differ at index " << i;
}

TEST(EluPluginTests, BasicFP16)
{
    Dims in_dims;
    Dims out_dims;
    FloatVec in  = readBinaryFile(g_data_dir + "elu_i_01.bin", in_dims);
    FloatVec out = readBinaryFile(g_data_dir + "elu_o_01.bin", out_dims);
    ASSERT_EQ(in_dims.nbDims, out_dims.nbDims);
    ASSERT_EQ(in_dims.nbDims, 4);
    ASSERT_EQ(in_dims.d[0],   1);

    auto data_type = DataType::kHALF;
    auto factory = IPluginContainer::create(*g_logger);
    auto actual  = runPlugin(1, {{"input", in_dims, in}}, out_dims,
                             [&] { return factory->createEluPlugin(data_type, "ELU"); },
                             [] (INetworkDefinition*, ILayer* plugin, IPluginContainer&) { return plugin; },
                             *factory, data_type);

    ASSERT_EQ(out.size(), actual.size());
    for (size_t i = 0; i < actual.size(); i++)
        EXPECT_NEAR(out[i], actual[i], 0.01) << "Vectors 'actual' and 'out' differ at index " << i;
}

TEST(EluPluginTests, Input4DBatchSize2)
{
    Dims in_dims;
    Dims out_dims;
    FloatVec in  = readBinaryFile(g_data_dir + "elu_i_02.bin", in_dims);
    FloatVec out = readBinaryFile(g_data_dir + "elu_o_02.bin", out_dims);
    ASSERT_EQ(in_dims.nbDims, out_dims.nbDims);
    ASSERT_EQ(in_dims.nbDims, 5);
    ASSERT_EQ(in_dims.d[0],   2);

    auto factory = IPluginContainer::create(*g_logger);
    auto actual  = runPlugin(2, {{"input", in_dims, in}}, out_dims,
                             [&] { return factory->createEluPlugin(DataType::kFLOAT, "ELU"); },
                             [] (INetworkDefinition*, ILayer* plugin, IPluginContainer&) { return plugin; },
                             *factory);

    ASSERT_EQ(out.size(), actual.size());
    for (size_t i = 0; i < actual.size(); i++)
        EXPECT_FLOAT_EQ(out[i], actual[i]) << "Vectors 'actual' and 'out' differ at index " << i;
}

// -----------------------------------------------------------------
// Conv3DPlugin plugin tests.
// REVIEW alexeyk: consider converting all these tests to data-driven test.
// -----------------------------------------------------------------

// Post processing for Conv3D layer.
ILayer* addConv3DPostProcessing(INetworkDefinition* network, ILayer* plugin, IPluginContainer& factory)
{
    auto transform = factory.createTransformPlugin({1, 0, 2, 3}, "Conv3DTransform");
    EXPECT_NE(transform, nullptr);
    auto transform_in = plugin->getOutput(0);
    auto transform_layer = addPlugin(*network, &transform_in, 1, transform);
    EXPECT_NE(transform_layer, nullptr);
    return transform_layer;
}

TEST(Conv3DPluginTests, Basic)
{
    Dims x_dims;
    Dims w_dims;
    Dims y_dims;
    FloatVec x = readBinaryFile(g_data_dir + "conv3d_01_x.bin", x_dims);
    FloatVec w = readBinaryFile(g_data_dir + "conv3d_01_w.bin", w_dims);
    FloatVec y = readBinaryFile(g_data_dir + "conv3d_01_y.bin", y_dims);
    ASSERT_EQ(x_dims.nbDims, 5);
    ASSERT_EQ(w_dims.nbDims, 5);
    ASSERT_EQ(y_dims.nbDims, 5);

    auto factory = IPluginContainer::create(*g_logger);
    auto actual  = runPlugin(1, {{"x", x_dims, x}}, y_dims,
                             [&]
                             {
                                 return factory->createConv3DPlugin(Conv3DType::kTensorFlow,
                                                                    w_dims, {3, {1, 1, 1}},
                                                                    {3, {0, 0, 0}}, {3, {0, 0, 0}},
                                                                    Weights{DataType::kFLOAT, w.data(), (int64_t)w.size() },
                                                                    Weights{DataType::kFLOAT, nullptr, 0 },
                                                                    "Conv3D");
                             },
                             addConv3DPostProcessing,
                             *factory);

    ASSERT_EQ(y.size(), actual.size());
    for (size_t i = 0; i < actual.size(); i++)
         EXPECT_FLOAT_EQ(y[i], actual[i]) << "Vectors 'actual' and 'y' differ at index " << i;
}

TEST(Conv3DPluginTests, HWStridesAndPadWithMultiK)
{
    Dims x_dims;
    Dims w_dims;
    Dims y_dims;
    FloatVec x = readBinaryFile(g_data_dir + "conv3d_02_x.bin", x_dims);
    FloatVec w = readBinaryFile(g_data_dir + "conv3d_02_w.bin", w_dims);
    FloatVec y = readBinaryFile(g_data_dir + "conv3d_02_y.bin", y_dims);
    ASSERT_EQ(x_dims.nbDims, 5);
    ASSERT_EQ(w_dims.nbDims, 5);
    ASSERT_EQ(y_dims.nbDims, 5);

    auto factory = IPluginContainer::create(*g_logger);
    auto actual  = runPlugin(1, {{"x", x_dims, x}}, y_dims,
                             [&]
                             {
                                 return factory->createConv3DPlugin(Conv3DType::kTensorFlow,
                                                                    w_dims, {3, {1, 2, 2}},
                                                                    {3, {0, 1, 1}}, {3, {0, 1, 1}},
                                                                    Weights{DataType::kFLOAT, w.data(), (int64_t)w.size() },
                                                                    Weights{DataType::kFLOAT, nullptr, 0 },
                                                                    "Conv3D");
                             },
                             addConv3DPostProcessing,
                             *factory);

    ASSERT_EQ(y.size(), actual.size());
    for (size_t i = 0; i < actual.size(); i++)
         EXPECT_NEAR(y[i], actual[i], 0.00001) << "Vectors 'actual' and 'y' differ at index " << i;
}

TEST(Conv3DPluginTests, DHWStridesAndPadWithMultiK)
{
    Dims x_dims;
    Dims w_dims;
    Dims y_dims;
    FloatVec x = readBinaryFile(g_data_dir + "conv3d_03_x.bin", x_dims);
    FloatVec w = readBinaryFile(g_data_dir + "conv3d_03_w.bin", w_dims);
    FloatVec y = readBinaryFile(g_data_dir + "conv3d_03_y.bin", y_dims);
    ASSERT_EQ(x_dims.nbDims, 5);
    ASSERT_EQ(w_dims.nbDims, 5);
    ASSERT_EQ(y_dims.nbDims, 5);

    auto factory = IPluginContainer::create(*g_logger);
    // In this test we have to manually pad input in D dimension.
    x_dims.d[1] += 1;
    FloatVec zero_pad(x_dims.d[2] * x_dims.d[3] * x_dims.d[4]);
    x.insert(x.end(), zero_pad.cbegin(), zero_pad.cend());
    auto actual = runPlugin(1, {{"x", x_dims, x}}, y_dims,
                            [&]
                            {
                                 return factory->createConv3DPlugin(Conv3DType::kTensorFlow,
                                                                    w_dims, {3, {1, 2, 2}},
                                                                    {3, {0, 1, 1}}, {3, {0, 1, 1}},
                                                                    Weights{DataType::kFLOAT, w.data(), (int64_t)w.size() },
                                                                    Weights{DataType::kFLOAT, nullptr, 0 },
                                                                    "Conv3D");
                            },
                            addConv3DPostProcessing,
                            *factory);

    ASSERT_EQ(y.size(), actual.size());
    for (size_t i = 0; i < actual.size(); i++)
         EXPECT_NEAR(y[i], actual[i], 0.00001) << "Vectors 'actual' and 'y' differ at index " << i;
}

TEST(Conv3DPluginTests, UnitStridesAndPadSymDWithMultiK)
{
    Dims x_dims;
    Dims w_dims;
    Dims y_dims;
    FloatVec x = readBinaryFile(g_data_dir + "conv3d_04_x.bin", x_dims);
    FloatVec w = readBinaryFile(g_data_dir + "conv3d_04_w.bin", w_dims);
    FloatVec y = readBinaryFile(g_data_dir + "conv3d_04_y.bin", y_dims);
    ASSERT_EQ(x_dims.nbDims, 5);
    ASSERT_EQ(w_dims.nbDims, 5);
    ASSERT_EQ(y_dims.nbDims, 5);

    auto factory = IPluginContainer::create(*g_logger);
    auto actual  = runPlugin(1, {{"x", x_dims, x}}, y_dims,
                             [&]
                             {
                                 return factory->createConv3DPlugin(Conv3DType::kTensorFlow,
                                                                    w_dims, {3, {1, 1, 1}},
                                                                    {3, {1, 1, 1}}, {3, {1, 1, 1}},
                                                                    Weights{DataType::kFLOAT, w.data(), (int64_t)w.size() },
                                                                    Weights{DataType::kFLOAT, nullptr, 0 },
                                                                    "Conv3D");
                             },
                             addConv3DPostProcessing,
                             *factory);

    ASSERT_EQ(y.size(), actual.size());
    for (size_t i = 0; i < actual.size(); i++)
         EXPECT_NEAR(y[i], actual[i], 0.0001) << "Vectors 'actual' and 'y' differ at index " << i;
}

TEST(Conv3DPluginTests, DHWStridesAndPadAsymDWithMultiK)
{
    Dims x_dims;
    Dims w_dims;
    Dims y_dims;
    FloatVec x = readBinaryFile(g_data_dir + "conv3d_05_x.bin", x_dims);
    FloatVec w = readBinaryFile(g_data_dir + "conv3d_05_w.bin", w_dims);
    FloatVec y = readBinaryFile(g_data_dir + "conv3d_05_y.bin", y_dims);
    ASSERT_EQ(x_dims.nbDims, 5);
    ASSERT_EQ(w_dims.nbDims, 5);
    ASSERT_EQ(y_dims.nbDims, 5);

    auto factory = IPluginContainer::create(*g_logger);
    // In this test we have to manually pad input in D dimension.
    x_dims.d[1] += 1;
    FloatVec zero_pad(x_dims.d[2] * x_dims.d[3] * x_dims.d[4]);
    x.insert(x.end(), zero_pad.cbegin(), zero_pad.cend());
    auto actual = runPlugin(1, {{"x", x_dims, x}}, y_dims,
                            [&]
                            {
                                return factory->createConv3DPlugin(Conv3DType::kTensorFlow,
                                                                   w_dims, {3, {2, 2, 2}},
                                                                   {3, {0, 1, 1}}, {3, {1, 1, 1}},
                                                                   Weights{DataType::kFLOAT, w.data(), (int64_t)w.size() },
                                                                   Weights{DataType::kFLOAT, nullptr, 0 },
                                                                   "Conv3D");
                            },
                            addConv3DPostProcessing,
                            *factory);

    ASSERT_EQ(y.size(), actual.size());
    for (size_t i = 0; i < actual.size(); i++)
         EXPECT_NEAR(y[i], actual[i], 0.0001) << "Vectors 'actual' and 'y' differ at index " << i;
}

TEST(Conv3DPluginTests, DHWStridesAndPadAsymDWithMultiKWithBiasAndElu)
{
    Dims x_dims;
    Dims w_dims;
    Dims b_dims;
    Dims y_dims;
    FloatVec x = readBinaryFile(g_data_dir + "conv3d_06_x.bin", x_dims);
    FloatVec w = readBinaryFile(g_data_dir + "conv3d_06_w.bin", w_dims);
    FloatVec b = readBinaryFile(g_data_dir + "conv3d_06_b.bin", b_dims);
    FloatVec y = readBinaryFile(g_data_dir + "conv3d_06_y.bin", y_dims);
    ASSERT_EQ(x_dims.nbDims, 5);
    ASSERT_EQ(w_dims.nbDims, 5);
    ASSERT_EQ(b_dims.nbDims, 1);
    ASSERT_EQ(y_dims.nbDims, 5);

    auto data_type = DataType::kFLOAT;
    auto factory = IPluginContainer::create(*g_logger);
    // In this test we have to manually pad input in D dimension.
    x_dims.d[1] += 1;
    FloatVec zero_pad(x_dims.d[2] * x_dims.d[3] * x_dims.d[4]);
    x.insert(x.end(), zero_pad.cbegin(), zero_pad.cend());
    auto actual = runPlugin(1, {{"x", x_dims, x}}, y_dims,
                            [&]
                            {
                                return factory->createConv3DPlugin(Conv3DType::kTensorFlow,
                                                                   w_dims, {3, {2, 2, 2}},
                                                                   {3, {0, 1, 1}}, {3, {1, 1, 1}},
                                                                   Weights{DataType::kFLOAT, w.data(), (int64_t)w.size() },
                                                                   Weights{DataType::kFLOAT, b.data(), (int64_t)b.size() },
                                                                   "Conv3D");
                            },
                            [&] (INetworkDefinition* network, ILayer* plugin, IPluginContainer& f)
                            {
                                auto transform = addConv3DPostProcessing(network, plugin, f);
                                // Add ELU.
                                auto elu_plugin    = f.createEluPlugin(data_type, "ELU");
                                EXPECT_NE(elu_plugin, nullptr);
                                auto elu_plugin_in = transform->getOutput(0);
                                auto plugin_layer  = addPlugin(*network, &elu_plugin_in, 1, elu_plugin);
                                EXPECT_NE(plugin_layer, nullptr);
                                return plugin_layer;
                            },
                            *factory);

    ASSERT_EQ(y.size(), actual.size());
    for (size_t i = 0; i < actual.size(); i++)
         EXPECT_NEAR(y[i], actual[i], 0.0001) << "Vectors 'actual' and 'y' differ at index " << i;
}

TEST(Conv3DPluginTests, Multiple)
{
    Dims x_dims;
    Dims w_dims;
    Dims y_dims;
    FloatVec x = readBinaryFile(g_data_dir + "conv3d_07_x.bin", x_dims);
    FloatVec w = readBinaryFile(g_data_dir + "conv3d_07_w.bin", w_dims);
    FloatVec y = readBinaryFile(g_data_dir + "conv3d_07_y.bin", y_dims);
    ASSERT_EQ(x_dims.nbDims, 5);
    ASSERT_EQ(w_dims.nbDims, 5);
    ASSERT_EQ(y_dims.nbDims, 5);

    auto factory = IPluginContainer::create(*g_logger);
    auto actual  = runPlugin(1, {{"x", x_dims, x}}, y_dims,
                             [&]
                             {
                                 return factory->createConv3DPlugin(Conv3DType::kTensorFlow,
                                                                    w_dims, {3, {1, 1, 1}},
                                                                    {3, {1, 1, 1}}, {3, {1, 1, 1}},
                                                                    Weights{DataType::kFLOAT, w.data(), (int64_t)w.size() },
                                                                    Weights{DataType::kFLOAT, nullptr, 0 },
                                                                    "Conv3D_1");
                             },
                             [&] (INetworkDefinition* network, ILayer* plugin, IPluginContainer& f)
                             {
                                 auto transform_1 = addConv3DPostProcessing(network, plugin, f);
                                 auto pad_plugin = f.createPaddingPlugin({0, 0, 0, 0}, {1, 0, 0, 0}, "Pad_1");
                                 EXPECT_NE(pad_plugin, nullptr);
                                 auto pad_plugin_in = transform_1->getOutput(0);
                                 auto pad_plugin_layer  = addPlugin(*network, &pad_plugin_in, 1, pad_plugin);
                                 EXPECT_NE(pad_plugin_layer, nullptr);
                                 // Add second Conv3D.
                                 auto conv_plugin_2  = f.createConv3DPlugin(Conv3DType::kTensorFlow,
                                                                            w_dims, DimsCHW{2, 2, 2},
                                                                            DimsCHW{0, 1, 1}, DimsCHW{0, 1, 1},
                                                                            Weights{DataType::kFLOAT, w.data(), (int64_t)w.size() },
                                                                            Weights{DataType::kFLOAT, nullptr, 0 },
                                                                            "Conv3D_2");
                                 EXPECT_NE(conv_plugin_2, nullptr);
                                 auto conv_plugin_2_in = pad_plugin_layer->getOutput(0);
                                 auto plugin_layer_2   = addPlugin(*network, &conv_plugin_2_in, 1, conv_plugin_2);
                                 EXPECT_NE(plugin_layer_2, nullptr);
                                 // Add transform for the second convo.
                                 auto transform_2 = addConv3DPostProcessing(network, plugin_layer_2, f);
                                 return transform_2;
                             },
                             *factory);

    ASSERT_EQ(y.size(), actual.size());
    for (size_t i = 0; i < actual.size(); i++)
         EXPECT_NEAR(y[i], actual[i], 0.0001) << "Vectors 'actual' and 'y' differ at index " << i;
}

// -----------------------------------------------------------------
// Conv3DTransposePlugin plugin tests.
// -----------------------------------------------------------------

// Post processing for Conv3DTranspose layer.
ILayer* addConv3DTranPostProc(INetworkDefinition* network, ILayer* plugin, IPluginContainer& factory)
{
    auto transform = factory.createTransformPlugin({1, 0, 2, 3}, "Conv3DTransposeTransform");
    EXPECT_NE(transform, nullptr);
    auto transform_in = plugin->getOutput(0);
    auto transform_layer = addPlugin(*network, &transform_in, 1, transform);
    EXPECT_NE(transform_layer, nullptr);
    return transform_layer;
}

ILayer* addConv3DTranSliceLayer(Dims dims, INetworkDefinition* network, ILayer* src_layer, IPluginContainer& factory)
{
    auto slice_plugin = factory.createSlicePlugin(dims,
                                                  {4, {0, 0, 0, 0}},
                                                  {4, {dims.d[0] - 1, dims.d[1], dims.d[2], dims.d[3]}},
                                                  "Slice");
    EXPECT_NE(slice_plugin, nullptr);
    auto slice_plugin_in = src_layer->getOutput(0);
    auto slice_plugin_layer  = addPlugin(*network, &slice_plugin_in, 1, slice_plugin);
    EXPECT_NE(slice_plugin_layer, nullptr);
    return slice_plugin_layer;
}

TEST(Conv3DTransposePluginTests, Basic)
{
    Dims y_dims;
    Dims w_dims;
    Dims x_dims;
    FloatVec y = readBinaryFile(g_data_dir + "conv3d_tran_01_y.bin", y_dims);
    FloatVec w = readBinaryFile(g_data_dir + "conv3d_tran_01_w.bin", w_dims);
    FloatVec x = readBinaryFile(g_data_dir + "conv3d_tran_01_x.bin", x_dims);
    ASSERT_EQ(y_dims.nbDims, 5);
    ASSERT_EQ(w_dims.nbDims, 5);
    ASSERT_EQ(x_dims.nbDims, 5);

    auto factory = IPluginContainer::create(*g_logger);
    auto actual = runPlugin(1, {{"y", y_dims, y}}, x_dims,
                            [&]
                            {
                                return factory->createConv3DTransposePlugin(
                                    Conv3DType::kTensorFlow,
                                    w_dims, {4, {x_dims.d[1], x_dims.d[2], x_dims.d[3], x_dims.d[4]}},
                                    {3, {1, 1, 1}}, {3, {0, 0, 0}}, {3, {0, 0, 0}},
                                    Weights{DataType::kFLOAT, w.data(), (int64_t)w.size() },
                                    Weights{DataType::kFLOAT, nullptr, 0 },
                                    "Conv3DTranspose");
                            },
                            addConv3DTranPostProc,
                            *factory);

    ASSERT_EQ(x.size(), actual.size());
    for (size_t i = 0; i < actual.size(); i++)
         EXPECT_FLOAT_EQ(x[i], actual[i]) << "Vectors 'x' and 'actual' differ at index " << i;
}

TEST(Conv3DTransposePluginTests, HWStridesAndPadWithMultiK)
{
    Dims y_dims;
    Dims w_dims;
    Dims x_dims;
    FloatVec y = readBinaryFile(g_data_dir + "conv3d_tran_02_y.bin", y_dims);
    FloatVec w = readBinaryFile(g_data_dir + "conv3d_tran_02_w.bin", w_dims);
    FloatVec x = readBinaryFile(g_data_dir + "conv3d_tran_02_x.bin", x_dims);
    ASSERT_EQ(y_dims.nbDims, 5);
    ASSERT_EQ(w_dims.nbDims, 5);
    ASSERT_EQ(x_dims.nbDims, 5);

    auto factory = IPluginContainer::create(*g_logger);
    auto actual = runPlugin(1, {{"y",y_dims, y}}, x_dims,
                            [&]
                            {
                                return factory->createConv3DTransposePlugin(
                                    Conv3DType::kTensorFlow,
                                    w_dims, {4, {x_dims.d[1], x_dims.d[2], x_dims.d[3], x_dims.d[4]}},
                                    {3, {1, 2, 2}}, {3, {0, 1, 1}}, {3, {0, 1, 1}},
                                    Weights{DataType::kFLOAT, w.data(), (int64_t)w.size() },
                                    Weights{DataType::kFLOAT, nullptr, 0 },
                                    "Conv3DTranspose");
                            },
                            addConv3DTranPostProc,
                            *factory);

    ASSERT_EQ(x.size(), actual.size());
    for (size_t i = 0; i < actual.size(); i++)
         EXPECT_NEAR(x[i], actual[i], 0.0001) << "Vectors 'x' and 'actual' differ at index " << i;
}

TEST(Conv3DTransposePluginTests, DHWStridesAndPadAsymDWithMultiK)
{
    Dims y_dims;
    Dims w_dims;
    Dims x_dims;
    FloatVec y = readBinaryFile(g_data_dir + "conv3d_tran_03_y.bin", y_dims);
    FloatVec w = readBinaryFile(g_data_dir + "conv3d_tran_03_w.bin", w_dims);
    FloatVec x = readBinaryFile(g_data_dir + "conv3d_tran_03_x.bin", x_dims);
    ASSERT_EQ(y_dims.nbDims, 5);
    ASSERT_EQ(w_dims.nbDims, 5);
    ASSERT_EQ(x_dims.nbDims, 5);

    auto factory = IPluginContainer::create(*g_logger);
    // In this test we have to manually pad output by 1 in D dimension.
    auto out_dims = DimsNCHW(x_dims.d[1], x_dims.d[2], x_dims.d[3], x_dims.d[4]);
    out_dims.d[0] += 1;
    auto actual = runPlugin(1, {{"y", y_dims, y}}, x_dims,
                            [&]
                            {
                                return factory->createConv3DTransposePlugin(
                                    Conv3DType::kTensorFlow,
                                    w_dims, out_dims,
                                    {3, {2, 2, 2}}, {3, {0, 1, 1}}, {3, {0, 1, 1}},
                                    Weights{DataType::kFLOAT, w.data(), (int64_t)w.size() },
                                    Weights{DataType::kFLOAT, nullptr, 0 },
                                    "Conv3DTranspose");
                            },
                            [&] (INetworkDefinition* network, ILayer* plugin, IPluginContainer& f)
                            {
                                 auto slice_plugin = f.createSlicePlugin(out_dims,
                                                                         {4, {0, 0, 0, 0}},
                                                                         {4, {x_dims.d[1], x_dims.d[2], x_dims.d[3], x_dims.d[4]}},
                                                                         "Slice");
                                 EXPECT_NE(slice_plugin, nullptr);
                                 auto slice_plugin_in = plugin->getOutput(0);
                                 auto slice_plugin_layer  = addPlugin(*network, &slice_plugin_in, 1, slice_plugin);
                                 EXPECT_NE(slice_plugin_layer, nullptr);
                                 return slice_plugin_layer;
                            },
                            *factory);

    ASSERT_EQ(x.size(), actual.size());
    for (size_t i = 0; i < actual.size(); i++)
         EXPECT_NEAR(x[i], actual[i], 0.0001) << "Vectors 'x' and 'actual' differ at index " << i;
}

TEST(Conv3DTransposePluginTests, DHWStridesAndPadAsymDWithMultiKWithBiasAndElu)
{
    Dims y_dims;
    Dims w_dims;
    Dims x_dims;
    Dims b_dims;
    FloatVec y = readBinaryFile(g_data_dir + "conv3d_tran_04_y.bin", y_dims);
    FloatVec w = readBinaryFile(g_data_dir + "conv3d_tran_04_w.bin", w_dims);
    FloatVec b = readBinaryFile(g_data_dir + "conv3d_tran_04_b.bin", b_dims);
    FloatVec x = readBinaryFile(g_data_dir + "conv3d_tran_04_x.bin", x_dims);
    ASSERT_EQ(y_dims.nbDims, 5);
    ASSERT_EQ(w_dims.nbDims, 5);
    ASSERT_EQ(b_dims.nbDims, 1);
    ASSERT_EQ(x_dims.nbDims, 5);

    auto data_type = DataType::kFLOAT;
    auto factory = IPluginContainer::create(*g_logger);
    // In this test we have to manually pad output by 1 in D dimension.
    auto out_dims = DimsNCHW(x_dims.d[1], x_dims.d[2], x_dims.d[3], x_dims.d[4]);
    out_dims.d[0] += 1;
    auto actual = runPlugin(1, {{"y", y_dims, y}}, x_dims,
                            [&]
                            {
                                return factory->createConv3DTransposePlugin(
                                    Conv3DType::kTensorFlow,
                                    w_dims, out_dims,
                                    {3, {2, 2, 2}}, {3, {0, 1, 1}}, {3, {0, 1, 1}},
                                    Weights{DataType::kFLOAT, w.data(), (int64_t)w.size() },
                                    Weights{DataType::kFLOAT, b.data(), (int64_t)b.size() },
                                    "Conv3DTranspose");
                            },
                            [&] (INetworkDefinition* network, ILayer* plugin, IPluginContainer& f)
                            {
                                 auto slice_plugin = f.createSlicePlugin(out_dims,
                                                                         {4, {0, 0, 0, 0}},
                                                                         {4, {x_dims.d[1], x_dims.d[2], x_dims.d[3], x_dims.d[4]}},
                                                                         "Slice");
                                 EXPECT_NE(slice_plugin, nullptr);
                                 auto slice_plugin_in = plugin->getOutput(0);
                                 auto slice_plugin_layer  = addPlugin(*network, &slice_plugin_in, 1, slice_plugin);
                                 EXPECT_NE(slice_plugin_layer, nullptr);
                                 // Add ELU.
                                 auto elu_plugin    = f.createEluPlugin(data_type, "ELU");
                                 EXPECT_NE(elu_plugin, nullptr);
                                 auto elu_plugin_in = slice_plugin_layer->getOutput(0);
                                 auto elu_layer  = addPlugin(*network, &elu_plugin_in, 1, elu_plugin);
                                 EXPECT_NE(elu_layer, nullptr);
                                 return elu_layer;
                            },
                            *factory);

    ASSERT_EQ(x.size(), actual.size());
    for (size_t i = 0; i < actual.size(); i++)
         EXPECT_NEAR(x[i], actual[i], 0.0001) << "Vectors 'x' and 'actual' differ at index " << i;
}

TEST(Conv3DTransposePluginTests, Multiple)
{
    Dims y_dims;
    Dims w1_dims;
    Dims w2_dims;
    Dims x_dims;
    FloatVec y  = readBinaryFile(g_data_dir + "conv3d_tran_05_y.bin",  y_dims);
    FloatVec w1 = readBinaryFile(g_data_dir + "conv3d_tran_05_w1.bin", w1_dims);
    FloatVec w2 = readBinaryFile(g_data_dir + "conv3d_tran_05_w2.bin", w2_dims);
    FloatVec x  = readBinaryFile(g_data_dir + "conv3d_tran_05_x.bin",  x_dims);
    ASSERT_EQ(y_dims.nbDims,  5);
    ASSERT_EQ(w1_dims.nbDims, 5);
    ASSERT_EQ(w2_dims.nbDims, 5);
    ASSERT_EQ(x_dims.nbDims,  5);

    auto factory = IPluginContainer::create(*g_logger);
    // In this test we have to manually pad output by 1 in D dimension.
    auto out_dims1 = DimsNCHW(8 + 1, 8, 9, 9); // REVIEW alexeyk: hardcoded for now.
    auto out_dims2 = DimsNCHW(x_dims.d[1], x_dims.d[2], x_dims.d[3], x_dims.d[4]);
    out_dims2.d[0] += 1;
    auto actual = runPlugin(1, {{"y", y_dims, y}}, x_dims,
                            [&]
                            {
                                return factory->createConv3DTransposePlugin(
                                    Conv3DType::kTensorFlow,
                                    w1_dims, out_dims1,
                                    {3, {2, 2, 2}}, {3, {0, 1, 1}}, {3, {0, 1, 1}},
                                    Weights{DataType::kFLOAT, w1.data(), (int64_t)w1.size() },
                                    Weights{DataType::kFLOAT, nullptr, 0 },
                                    "Conv3DTranspose_1");
                            },
                            [&] (INetworkDefinition* network, ILayer* plugin, IPluginContainer& f)
                            {
                                // Slice for the first convo.
                                auto slice_1_plugin_layer = addConv3DTranSliceLayer(out_dims1, network, plugin, f);
                                auto transform_layer      = addConv3DTranPostProc(network, slice_1_plugin_layer, f);

                                 // Second convo.
                                auto conv_tran_2_plugin = f.createConv3DTransposePlugin(
                                    Conv3DType::kTensorFlow,
                                    w2_dims, out_dims2,
                                    {3, {2, 2, 2}}, {3, {0, 1, 1}}, {3, {0, 1, 1}},
                                    Weights{DataType::kFLOAT, w2.data(), (int64_t)w2.size() },
                                    Weights{DataType::kFLOAT, nullptr, 0 },
                                    "Conv3DTranspose_2");
                                 EXPECT_NE(conv_tran_2_plugin, nullptr);
                                 auto conv_tran_2_plugin_in = transform_layer->getOutput(0);
                                 auto conv_tran_2_plugin_layer = addPlugin(*network, &conv_tran_2_plugin_in, 1, conv_tran_2_plugin);
                                 EXPECT_NE(conv_tran_2_plugin_layer, nullptr);

                                // Slice for the second convo.
                                auto slice_2_plugin_layer = addConv3DTranSliceLayer(out_dims2, network, conv_tran_2_plugin_layer, f);
                                return slice_2_plugin_layer;
                            },
                            *factory);

    ASSERT_EQ(x.size(), actual.size());
    for (size_t i = 0; i < actual.size(); i++)
         EXPECT_NEAR(x[i], actual[i], 0.0001) << "Vectors 'x' and 'actual' differ at index " << i;
}

// -----------------------------------------------------------------
// CostVolumePlugin plugin tests.
// -----------------------------------------------------------------

TEST(CostVolumePluginTests, Basic)
{
    Dims left_dims;
    Dims right_dims;
    Dims cost_vol_dims;
    FloatVec left     = readBinaryFile(g_data_dir + "cost_vol_01_l.bin",  left_dims);
    FloatVec right    = readBinaryFile(g_data_dir + "cost_vol_01_r.bin",  right_dims);
    FloatVec cost_vol = readBinaryFile(g_data_dir + "cost_vol_01_cv.bin", cost_vol_dims);
    ASSERT_EQ(left_dims.nbDims,     4);
    ASSERT_EQ(right_dims.nbDims,    4);
    ASSERT_EQ(cost_vol_dims.nbDims, 5);

    auto factory = IPluginContainer::create(*g_logger);
    auto actual  = runPlugin(1, {{"left", left_dims, left}, {"right", right_dims, right}}, cost_vol_dims,
                             [&]
                             {
                                 return factory->createCostVolumePlugin(DataType::kFLOAT, CostVolumeType::kDefault, cost_vol_dims.d[1], "CostVolume");
                             },
                             [] (INetworkDefinition*, ILayer* plugin, IPluginContainer&) { return plugin; },
                             *factory);

    ASSERT_EQ(cost_vol.size(), actual.size());
    for (size_t i = 0; i < actual.size(); i++)
         EXPECT_FLOAT_EQ(cost_vol[i], actual[i]) << "Vectors 'actual' and 'cost_vol' differ at index " << i;
}

TEST(CostVolumePluginTests, Large)
{
    Dims left_dims;
    Dims right_dims;
    Dims cost_vol_dims;
    FloatVec left     = readBinaryFile(g_data_dir + "cost_vol_02_l.bin",  left_dims);
    FloatVec right    = readBinaryFile(g_data_dir + "cost_vol_02_r.bin",  right_dims);
    FloatVec cost_vol = readBinaryFile(g_data_dir + "cost_vol_02_cv.bin", cost_vol_dims);
    ASSERT_EQ(left_dims.nbDims,     4);
    ASSERT_EQ(right_dims.nbDims,    4);
    ASSERT_EQ(cost_vol_dims.nbDims, 5);

    auto factory = IPluginContainer::create(*g_logger);
    auto actual  = runPlugin(1, {{"left", left_dims, left}, {"right", right_dims, right}}, cost_vol_dims,
                             [&]
                             {
                                 return factory->createCostVolumePlugin(DataType::kFLOAT, CostVolumeType::kDefault, cost_vol_dims.d[1], "CostVolume");
                             },
                             [] (INetworkDefinition*, ILayer* plugin, IPluginContainer&) { return plugin; },
                             *factory);

    ASSERT_EQ(cost_vol.size(), actual.size());
    for (size_t i = 0; i < actual.size(); i++)
         EXPECT_FLOAT_EQ(cost_vol[i], actual[i]) << "Vectors 'actual' and 'cost_vol' differ at index " << i;
}

// Performance tests, should be run under nvprof.
// REVIEW alexeyk: add more details and script to run as well as parse nvprof results.
TEST(CostVolumePluginPerfTests, NVSmall)
{
    Dims in_dims{4, {1,      32, 161, 513}};
    Dims cv_dims{5, {1, 48,  64, 161, 513}};

    // REVIEW alexeyk: populate with random values.
    FloatVec left(DimsUtils::getTensorSize(in_dims));
    FloatVec right(DimsUtils::getTensorSize(in_dims));
    FloatVec cost_vol(DimsUtils::getTensorSize(cv_dims));

    auto factory = IPluginContainer::create(*g_logger);
    auto actual  = runPlugin(1, {{"left", in_dims, left}, {"right", in_dims, right}}, cv_dims,
                             [&]
                             {
                                 return factory->createCostVolumePlugin(DataType::kFLOAT, CostVolumeType::kDefault, cv_dims.d[1], "CostVolume");
                             },
                             [] (INetworkDefinition*, ILayer* plugin, IPluginContainer&) { return plugin; },
                             *factory);

    ASSERT_EQ(cost_vol.size(), actual.size());
}

// Cost volume - correlation version.
TEST(CorrCostVolumePluginTests, Basic)
{
    Dims left_dims;
    Dims right_dims;
    Dims cost_vol_dims;
    FloatVec left     = readBinaryFile(g_data_dir + "corr_cost_vol_01_l.bin",  left_dims);
    FloatVec right    = readBinaryFile(g_data_dir + "corr_cost_vol_01_r.bin",  right_dims);
    FloatVec cost_vol = readBinaryFile(g_data_dir + "corr_cost_vol_01_cv.bin", cost_vol_dims);
    ASSERT_EQ(left_dims.nbDims,     4);
    ASSERT_EQ(right_dims.nbDims,    4);
    ASSERT_EQ(cost_vol_dims.nbDims, 5);

    auto factory = IPluginContainer::create(*g_logger);
    auto actual  = runPlugin(1, {{"left", left_dims, left}, {"right", right_dims, right}}, cost_vol_dims,
                             [&]
                             {
                                 return factory->createCostVolumePlugin(DataType::kFLOAT, CostVolumeType::kCorrelation,
                                                                        cost_vol_dims.d[1], "CostVolume");
                             },
                             [] (INetworkDefinition*, ILayer* plugin, IPluginContainer&) { return plugin; },
                             *factory);

    ASSERT_EQ(cost_vol.size(), actual.size());
    for (size_t i = 0; i < actual.size(); i++)
         EXPECT_NEAR(cost_vol[i], actual[i], 0.000001) << "Vectors 'actual' and 'cost_vol' differ at index " << i;
}

TEST(CorrCostVolumePluginTests, BasicFP16NC2HW2)
{
    // REVIEW alexeyk: TRT 4.0 fails with assert when using FP16/NC2HW2 combination on platforms
    // with no native FP16 support. Should be fixed in the future release of TRT.
    auto builder  = createInferBuilder(*g_logger);
    bool has_fp16 = builder->platformHasFastFp16();
    builder->destroy();
    if (!has_fp16)
    {
        testing::internal::ColoredPrintf(testing::internal::COLOR_YELLOW,
                                         "[**********] Current platofrm does not have native FP16 support, so the test will be skipped.\n");
        return;
    }

    Dims left_dims;
    Dims right_dims;
    Dims cost_vol_dims;
    FloatVec left     = readBinaryFile(g_data_dir + "corr_cost_vol_01_l.bin",  left_dims);
    FloatVec right    = readBinaryFile(g_data_dir + "corr_cost_vol_01_r.bin",  right_dims);
    FloatVec cost_vol = readBinaryFile(g_data_dir + "corr_cost_vol_01_cv.bin", cost_vol_dims);
    ASSERT_EQ(left_dims.nbDims,     4);
    ASSERT_EQ(right_dims.nbDims,    4);
    ASSERT_EQ(cost_vol_dims.nbDims, 5);

    auto data_type = DataType::kHALF;
    auto factory = IPluginContainer::create(*g_logger);
    auto actual  = runPlugin(1, {{"left", left_dims, left}, {"right", right_dims, right}}, cost_vol_dims,
                             [&]
                             {
                                 return factory->createCostVolumePlugin(data_type, CostVolumeType::kCorrelation,
                                                                        cost_vol_dims.d[1], "CostVolume");
                             },
                             [] (INetworkDefinition*, ILayer* plugin, IPluginContainer&) { return plugin; },
                             *factory, data_type);

    ASSERT_EQ(cost_vol.size(), actual.size());
    for (size_t i = 0; i < actual.size(); i++)
        EXPECT_NEAR(cost_vol[i], actual[i], 0.01) << "Vectors 'actual' and 'cost_vol' differ at index " << i;
}

// -----------------------------------------------------------------
// SoftargmaxPlugin plugin tests.
// -----------------------------------------------------------------

TEST(SoftargmaxPluginTests, ArgMinBasic)
{
    Dims x_dims;
    Dims y_dims;
    FloatVec x = readBinaryFile(g_data_dir + "softargmax_01_x.bin", x_dims);
    FloatVec y = readBinaryFile(g_data_dir + "softargmax_01_y.bin", y_dims);
    ASSERT_EQ(x_dims.nbDims, 5);
    ASSERT_EQ(y_dims.nbDims, 4);

    auto factory = IPluginContainer::create(*g_logger);
    auto actual  = runPlugin(1, {{"x", x_dims, x}}, y_dims,
                             [&]
                             {
                                 return factory->createSoftargmaxPlugin(DataType::kFLOAT, SoftargmaxType::kMin, "Softargmax");
                             },
                             [] (INetworkDefinition*, ILayer* plugin, IPluginContainer&) { return plugin; },
                             *factory);

    ASSERT_EQ(y.size(), actual.size());
    for (size_t i = 0; i < actual.size(); i++)
         EXPECT_FLOAT_EQ(y[i], actual[i]) << "Vectors 'actual' and 'y' differ at index " << i;
}

TEST(SoftargmaxPluginTests, ArgMinBatchSize2)
{
    Dims x_dims;
    Dims y_dims;
    FloatVec x = readBinaryFile(g_data_dir + "softargmax_02_x.bin", x_dims);
    FloatVec y = readBinaryFile(g_data_dir + "softargmax_02_y.bin", y_dims);
    ASSERT_EQ(x_dims.nbDims, 5);
    ASSERT_EQ(y_dims.nbDims, 4);

    auto factory = IPluginContainer::create(*g_logger);
    auto actual  = runPlugin(2, {{"x", x_dims, x}}, y_dims,
                             [&]
                             {
                                 return factory->createSoftargmaxPlugin(DataType::kFLOAT, SoftargmaxType::kMin, "Softargmax");
                             },
                             [] (INetworkDefinition*, ILayer* plugin, IPluginContainer&) { return plugin; },
                             *factory);

    ASSERT_EQ(y.size(), actual.size());
    for (size_t i = 0; i < actual.size(); i++)
         EXPECT_NEAR(y[i], actual[i], 0.00001) << "Vectors 'actual' and 'y' differ at index " << i;
}

TEST(SoftargmaxPluginTests, ArgMaxBasic)
{
    Dims x_dims;
    Dims y_dims;
    FloatVec x = readBinaryFile(g_data_dir + "softargmax_03_x.bin", x_dims);
    FloatVec y = readBinaryFile(g_data_dir + "softargmax_03_y.bin", y_dims);
    ASSERT_EQ(x_dims.nbDims, 5);
    ASSERT_EQ(y_dims.nbDims, 4);

    auto factory = IPluginContainer::create(*g_logger);
    auto actual  = runPlugin(1, {{"x", x_dims, x}}, y_dims,
                             [&]
                             {
                                 return factory->createSoftargmaxPlugin(DataType::kFLOAT, SoftargmaxType::kMax, "Softargmax");
                             },
                             [] (INetworkDefinition*, ILayer* plugin, IPluginContainer&) { return plugin; },
                             *factory);

    ASSERT_EQ(y.size(), actual.size());
    for (size_t i = 0; i < actual.size(); i++)
         EXPECT_FLOAT_EQ(y[i], actual[i]) << "Vectors 'actual' and 'y' differ at index " << i;
}

// -----------------------------------------------------------------
// End of tests.
// -----------------------------------------------------------------

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    if (argc < 2)
    {
        std::cout << "Usage: nvstereo_tests <test_data_directory> [TRT log level: 0|1|2|3]" << std::endl;
        return 1;
    }

    // Set tests data directory.
    g_data_dir = argv[1];
    assert(g_data_dir.size() > 0);
    if (g_data_dir[g_data_dir.size() - 1] != '/')
        g_data_dir += '/';

    // Create logger.
    auto max_log_level = ILogger::Severity::kWARNING;
    if (argc > 2)
    {
        max_log_level = (ILogger::Severity)std::stoi(argv[2]);
        assert((int)max_log_level < EnumMax<ILogger::Severity>());
    }
    g_logger = std::make_unique<Logger>(max_log_level);

    //getchar();
    return RUN_ALL_TESTS();
}
