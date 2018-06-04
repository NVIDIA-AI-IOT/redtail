// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include <cuda_runtime_api.h>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <cudnn.h>
#include <opencv2/opencv.hpp>

#include "redtail_tensorrt_plugins.h"

#define UNUSED(x) ((void)(x))

#define CHECK(status) do {   \
    int res = (int)(status); \
    assert(res == 0);        \
    UNUSED(res);             \
} while(false)

using namespace nvinfer1;
using namespace redtail::tensorrt;

namespace redtail { namespace tensorrt
{
using weight_map = std::unordered_map<std::string, Weights>;

// NVSmall DNN: 1025x321 input, 96 max disparity.
INetworkDefinition* createNVSmall1025x321Network(IBuilder& builder, IPluginContainer& plugin_factory,
                                                 DimsCHW img_dims, const weight_map& weights, DataType data_type, ILogger& log);

// Tiny version of NVSmall DNN: 513x161 input, 48 max disparity.
INetworkDefinition* createNVTiny513x161Network(IBuilder& builder, IPluginContainer& plugin_factory,
                                               DimsCHW img_dims, const weight_map& weights, DataType data_type,
                                               ILogger& log);

// Baseline ResNet-18 DNN: 1025x321 input, 136 max disparity.
INetworkDefinition* createResNet18_1025x321Network(IBuilder& builder, IPluginContainer& plugin_factory,
                                                   DimsCHW img_dims, const weight_map& weights, DataType data_type,
                                                   ILogger& log);

// ResNet18_2D DNN: 513x256 input, 96 max disparity.
INetworkDefinition* createResNet18_2D_513x257Network(IBuilder& builder, IPluginContainer& plugin_factory,
                                                     DimsCHW img_dims, const weight_map& weights, DataType data_type, ILogger& log);

} }

class Logger : public nvinfer1::ILogger
{
    public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) override
    {
        // Skip info (verbose) messages.
        // if (severity == Severity::kINFO)
        //     return;

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
};

static Logger gLogger;

std::vector<float> readImgFile(const std::string& filename, int w, int h)
{
    auto img = cv::imread(filename);
    assert(img.data != nullptr);
    // 0. Convert to float.
    img.convertTo(img, CV_32F);
    // 1. Resize.
    cv::resize(img, img, cv::Size(w, h), 0, 0,cv::INTER_AREA);
    // 2. Convert BGR -> RGB.
    cv::cvtColor(img, img, CV_BGR2RGB);
    // 3. Convert HWC -> CHW.
    cv::Mat res = img.reshape(1, w * h).t();
    // 4. Scale.
    res /= 255.0;
    return std::vector<float>(res.ptr<float>(0), res.ptr<float>(0) + w * h * 3);
}

std::vector<float> readBinFile(const std::string& filename)
{
    std::ifstream input_file(filename, std::ios::binary | std::ios::ate);
    assert(input_file.is_open());
    size_t size = input_file.tellg();
    input_file.seekg(0, std::ios_base::beg);
    std::vector<float> data(size / sizeof(float));
    input_file.read((char*)data.data(), size);
    return data;
}

std::unordered_map<std::string, Weights> readWeights(const std::string& filename, DataType data_type)
{
    assert(data_type == DataType::kFLOAT || data_type == DataType::kHALF);

    std::unordered_map<std::string, Weights> weights;
    std::ifstream weights_file(filename, std::ios::binary);
    assert(weights_file.is_open());
    while (weights_file.peek() != std::ifstream::traits_type::eof())
    {
        std::string name;
        uint32_t    count;
        Weights     w {data_type, nullptr, 0};
        std::getline(weights_file, name, '\0');
        weights_file.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));
        w.count = count;
        size_t el_size_bytes = data_type == DataType::kFLOAT ? 4 : 2;
        auto p = new uint8_t[count * el_size_bytes];
        weights_file.read(reinterpret_cast<char*>(p), count * el_size_bytes);
        w.values = p;
        assert(weights.find(name) == weights.cend());
        weights[name] = w;
    }
    return weights;
}

int main(int argc, char** argv)
{
    if (argc < 8)
    {
        printf("\n"
               "Usage  : nvstereo_sample_app[_debug] <model_type> <width> <height> <path_to_weights_file> <path_to_left_image> <path_to_right_image> <disparity_output> [data_type]\n"
               "where  : model_type is the type of the DNN, supported are: nvsmall, resnet18, resnet18_2D\n"
               "         width and height are dimensions of the network (e.g. 1025 321)\n"
               "         weights file is the output of TensorRT model builder script\n"
               "         left and right are images that will be scaled to <width> x <height>\n"
               "         disparity output is the output of the network of size <width> x <height> (bin and PNG files are created)\n"
               "         data type(optional) is the data type of the model: fp32 (default) or fp16\n"
               "See <stereoDNN>/models directory for model files\n"
               "Example: nvstereo_sample_app nvsmall 1025 321 trt_weights.bin img_left.png img_right.png out_disp.bin\n\n");
        return 1;
    }
    //getchar();

    auto model_type = std::string(argv[1]);
    if (model_type != "nvsmall" && model_type != "resnet18" &&
        model_type != "resnet18_2D")
    {
        printf("Invalid model type %s, supported: nvsmall, resnet18, resnet18_2D.\n", model_type.c_str());
        exit(1);
    }

    DataType data_type = DataType::kFLOAT;
    if (argc >= 9)
    {
        auto d_type = std::string(argv[8]);
        if (d_type == "fp32")
            data_type = DataType::kFLOAT;
        else if (d_type == "fp16")
            data_type = DataType::kHALF;
        else
        {
            printf("Data type %s is not supported, supported types: fp32, fp16.\n", d_type.c_str());
            exit(1);
        }
    }
    printf("Using %s data type.\n", data_type == DataType::kFLOAT ? "fp32" : "fp16");

    // Read weights.
    // Note: the weights object lifetime must be at least the same as engine.
    auto weights = readWeights(argv[4], data_type);
    printf("Loaded %zu weight sets.\n", weights.size());

    //const int b = 1;
    const int c = 3;
    const int h = std::stoi(argv[3]);
    const int w = std::stoi(argv[2]);
    printf("Using [%d, %d](width, height) as network input dimensions.\n", w, h);

    // Read images.
    auto img_left  = readImgFile(argv[5], w, h);
    //auto img_left  = readBinFile(argv[5]);
    assert(img_left.size() == (size_t)c * h * w);
    auto img_right = readImgFile(argv[6], w, h);
    //auto img_right = readBinFile(argv[6]);
    assert(img_right.size() == (size_t)c * h * w);

    // Create builder and network.
    IBuilder* builder = createInferBuilder(gLogger);

    // Note: the plugin_factory object lifetime must be at least the same as engine.
    auto plugin_factory = IPluginContainer::create(gLogger);
    // TRT v3 supports FP16 only for the weights (e.g. convolutions) but not the data so use float data type.
    INetworkDefinition* network = nullptr;
    if (model_type == "nvsmall")
    {
        if (w == 1025)
            network = createNVSmall1025x321Network(*builder, *plugin_factory, DimsCHW { c, h, w }, weights, DataType::kFLOAT, gLogger);
        else if (w == 513)
            network = createNVTiny513x161Network(  *builder, *plugin_factory, DimsCHW { c, h, w }, weights, DataType::kFLOAT, gLogger);
        else
            assert(false);
    }
    else if (model_type == "resnet18")
    {
        if (w == 1025)
            network = createResNet18_1025x321Network(*builder, *plugin_factory, DimsCHW { c, h, w }, weights, DataType::kFLOAT, gLogger);
        else
        {
            printf("ResNet-18 model supports only 1025x321 input image.\n");
            exit(1);
        }
    }
    else if (model_type == "resnet18_2D")
    {
        if (w == 513)
            network = createResNet18_2D_513x257Network(*builder, *plugin_factory, DimsCHW { c, h, w }, weights, DataType::kFLOAT, gLogger);
        else
        {
            printf("ResNet18_2D model supports only 513x257 input image.\n");
            exit(1);
        }
    }
    else
        assert(false);

    builder->setMaxBatchSize(1);
    size_t workspace_bytes = 1024 * 1024 * 1024;
    builder->setMaxWorkspaceSize(workspace_bytes);

    builder->setHalf2Mode(data_type == DataType::kHALF);
    // Build the network.
    auto engine = builder->buildCudaEngine(*network);
    network->destroy();

    // REVIEW alexeyk: serialization is not yet supported. Need to implement IPluginFactory properly.
    // IHostMemory *model_stream = engine->serialize();
    // engine->destroy();
    // builder->destroy();

    // IRuntime* runtime = createInferRuntime(gLogger);
    // engine = runtime->deserializeCudaEngine(model_stream->data(), model_stream->size(), nullptr);
    // model_stream->destroy();

    assert(engine->getNbBindings() == 3);
    void* buffers[3];
    int in_idx_left = engine->getBindingIndex("left");
    assert(in_idx_left == 0);
    int in_idx_right = engine->getBindingIndex("right");
    assert(in_idx_right == 1);
    int out_idx = engine->getBindingIndex("disp");
    assert(out_idx == 2);

    IExecutionContext *context = engine->createExecutionContext();

    std::vector<float> output(h * w);

    // Allocate GPU memory and copy data.
    CHECK(cudaMalloc(&buffers[in_idx_left],  img_left.size() * sizeof(float)));
    CHECK(cudaMalloc(&buffers[in_idx_right], img_right.size() * sizeof(float)));
    CHECK(cudaMalloc(&buffers[out_idx],      output.size() * sizeof(float)));

    CHECK(cudaMemcpy(buffers[in_idx_left],  img_left.data(),  img_left.size() * sizeof(float),  cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(buffers[in_idx_right], img_right.data(), img_right.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Do the inference.
    auto host_start = std::chrono::high_resolution_clock::now();
    auto err        = context->execute(1, buffers);
    auto host_end   = std::chrono::high_resolution_clock::now();
    assert(err);
    UNUSED(err);
    auto host_elapsed_ms = std::chrono::duration<float, std::milli>(host_end - host_start).count();
    printf("Host time: %.4fms\n", host_elapsed_ms);

    // Copy output back to host.
    CHECK(cudaMemcpy(output.data(), buffers[out_idx], output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Write results.
    // 1. As binary file.
    auto res_file = std::ofstream(argv[7], std::ios::binary);
    res_file.write((char*)output.data(), output.size() * sizeof(float));
    // 2. As PNG image.
    auto img_f = cv::Mat(h, w, CV_32F, output.data());
    // Same as in KITTI, reduce quantization effects by storing as 16-bit PNG.
    img_f *= 256;
    // resnet18_2D model normalizes disparity using sigmoid, so bring it back to pixels.
    if (model_type == "resnet18_2D")
        img_f *= w;
    cv::Mat img_u16;
    img_f.convertTo(img_u16, CV_16U);
    cv::imwrite(std::string(argv[7]) + ".png", img_u16);

    // Cleanup.
    context->destroy();
    engine->destroy();
    for (auto b: buffers)
        CHECK(cudaFree(b));

    printf("Done\n");
    return 0;
}