// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include <fstream>
#include <NvInfer.h>
#include <NvCaffeParser.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <cublas_v2.h>
#include "caffe_ros/tensor_net.h"

namespace caffe_ros
{

TensorNet::Logger   TensorNet::s_log;
TensorNet::Profiler TensorNet::s_profiler;

static nvinfer1::DimsCHW DimsToCHW(nvinfer1::Dims dims)
{
    ROS_ASSERT(dims.nbDims == 3);
    ROS_ASSERT(dims.type[0] == nvinfer1::DimensionType::kCHANNEL);
    ROS_ASSERT(dims.type[1] == nvinfer1::DimensionType::kSPATIAL);
    ROS_ASSERT(dims.type[2] == nvinfer1::DimensionType::kSPATIAL);
    return nvinfer1::DimsCHW(dims.d[0], dims.d[1], dims.d[2]);
}

void TensorNet::Logger::log(Severity severity, const char *msg)
{
    if (severity != Severity::kINFO)
        ROS_INFO("[TensorRT] %s", msg);
}

void TensorNet::Profiler::reportLayerTime(const char *layerName, float ms)
{
    auto record = std::find_if(profile_.begin(), profile_.end(), [&](const Record &r) { return r.first == layerName; });
    if (record == profile_.end())
        profile_.push_back(std::make_pair(layerName, ms));
    else
        record->second = ms;
}

void TensorNet::Profiler::printLayerTimes()
{
    float total_time = 0;
    for (size_t i = 0; i < profile_.size(); i++)
    {
        //ROS_INFO("%-40.40s %4.3fms", profile_[i].first.c_str(), profile_[i].second);
        total_time += profile_[i].second;
    }
    ROS_INFO("All layers  : %4.3f", total_time);
}

TensorNet::TensorNet()
{
}

TensorNet::~TensorNet()
{
    if (out_h_ != nullptr)
    {
        cudaError_t err = cudaFreeHost(out_h_);
        if (err != cudaSuccess)
            ROS_WARN("cudaFreeHost returned %d", (int)err);
        out_h_ = nullptr;
        out_d_ = nullptr;
    }
}

void TensorNet::profileModel(ConstStr& prototxt_path, ConstStr& model_path, bool use_FP16,
                             ConstStr& output_blob, std::ostream& model)
{
    auto builder = nvinfer1::createInferBuilder(s_log);
    auto network = builder->createNetwork();

    builder->setMinFindIterations(3); // allow time for TX1 GPU to spin up
    builder->setAverageFindIterations(2);

    auto parser = nvcaffeparser1::createCaffeParser();
    bool has_fast_FP16 = builder->platformHasFastFp16();
    ROS_INFO("Hardware support of fast FP16: %s.", has_fast_FP16 ? "yes" : "no");
    if (has_fast_FP16 && !use_FP16)
        ROS_INFO("... however, the model will be loaded as FP32.");
    
    nvinfer1::DataType model_data_type = (has_fast_FP16 && use_FP16) ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
    auto blob_finder = parser->parse(prototxt_path.c_str(), model_path.c_str(), *network, model_data_type);
    if (blob_finder == nullptr)
    {
        ROS_FATAL("Failed to parse network: %s, %s", prototxt_path.c_str(), model_path.c_str());
        ros::shutdown();
    }
    ROS_INFO("Loaded model from: %s, %s", prototxt_path.c_str(), model_path.c_str());
    
    auto out_b = blob_finder->find(output_blob.c_str());
    if (out_b == nullptr)
    {
        ROS_FATAL("Could not find output blob: %s", output_blob.c_str());
        ros::shutdown();
    }
    network->markOutput(*out_b);

    // Build model.
    // REVIEW alexeyk: make configurable?
    // Note: FP16 requires batch size to be even, TensorRT will switch automatically when building an engine.
    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(16 * 1024 * 1024);

    builder->setHalf2Mode(has_fast_FP16 && use_FP16);

    ROS_INFO("Building CUDA engine...");
    auto engine = builder->buildCudaEngine(*network);
    if (engine == nullptr)
    {
        ROS_FATAL("Failed to build CUDA engine.");
        ros::shutdown();
    }

    // Save model.
    nvinfer1::IHostMemory* model_ptr = engine->serialize();
    ROS_ASSERT(model_ptr != nullptr);
    model.write(reinterpret_cast<const char*>(model_ptr->data()), model_ptr->size());
    model_ptr->destroy();
    
    ROS_INFO("Done building.");

    // Cleanup.
    network->destroy();
    parser->destroy();
    engine->destroy();
    builder->destroy();
}

void TensorNet::loadNetwork(ConstStr& prototxt_path, ConstStr& model_path,
                            ConstStr& input_blob, ConstStr& output_blob, 
                            bool use_FP16, bool use_cached_model)
{
    infer_ = nvinfer1::createInferRuntime(s_log);
    if (infer_ == nullptr)
    {
        ROS_FATAL("Failed to create inference runtime.");
        ros::shutdown();
    }

    std::stringstream model;
    if (!use_cached_model)
        profileModel(prototxt_path, model_path, use_FP16, output_blob, model);
    else
    {
        std::string cached_model_path(model_path + ".cache");
        std::ifstream cached_model(cached_model_path, std::ios::binary);

        if (cached_model.good())
        {
            ROS_INFO("Loading cached model from: %s", cached_model_path.c_str());
            model << cached_model.rdbuf();
        }
        else
        {
            profileModel(prototxt_path, model_path, use_FP16, output_blob, model);
            ROS_INFO("Saving cached model to: %s", cached_model_path.c_str());
            std::ofstream cacheFile(cached_model_path, std::ios::binary);
            cacheFile << model.rdbuf();
        }
    }

    model.seekg(0, model.beg);
    const auto& model_final = model.str();

    engine_ = infer_->deserializeCudaEngine(model_final.c_str(), model_final.size(), nullptr);
    if (engine_ == nullptr)
    {
        ROS_FATAL("Failed to deserialize engine.");
        ros::shutdown();
    }

    context_ = engine_->createExecutionContext();
    if (context_ == nullptr)
    {
        ROS_FATAL("Failed to create execution context.");
        ros::shutdown();
    }
    ROS_INFO("Created CUDA engine and context.");

    int iinp = engine_->getBindingIndex(input_blob.c_str());
    in_dims_ = DimsToCHW(engine_->getBindingDimensions(iinp));
    ROS_INFO("Input : (W:%4u, H:%4u, C:%4u).", in_dims_.w(), in_dims_.h(), in_dims_.c());
    //cv::gpu::ensureSizeIsEnough(in_dims_.h(), in_dims_.w(), CV_8UC3, in_d_);
    in_d_ = cv::gpu::createContinuous(in_dims_.c(), in_dims_.w() * in_dims_.h(), CV_32FC1);
    assert(in_d_.isContinuous());
    
    int iout  = engine_->getBindingIndex(output_blob.c_str());
    out_dims_ = DimsToCHW(engine_->getBindingDimensions(iout));
    ROS_INFO("Output: (W:%4u, H:%4u, C:%4u).", out_dims_.w(), out_dims_.h(), out_dims_.c());

    // Allocate mapped memory for the outputs.
    size_t outSizeBytes = out_dims_.w() * out_dims_.h() * out_dims_.c() * sizeof(float);
    if (cudaHostAlloc(&out_h_, outSizeBytes, cudaHostAllocMapped) != cudaSuccess)
    {
        ROS_FATAL("Could not allocate %zu bytes for the output, error: %u.", outSizeBytes, cudaGetLastError());
        ros::shutdown();
    }
    if (cudaHostGetDevicePointer(&out_d_, out_h_, 0) != cudaSuccess)
    {
        ROS_FATAL("Could not get device pointer for the output, error: %u.", cudaGetLastError());
        ros::shutdown();
    }
}

void TensorNet::forward(const unsigned char* input, size_t w, size_t h, size_t c, const std::string& encoding)
{
    ROS_ASSERT(encoding == "rgb8" || encoding == "bgr8");
    ROS_ASSERT(c == (size_t)in_dims_.c());
    //ROS_DEBUG("Forward: input image is (%zu, %zu, %zu), network input is (%u, %u, %u)", w, h, c, in_dims_.w(), in_dims_.h(), in_dims_.c());

    // REVIEW alexeyk: extract to a separate methog/transformer class.
    // Perform image pre-processing (scaling, conversion etc).

    ros::Time start = ros::Time::now();

    in_h_ = cv::Mat((int)h, (int)w, CV_8UC3, (void*)input);
    // Handle encodings.
    if (inp_fmt_ == InputFormat::BGR)
    {
        // Convert image from RGB to BGR format used by OpenCV if needed.
        if (encoding == "rgb8")
            cv::cvtColor(in_h_, in_h_, CV_RGB2BGR);
    }
    else if (inp_fmt_ == InputFormat::RGB)
    {
        // Input image in OpenCV BGR, convert to RGB.
        if (encoding == "bgr8")
            cv::cvtColor(in_h_, in_h_, CV_BGR2RGB);
    }
    //ROS_INFO("Dims: (%zu, %zu) -> (%zu, %zu)", w, h, (size_t)in_dims_.w(), (size_t)in_dims_.h());
    // Convert to floating point type.
    in_h_.convertTo(in_h_, CV_32F);
    // Resize (anisotropically) to input layer size.
    cv::resize(in_h_, in_h_, cv::Size(in_dims_.w(), in_dims_.h()), 0, 0, cv::INTER_CUBIC);
    // Scale if needed.
    if (inp_scale_ != 1)
        in_h_ *= inp_scale_;
    // Shift if needed.
    if (inp_shift_ != 0)
        in_h_ = in_h_ + inp_shift_;
    // Transpose to get CHW format.
    in_final_h_ = in_h_.reshape(1, in_dims_.w() * in_dims_.h()).t();
    // Copy to the device.
    ROS_ASSERT(in_final_h_.isContinuous());
    ROS_ASSERT(in_d_.isContinuous());
    ROS_ASSERT(in_d_.size().area() * in_d_.channels() == in_final_h_.size().area() * in_final_h_.channels());
    if (cudaMemcpy(in_d_.ptr<float>(0), in_final_h_.ptr<float>(0), 
                   in_d_.size().area() * in_d_.channels() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        ROS_FATAL("Could not copy data to device, error: %u.", cudaGetLastError());
        ros::shutdown();
    }

    if (debug_mode_)
        ROS_INFO("Preproc time: %.3f", (ros::Time::now() - start).toSec() * 1000);

    // GPU version:
    // in_orig_d_.upload(cv::Mat((int)h, (int)w, CV_8UC3, (void*)input));
    // // Resize to input layer size.
    // cv::gpu::resize(in_orig_d_, in_d_, cv::Size(in_dims_.w(), in_dims_.h()), 0, 0, cv::INTER_CUBIC);
    // // Convert to floating point type.
    // in_d_.convertTo(in_f_d_, CV_32FC3);
    // // Subtract shift.
    // // REVIEW alexeyk: should be configurable as some models already have this in prototxt.
    // cv::gpu::subtract(in_f_d_, 128.0f, in_f_d_);
    // // Transpose to get CHW format.
    // ROS_DEBUG("in_f_d_: %zu", in_f_d_.elemSize());
    // cv::gpu::transpose(in_f_d_, in_f_d_);

    // cv::Mat cpuM;
    // in_d_.download(cpuM);
    // std::ofstream file("/home/ubuntu/tmp.raw", std::ios::binary);
    // file.write(cpuM.ptr<char>(0), std::streamsize(in_dims_.w() * in_dims_.h() * c));
    // file.close();

    void* bufs[] = {in_d_.ptr<float>(0), out_d_};
    context_->execute(1, bufs);
    if (debug_mode_)
        s_profiler.printLayerTimes();
    ROS_DEBUG("Forward out (first 3 values): [%.4f, %.4f, %.4f]", out_h_[0], out_h_[1], out_h_[2]);
}

}

