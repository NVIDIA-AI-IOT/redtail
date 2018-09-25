// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include <fstream>
#include <NvInfer.h>
#include <NvCaffeParser.h>
#include <cuda_runtime.h>
#include "caffe_ros/tensor_net.h"
#include <boost/algorithm/string.hpp>

namespace caffe_ros
{

using namespace nvinfer1;

TensorNet::Logger   TensorNet::s_log;
TensorNet::Profiler TensorNet::s_profiler;

static DimsCHW DimsToCHW(Dims dims)
{
    ROS_ASSERT(dims.nbDims == 3);
    ROS_ASSERT(dims.type[0] == DimensionType::kCHANNEL);
    ROS_ASSERT(dims.type[1] == DimensionType::kSPATIAL);
    ROS_ASSERT(dims.type[2] == DimensionType::kSPATIAL);
    return DimsCHW(dims.d[0], dims.d[1], dims.d[2]);
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
    cudaError_t err = cudaSuccess;
    if (in_d_ != nullptr)
    {
        err = cudaFree(in_d_);
        if (err != cudaSuccess)
            ROS_WARN("cudaFree returned %d", (int)err);
    }

    if (out_h_ != nullptr)
    {
        err = cudaFreeHost(out_h_);
        if (err != cudaSuccess)
            ROS_WARN("cudaFreeHost returned %d", (int)err);
    }
    in_d_  = nullptr;
    out_h_ = nullptr;
    out_d_ = nullptr;
}

void TensorNet::profileModel(ConstStr& prototxt_path, ConstStr& model_path, DataType data_type,
                             ConstStr& input_blob, ConstStr& output_blob, std::ostream& model)
{
    auto builder = createInferBuilder(s_log);
    auto network = builder->createNetwork();

    // Set TRT autotuner parameters.
    builder->setMinFindIterations(3);
    builder->setAverageFindIterations(2);

    auto parser = nvcaffeparser1::createCaffeParser();

    DataType model_data_type = DataType::kFLOAT;
    // Check for FP16.
    bool has_fast_FP16 = builder->platformHasFastFp16();
    ROS_INFO("Hardware support of fast FP16: %s.", has_fast_FP16 ? "yes" : "no");
    if (has_fast_FP16)
    {
        if (data_type == DataType::kHALF)
        {
            model_data_type = DataType::kHALF;
            builder->setHalf2Mode(true);
        }
        else
            ROS_INFO("... however, FP16 will not be used for this model.");
    }
    // Check for Int8.
    bool has_fast_int8 = builder->platformHasFastInt8();
    ROS_INFO("Hardware support of fast INT8: %s.", has_fast_int8 ? "yes" : "no");
    if (has_fast_int8)
    {
        if (data_type == DataType::kINT8)
        {
            ROS_ASSERT(int8_calib_ != nullptr);
            model_data_type = DataType::kINT8;
            builder->setInt8Mode(true);
            builder->setInt8Calibrator(int8_calib_.get());
        }
        else
            ROS_INFO("... however, INT8 will not be used for this model.");
    }

    ROS_INFO("Using %s model data type.", toString(model_data_type).c_str());
    // Note: for INT8 models parsing, data type must be set to FP32, see TRT docs.
    auto blob_finder = parser->parse(prototxt_path.c_str(), model_path.c_str(), *network,
                                     model_data_type == DataType::kINT8 ? DataType::kFLOAT : model_data_type);
    if (blob_finder == nullptr)
    {
        ROS_FATAL("Failed to parse network: %s, %s", prototxt_path.c_str(), model_path.c_str());
        ros::shutdown();
    }
    ROS_INFO("Loaded model from: %s, %s", prototxt_path.c_str(), model_path.c_str());

    // Need to set input dimensions for INT8 calibrator.
    if (data_type == DataType::kINT8)
    {
        auto in_b = blob_finder->find(input_blob.c_str());
        if (in_b == nullptr)
        {
            ROS_FATAL("Could not find input blob: %s", input_blob.c_str());
            ros::shutdown();
        }
        int8_calib_->setInputDims(DimsToCHW(in_b->getDimensions()));
    }

    // Find output blob and mark it as a network output.
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

    ROS_INFO("Building CUDA engine...");
    auto engine = builder->buildCudaEngine(*network);
    if (engine == nullptr)
    {
        ROS_FATAL("Failed to build CUDA engine.");
        ros::shutdown();
    }

    // Save model.
    IHostMemory* model_ptr = engine->serialize();
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
                            DataType data_type, bool use_cached_model)
{
    infer_ = createInferRuntime(s_log);
    if (infer_ == nullptr)
    {
        ROS_FATAL("Failed to create inference runtime.");
        ros::shutdown();
    }

    std::stringstream model;
    if (!use_cached_model)
        profileModel(prototxt_path, model_path, data_type, input_blob, output_blob, model);
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
            profileModel(prototxt_path, model_path, data_type, input_blob, output_blob, model);
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
    size_t in_size_bytes = in_dims_.c() * in_dims_.h() * in_dims_.w() * sizeof(float);
    // Allocate memory for the inputs.
    if (cudaMalloc(&in_d_, in_size_bytes) != cudaSuccess)
    {
        ROS_FATAL("Could not allocate %zu bytes for the input, error: %u.", in_size_bytes, cudaGetLastError());
        ros::shutdown();
    }

    int iout  = engine_->getBindingIndex(output_blob.c_str());
    out_dims_ = DimsToCHW(engine_->getBindingDimensions(iout));
    ROS_INFO("Output: (W:%4u, H:%4u, C:%4u).", out_dims_.w(), out_dims_.h(), out_dims_.c());

    // Allocate mapped memory for the outputs.
    size_t out_size_bytes = out_dims_.w() * out_dims_.h() * out_dims_.c() * sizeof(float);
    if (cudaHostAlloc(&out_h_, out_size_bytes, cudaHostAllocMapped) != cudaSuccess)
    {
        ROS_FATAL("Could not allocate %zu bytes for the output, error: %u.", out_size_bytes, cudaGetLastError());
        ros::shutdown();
    }
    if (cudaHostGetDevicePointer(&out_d_, out_h_, 0) != cudaSuccess)
    {
        ROS_FATAL("Could not get device pointer for the output, error: %u.", cudaGetLastError());
        ros::shutdown();
    }
}

void TensorNet::forward(const unsigned char* input, size_t w, size_t h, const std::string& encoding)
{
    ROS_ASSERT(encoding == "rgb8" || encoding == "bgr8" || encoding == "bgra8");
    //ROS_DEBUG("Forward: input image is (%zu, %zu, %zu), network input is (%u, %u, %u)", w, h, c, in_dims_.w(), in_dims_.h(), in_dims_.c());

    // REVIEW alexeyk: extract to a separate methog/transformer class.
    // Perform image pre-processing (scaling, conversion etc).

    ros::Time start = ros::Time::now();

    in_h_ = cv::Mat((int)h, (int)w, encoding == "bgra8" ? CV_8UC4 : CV_8UC3, (void*)input);
    in_final_h_ = preprocessImage(in_h_, in_dims_.w(), in_dims_.h(), inp_fmt_, encoding, inp_scale_, inp_shift_);
    // Copy to the device.
    ROS_ASSERT(in_final_h_.isContinuous());
    if (cudaMemcpy(in_d_, in_final_h_.ptr<float>(0),
                   in_final_h_.size().area() * in_final_h_.channels() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        ROS_FATAL("Could not copy data to device, error: %u.", cudaGetLastError());
        ros::shutdown();
    }

    if (debug_mode_)
        ROS_INFO("Preproc time: %.3f", (ros::Time::now() - start).toSec() * 1000);

    void* bufs[] = {in_d_, out_d_};
    context_->execute(1, bufs);
    if (debug_mode_)
        s_profiler.printLayerTimes();
    ROS_DEBUG("Forward out (first 3 values): [%.4f, %.4f, %.4f]", out_h_[0], out_h_[1], out_h_[2]);
}

void TensorNet::createInt8Calibrator(ConstStr& int8_calib_src, ConstStr& int8_calib_cache)
{
    ROS_ASSERT(!int8_calib_src.empty() || !int8_calib_cache.empty());

    if (!int8_calib_src.empty())
        ROS_INFO("INT8 calibration is requested. This may take some time.");

    int8_calib_ = std::make_unique<Int8EntropyCalibrator>(int8_calib_src, int8_calib_cache);
}

cv::Mat preprocessImage(cv::Mat img, int dst_img_w, int dst_img_h, InputFormat inp_fmt, ConstStr& encoding,
                        float inp_scale, float inp_shift)
{
    // Handle encodings.
    if (inp_fmt == InputFormat::BGR)
    {
        // Convert image from RGB/BGRA to BGR format.
        if (encoding == "rgb8")
            cv::cvtColor(img, img, CV_RGB2BGR);
        else if (encoding == "bgra8")
            cv::cvtColor(img, img, CV_BGRA2BGR);
    }
    else if (inp_fmt == InputFormat::RGB)
    {
        // Input image in OpenCV BGR, convert to RGB.
        if (encoding == "bgr8")
            cv::cvtColor(img, img, CV_BGR2RGB);
        else if (encoding == "bgra8")
            cv::cvtColor(img, img, CV_BGRA2RGB);
    }
    //ROS_INFO("Dims: (%zu, %zu) -> (%zu, %zu)", w, h, (size_t)dst_img_w, (size_t)dst_img_h);
    // Convert to floating point type.
    img.convertTo(img, CV_32F);
    // Resize (anisotropically) to input layer size.
    cv::resize(img, img, cv::Size(dst_img_w, dst_img_h), 0, 0, cv::INTER_CUBIC);
    // Scale if needed.
    if (inp_scale != 1)
        img *= inp_scale;
    // Shift if needed.
    if (inp_shift != 0)
        img += inp_shift;
    // Transpose to get CHW format.
    return img.reshape(1, dst_img_w * dst_img_h).t();
}

DataType parseDataType(const std::string& src)
{
    if (boost::iequals(src, "FP32"))
        return DataType::kFLOAT;
    if (boost::iequals(src, "FP16"))
        return DataType::kHALF;
    if (boost::iequals(src, "INT8"))
        return DataType::kINT8;
    else
    {
        ROS_FATAL("Invalid data type: %s. Supported data types: FP32, FP16, INT8.", src.c_str());
        ros::shutdown();
        // Will not get here (well, should not).
        return (DataType)-1;
    }
}

std::string toString(DataType src)
{
    switch (src)
    {
    case DataType::kFLOAT:
        return "FP32";
    case DataType::kHALF:
        return "FP16";
    case DataType::kINT8:
        return "INT8";
    default:
        return "<Unknown>";
    }
}

}
