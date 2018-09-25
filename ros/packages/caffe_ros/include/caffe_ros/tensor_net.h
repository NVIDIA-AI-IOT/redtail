// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#ifndef CAFFE_ROS_TENSOR_NET_H
#define CAFFE_ROS_TENSOR_NET_H

#include <ros/ros.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
// REVIEW alexeyk: OpenCV that comes with JetPack 3.2 is compiled without CUDA support.
// #include <opencv2/core/cuda.hpp>

#include "int8_calibrator.h"

namespace caffe_ros
{

class TensorNet
{
public:
    TensorNet();
    virtual ~TensorNet();

    void loadNetwork(ConstStr& prototxt_path, ConstStr& model_path,
                     ConstStr& input_blob, ConstStr& output_blob,
                     nvinfer1::DataType data_type, bool use_cached_model);

    void forward(const unsigned char* input, size_t w, size_t h, const std::string& encoding);

    int getInWidth() const    { return in_dims_.w(); }
    int getInHeight() const   { return in_dims_.h(); }
    int getInChannels() const { return in_dims_.c(); }

    int getOutWidth() const    { return out_dims_.w(); }
    int getOutHeight() const   { return out_dims_.h(); }
    int getOutChannels() const { return out_dims_.c(); }

    const float* getOutput() const { return out_h_; }

    void setInputFormat(ConstStr& input_format)
    {
        if (input_format == "BGR")
            inp_fmt_ = InputFormat::BGR;
        else if (input_format == "RGB")
            inp_fmt_ = InputFormat::RGB;
        else
        {
            ROS_FATAL("Input format %s is not supported. Supported formats: BGR and RGB", input_format.c_str());
            ros::shutdown();
        }
    }

    void setShift(float shift) 
    { 
        assert(std::isfinite(shift));
        inp_shift_ = shift;
    }

    void setScale(float scale) 
    { 
        assert(std::isfinite(scale));
        inp_scale_ = scale;
    }

    void showProfile(bool on)
    {
        assert(context_ != nullptr);
        debug_mode_ = on;
        context_->setProfiler(on ? &s_profiler : nullptr);
    }

    void createInt8Calibrator(ConstStr& int8_calib_src, ConstStr& int8_calib_cache);

protected:

    void profileModel(ConstStr& prototxt_path, ConstStr& model_path, nvinfer1::DataType data_type,
                      ConstStr& input_blob, ConstStr& output_blob, std::ostream& model);

    class Logger : public nvinfer1::ILogger
    {
        void log(Severity severity, const char *msg) override;
    };
    static Logger s_log;

    class Profiler : public nvinfer1::IProfiler
    {
    public:
        void printLayerTimes();
    protected:
        void reportLayerTime(const char *layerName, float ms) override;
    private:
        using Record = std::pair<std::string, float>;
        std::vector<Record> profile_;
    };
    static Profiler s_profiler;

    nvinfer1::IRuntime*          infer_;
    nvinfer1::ICudaEngine*       engine_;
    nvinfer1::IExecutionContext* context_;
    
    nvinfer1::DimsCHW in_dims_;
    nvinfer1::DimsCHW out_dims_;

    // DNN input format.
    InputFormat inp_fmt_ = InputFormat::BGR;

    cv::Mat in_h_;
    cv::Mat in_final_h_;
    float* in_d_  = nullptr;
    float* out_h_ = nullptr;
    float* out_d_ = nullptr;

    float inp_shift_ = 0;
    float inp_scale_ = 1;

    // This is a separate flag from ROS_DEBUG to enable only specific profiling
    // of data preparation and DNN feed forward.
    bool debug_mode_ = false;

    std::unique_ptr<Int8EntropyCalibrator> int8_calib_;
};

nvinfer1::DataType parseDataType(const std::string& src);
std::string        toString(nvinfer1::DataType src);

}

#endif
