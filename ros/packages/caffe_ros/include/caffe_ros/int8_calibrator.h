// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#ifndef CAFFE_ROS_INT8_CALIBRATOR_H
#define CAFFE_ROS_INT8_CALIBRATOR_H

#include <NvInfer.h>
#include <string>
#include "internal_utils.h"

namespace caffe_ros
{
class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator
{
public:
    Int8EntropyCalibrator(ConstStr& src, ConstStr& calib_cache);
    ~Int8EntropyCalibrator();

    int  getBatchSize() const override { return 1; }
    bool getBatch(void* bindings[], const char* names[], int nbBindings) override;

    const void* readCalibrationCache(size_t& length) override;
    void writeCalibrationCache(const void* cache, size_t length) override;

    void setInputDims(nvinfer1::DimsCHW dims);

private:
    std::string src_;
    std::string calib_cache_;
    
    std::deque<std::string> files_;

    nvinfer1::DimsCHW dims_ = nvinfer1::DimsCHW(0, 0, 0);

    float* img_d_ = nullptr;
};

}

#endif