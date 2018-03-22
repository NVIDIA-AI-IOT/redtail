// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include "caffe_ros/int8_calibrator.h"
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>

namespace fs = boost::filesystem;

namespace caffe_ros
{

Int8EntropyCalibrator::Int8EntropyCalibrator(ConstStr& src, ConstStr& calib_cache)
    : src_(src), calib_cache_(calib_cache)
{
    if (!src_.empty())
    {
        // If calibration source is not empty then it should be either directory
        // or single image file.
        if (!fs::is_directory(src_) && !fs::is_regular_file(src_))
        {
            ROS_FATAL("INT8 calibrator: not supported source \"%s\". Use directory or regular file name.", src_.c_str());
            ros::shutdown();
        }
        // Set calibration file cache if it's empty.
        if (calib_cache_.empty())
            calib_cache_ = (fs::path(src_) / "_int8_calib.cache").string();
        // Create file list.
        if (fs::is_directory(src_))
        {
            for(const auto& entry: boost::make_iterator_range(fs::directory_iterator(src_), {}))
                files_.push_back(entry.path().string());
        }
        else
            files_.push_back(src_);
    }
}

Int8EntropyCalibrator::~Int8EntropyCalibrator()
{
    if (img_d_ != nullptr)
        cudaFree(img_d_);
    img_d_ = nullptr;
}

bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings)
{
    if (files_.size() == 0)
        return false;

    auto file = files_.front();
    files_.pop_front();
    ROS_DEBUG("INT8 calibrator: using \"%s\" as a source.", file.c_str());
    auto img = cv::imread(file, cv::IMREAD_COLOR);
    auto img_h = preprocessImage(img, dims_.w(), dims_.h(), InputFormat::BGR, "bgr8", 1, 0);
    // Copy to the device.
    ROS_ASSERT(img_h.isContinuous());
    ROS_ASSERT(nbBindings == 1);
    ROS_ASSERT(std::string("data") == names[0]);

    size_t size = dims_.c() * dims_.h() * dims_.w() * sizeof(float);
    if (img_d_ == nullptr)
    {
        if (cudaMalloc((void**)&img_d_, size) != cudaSuccess)
        {
            ROS_ERROR("INT8 calibrator: could not allocate %zu bytes on the device, error: %u.", size, cudaGetLastError());
            return false;
        }
    }
    if (cudaMemcpy(img_d_, img_h.ptr<float>(0), size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        ROS_FATAL("INT8 calibrator: could not copy data to device, error: %u.", cudaGetLastError());
        return false;
    }
    bindings[0] = img_d_;
    return true;
}

const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length)
{
    // If calibration source is not empty then it takes precedence over cache
    // as we assume user has requested a calibration.
    if (!src_.empty() || !fs::exists(calib_cache_))
        return nullptr;

    ROS_INFO("Reading INT8 calibration cache from: %s", calib_cache_.c_str());
    // Open file and seek to the end.
    std::ifstream file(calib_cache_, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        ROS_ERROR("Could not open file %s for reading, INT8 cache will not be used.", calib_cache_.c_str());
        return nullptr;
    }
    // Get file size and read the data.
    length = file.tellg();
    file.seekg(0, std::ios_base::beg);
    char* data = new char[length];
    file.read(data, length);
    return data;
}

void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length)
{
    ROS_INFO("Writing INT8 calibration cache to: %s", calib_cache_.c_str());
    auto file = std::ofstream(calib_cache_, std::ios::binary);
    file.write((const char*)cache, length);
}

void Int8EntropyCalibrator::setInputDims(nvinfer1::DimsCHW dims)
{
    dims_ = dims;
    // Free image device cache.
    if (img_d_ != nullptr)
        cudaFree(img_d_);
    img_d_ = nullptr;
}

}