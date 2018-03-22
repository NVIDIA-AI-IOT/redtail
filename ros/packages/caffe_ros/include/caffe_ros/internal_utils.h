// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#ifndef CAFFE_ROS_INTERNAL_UTILS_H
#define CAFFE_ROS_INTERNAL_UTILS_H

#include <opencv2/opencv.hpp>

namespace caffe_ros
{
using ConstStr = const std::string;

// Formats of the input layer. BGR is usually used by most of the frameworks that use OpenCV.
enum class InputFormat
{
    BGR = 0,
    RGB
};

// Performs image preprocessing (resizing, scaling, format conversion etc)
// that is done before feeding the image into the networ.
cv::Mat preprocessImage(cv::Mat img, int dst_img_w, int dst_img_h, InputFormat inp_fmt, ConstStr& encoding,
                        float inp_scale, float inp_shift);

}

#endif
