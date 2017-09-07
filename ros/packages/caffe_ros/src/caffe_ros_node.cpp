// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include "caffe_ros/caffe_ros.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "caffe_ros");

    caffe_ros::CaffeRos caffe_r;
    caffe_r.spin();
    return 0;
}
