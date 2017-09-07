// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#ifndef CAFFE_ROS_CAFFE_ROS_H
#define CAFFE_ROS_CAFFE_ROS_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include "caffe_ros/tensor_net.h"

namespace caffe_ros
{
// Implements caffe_ros node.
class CaffeRos
{
public:
    CaffeRos();
    ~CaffeRos() = default;

    void spin();

private:
    // Specifies whether to apply any post processing to the output of the DNN.
    enum class PostProc
    {
        None = 0,
        YOLO        // Compute object boxes from the output of YOLO DNN.
    };

    // Default camera queue size. Recommended value is one as to make 
    // sure we process most recent image from the camera.
    const int DEFAULT_CAMERA_QUEUE_SIZE = 1;
    // DNN output (publisher) queue. Value of 1 makes sure only most recent
    // output gets published.
    const int DEFAULT_DNN_QUEUE_SIZE    = 1;

    // Current image being worked on.
    sensor_msgs::Image::ConstPtr cur_img_;

    // Publisher for the DNN output.
    ros::Publisher  output_pub_;
    // Subscriber to camera capture topic (gscam).
    ros::Subscriber image_sub_;
    // DNN predictor.
    TensorNet net_;
    
    bool        debug_mode_;
    std::string debug_dir_;

    PostProc post_proc_;

    // Probability and IOU thresholds used in object detection net (YOLO).
    float obj_det_threshold_;
    float iou_threshold_;
    
    // Max rate to run the node at (in Hz).
    float max_rate_hz_;

private:
    sensor_msgs::Image::ConstPtr computeOutputs();

    void imageCallback(const sensor_msgs::Image::ConstPtr& msg);

    void setPostProcessing(const std::string& postProc)
    {
        if (postProc.size() == 0)
            post_proc_ = PostProc::None;
        else if (postProc == "YOLO")
            post_proc_ = PostProc::YOLO;
        else
        {
            ROS_FATAL("Post processing %s is not supported. Supported: YOLO", postProc.c_str());
            ros::shutdown();
        }
    }
    
};  
}

#endif