// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include "caffe_ros/caffe_ros.h"
#include "caffe_ros/tensor_net.h"

#include <opencv2/opencv.hpp>
#include <boost/format.hpp>

#include "caffe_ros/yolo_prediction.h"

namespace caffe_ros
{
CaffeRos::CaffeRos()
{
    ROS_INFO("Starting Caffe ROS node...");
    ros::NodeHandle nh("~");

    std::string camera_topic;
    std::string prototxt_path;
    std::string model_path;
    std::string input_layer;
    std::string output_layer;
    std::string inp_fmt;
    std::string post_proc;
    std::string data_type_s;
    std::string int8_calib_src;
    std::string int8_calib_cache;
    bool        use_FP16;
    float       inp_scale;
    float       inp_shift;
    int         camera_queue_size;
    int         dnn_queue_size;
    bool        use_cached_model;

    nh.param<std::string>("camera_topic",  camera_topic, "/camera/image_raw");
    nh.param<std::string>("prototxt_path", prototxt_path, "");
    nh.param<std::string>("model_path",    model_path, "");
    nh.param<std::string>("input_layer",   input_layer, "data");
    nh.param<std::string>("output_layer",  output_layer, "prob");
    nh.param<std::string>("inp_fmt",       inp_fmt, "BGR");
    nh.param<std::string>("post_proc",     post_proc, "");
    nh.param<std::string>("data_type",     data_type_s, "fp16");
    nh.param<std::string>("int8_calib_src",   int8_calib_src,   "");
    nh.param<std::string>("int8_calib_cache", int8_calib_cache, "");
    
    // Backward compatibility: (use_FP16 == false) means use FP32.
    nh.param("use_fp16",  use_FP16, true);
    data_type_s = use_FP16 ? data_type_s : "fp32";

    nh.param("inp_scale", inp_scale, 1.0f);
    nh.param("inp_shift", inp_shift, 0.0f);
    nh.param("camera_queue_size", camera_queue_size, DEFAULT_CAMERA_QUEUE_SIZE);
    nh.param("dnn_queue_size",    dnn_queue_size,    DEFAULT_DNN_QUEUE_SIZE);
    nh.param("obj_det_threshold", obj_det_threshold_, 0.15f);
    nh.param("iou_threshold",     iou_threshold_,     0.2f);
    nh.param("max_rate_hz",       max_rate_hz_, 30.0f);
    nh.param("debug_mode",        debug_mode_,      false);
    nh.param("use_cached_model",  use_cached_model, true);

    ROS_INFO("Camera: %s", camera_topic.c_str());
    ROS_INFO("Proto : %s", prototxt_path.c_str());
    ROS_INFO("Model : %s", model_path.c_str());
    ROS_INFO("Input : %s", input_layer.c_str());
    ROS_INFO("Output: %s", output_layer.c_str());
    ROS_INFO("In Fmt: %s", inp_fmt.c_str());
    ROS_INFO("DType : %s", data_type_s.c_str());
    ROS_INFO("Scale : %.4f", inp_scale);
    ROS_INFO("Shift : %.2f", inp_shift);
    ROS_INFO("Cam Q : %d", camera_queue_size);
    ROS_INFO("DNN Q : %d", dnn_queue_size);
    ROS_INFO("Post P: %s", post_proc.empty() ? "none" : post_proc.c_str());
    ROS_INFO("Obj T : %.2f", obj_det_threshold_);
    ROS_INFO("IOU T : %.2f", iou_threshold_);
    ROS_INFO("Rate  : %.1f", max_rate_hz_);
    ROS_INFO("Debug : %s", debug_mode_ ? "yes" : "no");
    ROS_INFO("INT8 calib src  : %s", int8_calib_src.c_str());
    ROS_INFO("INT8 calib cache: %s", int8_calib_cache.c_str());
    //
    ROS_WARN("The use_FP16 parameter is deprecated though still supported. "
             "Please use data_type instead as use_FP16 will be removed in future release.");

    setPostProcessing(post_proc);

    auto data_type = parseDataType(data_type_s);

    if (data_type == nvinfer1::DataType::kINT8)
        net_.createInt8Calibrator(int8_calib_src, int8_calib_cache);

    net_.loadNetwork(prototxt_path, model_path, input_layer, output_layer,
                     data_type, use_cached_model);
    net_.setInputFormat(inp_fmt);
    net_.setScale(inp_scale);
    net_.setShift(inp_shift);
    if (debug_mode_)
        net_.showProfile(true);

    image_sub_  = nh.subscribe<sensor_msgs::Image>(camera_topic, camera_queue_size, &CaffeRos::imageCallback, this);
    output_pub_ = nh.advertise<sensor_msgs::Image>("network/output", dnn_queue_size);
}

void CaffeRos::spin()
{
    // We currently use single-threaded version of spinner as we have only one topic
    // that we subscribe to: image feed from camera.
    // DNN needs only the most recent image and GPU can process only one batch at a time
    // so there is no need for more than one thread or async processing.
    // The async code is provided anyway but commented out.

    // The code is based on tutorial: http://wiki.ros.org/roscpp/Overview/Callbacks%20and%20Spinning
    // ros::AsyncSpinner spinner(4); // Use 4 threads
    // spinner.start();
    // ros::waitForShutdown();
    // ROS_INFO("Caffe ROS node is stopped.");

    ros::Rate rate(max_rate_hz_);
    ros::spinOnce();
    while (ros::ok())
    {
        auto out_msg = computeOutputs();
        if (out_msg != nullptr)
            output_pub_.publish(out_msg);
        ros::spinOnce();
        rate.sleep();
    }
}

sensor_msgs::Image::ConstPtr CaffeRos::computeOutputs()
{
    if (cur_img_ == nullptr)
        return nullptr;

    auto img = *cur_img_;
    net_.forward(img.data.data(), img.width, img.height, img.encoding);
    auto out_msg = boost::make_shared<sensor_msgs::Image>();
    // Set stamp and frame id to the same value as source image so we can synchronize with other nodes if needed.
    out_msg->header.stamp.sec  = img.header.stamp.sec;
    out_msg->header.stamp.nsec = img.header.stamp.nsec;
    out_msg->header.frame_id   = img.header.frame_id;

    // Use single precision multidimensional array to represent outputs.
    // This can be useful in case DNN output is multidimensional such as in segmentation networks.
    // Note that encoding may not be compatible with other ROS code that uses Image in case number of channels > 4.
    // For classification nets, TensorRT uses 'c' dimension to reperesent number of classes.
    if (post_proc_ == PostProc::None)
    {
        out_msg->encoding = "32FC" + std::to_string(net_.getOutChannels());
        out_msg->width    = net_.getOutWidth();
        out_msg->height   = net_.getOutHeight();
        out_msg->step     = out_msg->width * net_.getOutChannels() * sizeof(float);
        size_t count      = out_msg->step * out_msg->height;
        auto ptr          = reinterpret_cast<const unsigned char*>(net_.getOutput());
        out_msg->data     = std::vector<unsigned char>(ptr, ptr + count);
    }
    else if (post_proc_ == PostProc::YOLO)
    {
        // Width and height are 1 in case of YOLO.
        ROS_ASSERT(net_.getOutWidth()  == 1);
        ROS_ASSERT(net_.getOutHeight() == 1);
        // Get bounding boxes and apply IOU filter.
        // REVIEW alexeyk: move magic constants to node arguments.
        auto preds = getYoloPredictions(net_.getOutput(), net_.getOutChannels(), img.width, img.height, obj_det_threshold_);
        preds = filterByIOU(preds, iou_threshold_);
        // Copy results to float array. Label and coords will be converted to float.
        const int num_col = 6;
        std::vector<float> msg_data(preds.size() * num_col);
        size_t i = 0;
        for (size_t row = 0; row < preds.size(); row++, i += num_col)
        {
            msg_data[i]     = (float)preds[row].label;
            msg_data[i + 1] = preds[row].prob;
            msg_data[i + 2] = (float)preds[row].x;
            msg_data[i + 3] = (float)preds[row].y;
            msg_data[i + 4] = (float)preds[row].w;
            msg_data[i + 5] = (float)preds[row].h;
        }
        ROS_ASSERT(i == msg_data.size());
        // Create message.
        // YOLO output is represented as a matrix where each row is
        // a predicted object vector of size 6: label, prob and 4 bounding box coordinates.
        out_msg->encoding = "32FC1";
        out_msg->width    = num_col;
        out_msg->height   = preds.size();
        out_msg->step     = out_msg->width * sizeof(float);
        size_t count      = out_msg->step * out_msg->height;
        auto ptr          = reinterpret_cast<const unsigned char*>(msg_data.data());
        out_msg->data     = std::vector<unsigned char>(ptr, ptr + count);
        ROS_ASSERT(msg_data.size() * sizeof(float) == count);
    }
    else
    {
        // Should not happen, yeah...
        ROS_FATAL("Invalid post processing.");
        ros::shutdown();
    }

    // Set to null to mark as completed.
    cur_img_ = nullptr;

    return out_msg;
}

void CaffeRos::imageCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    auto img = *msg;
    //ROS_DEBUG("imageCallback: %u, %u, %s", img.width, img.height, img.encoding.c_str());
    // Only RGB8 is currently supported.
    if (img.encoding != "rgb8" && img.encoding != "bgr8" && img.encoding != "bgra8")
    {
        ROS_FATAL("Image encoding %s is not yet supported. Supported encodings: rgb8, bgr8, bgra8", img.encoding.c_str());
        ros::shutdown();
    }
    cur_img_ = msg;
}

}