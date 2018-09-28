// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include <unordered_map>

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>

namespace stereo_dnn_ros_viz
{

using ConstStr = const std::string;

static sensor_msgs::Image::ConstPtr s_cur_img_l = nullptr;
static sensor_msgs::Image::ConstPtr s_cur_img_r = nullptr;
static sensor_msgs::Image::ConstPtr s_cur_dnn   = nullptr;

cv::Mat preprocessImage(sensor_msgs::Image::ConstPtr img_msg, int dst_img_w, int dst_img_h)
{
    auto img = cv::Mat((int)img_msg->height, (int)img_msg->width,
                       img_msg->encoding == "bgra8" ? CV_8UC4 : CV_8UC3,
                       (void*)img_msg->data.data());
    // Handle encodings.
    if (img_msg->encoding == "bgr8")
        cv::cvtColor(img, img, CV_BGR2RGB);
    else if (img_msg->encoding == "bgra8")
        cv::cvtColor(img, img, CV_BGRA2RGB);
    //ROS_INFO("Dims: (%zu, %zu) -> (%zu, %zu)", w, h, (size_t)dst_img_w, (size_t)dst_img_h);
    // Convert to floating point type.
    img.convertTo(img, CV_32F);
    // Resize (anisotropically) to input layer size.
    cv::resize(img, img, cv::Size(dst_img_w, dst_img_h), 0, 0, cv::INTER_AREA);
    // Convert back to uint8.
    img.convertTo(img, CV_8UC3);
    return img;
}

// This is a primitive, unoptimized implementation of disparity colorization using KITTI color scheme. 
// Based on original implementation from KITTI SDK.
cv::Mat dispToColor(sensor_msgs::Image::ConstPtr disp, float max_disp)
{
    // Weights and cumsum are precomputed from Python code.
    const float weights[]{8.77192974, 5.40540552, 8.77192974, 5.74712658, 8.77192974, 5.40540552, 8.77192974, 0};
    const float cumsum[] {0,          0.114,      0.299,      0.413,      0.587,      0.70100003, 0.88600004, 1};
    const float w_map[][3]{{0, 0, 0}, {0, 0, 1}, {1, 0, 0}, {1, 0, 1},
                           {0, 1, 0}, {0, 1, 1}, {1, 1, 0}, {1, 1, 1}};
    const int   w_num = sizeof(cumsum) / sizeof(cumsum[0]);

    cv::Mat dst(disp->height, disp->width, CV_8UC3);
    auto p_dst = dst.ptr<uint8_t>(0);
    auto p_src = (float*)disp->data.data();
    for (int row = 0; row < (int)disp->height; row++)
    {
        for (int col = 0; col < (int)disp->width; col++)
        {
            float cur_disp = *p_src / max_disp;
            int index = 1;
            while (index < w_num && cur_disp > cumsum[index])
                index++;
            index--;
            float w = 1.0 - (cur_disp - cumsum[index]) * weights[index];
            p_dst[0] = (uint8_t)((w * w_map[index][0] + (1.0 - w) * w_map[index + 1][0]) * 255.0);
            p_dst[1] = (uint8_t)((w * w_map[index][1] + (1.0 - w) * w_map[index + 1][1]) * 255.0);
            p_dst[2] = (uint8_t)((w * w_map[index][2] + (1.0 - w) * w_map[index + 1][2]) * 255.0);
            p_dst += 3;
            p_src++;
        }
    }
    return dst;
}

sensor_msgs::Image::ConstPtr computeOutput()
{
    if (s_cur_img_l == nullptr || s_cur_img_r == nullptr || s_cur_dnn == nullptr)
        return nullptr;

    int c = 3;
    int h = (int)s_cur_dnn->height;
    int w = (int)s_cur_dnn->width;
    auto img_left  = preprocessImage(s_cur_img_l, w, h);
    auto img_right = preprocessImage(s_cur_img_r, w, h);

    auto viz_msg = boost::make_shared<sensor_msgs::Image>();
    // Set stamp and frame id to the same value as source image so we can synchronize with other nodes if needed.
    auto img_l = *s_cur_img_l;
    viz_msg->header.stamp.sec  = img_l.header.stamp.sec;
    viz_msg->header.stamp.nsec = img_l.header.stamp.nsec;
    viz_msg->header.frame_id   = img_l.header.frame_id;
    viz_msg->encoding = "rgb8";
    viz_msg->width    = 2 * w;
    viz_msg->height   = 2 * h;
    viz_msg->step     = c * viz_msg->width;
    size_t count      = viz_msg->step * viz_msg->height;
    viz_msg->data.resize(count);

    cv::Mat dst(viz_msg->height, viz_msg->width, CV_8UC3, viz_msg->data.data());

    img_left.copyTo( dst(cv::Rect(0, 0, w, h)));
    img_right.copyTo(dst(cv::Rect(w, 0, w, h)));

    // REVIEW alexeyk: hardcode max disp for now.
    float max_disp = 96;
    auto disp_color = dispToColor(s_cur_dnn, max_disp);
    disp_color.copyTo(dst(cv::Rect(w, h, w, h)));

    // Brighten up according to max disp.
    cv::Mat output(h, w, CV_32FC1, (void*)s_cur_dnn->data.data());
    output *= 255.0 / 96;
    output.convertTo(output, CV_8UC1);
    cv::cvtColor(output, output, CV_GRAY2RGB);
    output.copyTo(dst(cv::Rect(0, h, w, h)));

    // ROS_INFO("computeOutput: %u, %u, %s", viz_msg->width, viz_msg->height, viz_msg->encoding.c_str());

    // Set to null to mark as completed.
    s_cur_img_l = nullptr;
    s_cur_img_r = nullptr;
    s_cur_dnn   = nullptr;

    return viz_msg;
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg_l, const sensor_msgs::ImageConstPtr& msg_r, 
                   const sensor_msgs::ImageConstPtr& msg_dnn)
{
    auto img_d = *msg_dnn;
    // ROS_INFO("imageCallback: %u, %u, %s", img_l.width, img_l.height, img_l.encoding.c_str());
    // ROS_INFO("imageCallback: %u, %u, %s", img_r.width, img_r.height, img_r.encoding.c_str());
    if (msg_l->encoding != "rgb8" && msg_l->encoding != "bgr8" && msg_l->encoding != "bgra8")
    {
        ROS_FATAL("Image encoding %s is not yet supported. Supported encodings: rgb8, bgr8, bgra8", msg_l->encoding.c_str());
        ros::shutdown();
    }
    if (msg_r->encoding != "rgb8" && msg_r->encoding != "bgr8" && msg_r->encoding != "bgra8")
    {
        ROS_FATAL("Image encoding %s is not yet supported. Supported encodings: rgb8, bgr8, bgra8", msg_r->encoding.c_str());
        ros::shutdown();
    }
    if (msg_dnn->encoding != "32FC1")
    {
        ROS_FATAL("DNN encoding %s is not yet supported. Supported encodings: 32FC1", msg_r->encoding.c_str());
        ros::shutdown();
    }
    s_cur_img_l = msg_l;
    s_cur_img_r = msg_r;
    s_cur_dnn   = msg_dnn;
}

} // stereo_dnn_ros_viz

namespace sd = stereo_dnn_ros_viz;
namespace mf = message_filters;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "stereo_dnn_ros_viz");

    ROS_INFO("Starting Stereo DNN visualization ROS node...");
    ros::NodeHandle nh("~");

    std::string camera_topic_l;
    std::string camera_topic_r;
    std::string dnn_topic;
    std::string viz_topic;
    std::string model_type;
    std::string model_path;
    std::string data_type_s;
    int         in_queue_size;
    int         out_queue_size;
    float       max_rate_hz;

    nh.param<std::string>("camera_topic_left",  camera_topic_l, "/zed/left/image_rect_color");
    nh.param<std::string>("camera_topic_right", camera_topic_r, "/zed/right/image_rect_color");
    nh.param<std::string>("dnn_topic",          dnn_topic,      "/stereo_dnn_ros/network/output");
    nh.param<std::string>("viz_topic",          viz_topic,      "/stereo_dnn_ros_viz/output");

    nh.param("in_queue_size",  in_queue_size,  2);
    nh.param("out_queue_size", out_queue_size, 2);
    nh.param("max_rate_hz",    max_rate_hz,    30.0f);

    ROS_INFO("Camera L: %s", camera_topic_l.c_str());
    ROS_INFO("Camera R: %s", camera_topic_r.c_str());
    ROS_INFO("DNN     : %s", dnn_topic.c_str());
    ROS_INFO("Viz     : %s", viz_topic.c_str());
    ROS_INFO("In Q    : %d", in_queue_size);
    ROS_INFO("Out Q   : %d", out_queue_size);
    ROS_INFO("Rate    : %.1f", max_rate_hz);

    mf::Subscriber<sensor_msgs::Image> image_sub_l(nh, camera_topic_l, in_queue_size);
    mf::Subscriber<sensor_msgs::Image> image_sub_r(nh, camera_topic_r, in_queue_size);
    mf::Subscriber<sensor_msgs::Image> dnn_sub_r(  nh, dnn_topic,      in_queue_size);

    using MySyncPolicy = mf::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image>;
    mf::Synchronizer<MySyncPolicy> sync(MySyncPolicy(in_queue_size), image_sub_l, image_sub_r, dnn_sub_r);
    sync.registerCallback(boost::bind(&sd::imageCallback, _1, _2, _3));

    auto viz_pub = nh.advertise<sensor_msgs::Image>(viz_topic, out_queue_size);

    ros::Rate rate(max_rate_hz);
    ros::spinOnce();
    while (ros::ok())
    {
        auto out_msg = sd::computeOutput();
        if (out_msg != nullptr)
            viz_pub.publish(out_msg);
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}
