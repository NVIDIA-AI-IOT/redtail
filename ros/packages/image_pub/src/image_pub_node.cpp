// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include <ros/ros.h>

#include <camera_info_manager/camera_info_manager.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{
    ROS_INFO("Starting image_pub ROS node...\n");

    ros::init(argc, argv, "image_pub");
    ros::NodeHandle nh("~");

    std::string camera_topic;
    std::string camera_info_topic;
    std::string camera_info_url;
    std::string img_path;
    std::string frame_id;
    float       pub_rate;
    int         start_sec;
    bool        repeat;
    nh.param<std::string>("camera_topic",      camera_topic,      "/camera/image_raw");
    nh.param<std::string>("camera_info_topic", camera_info_topic, "/camera/camera_info");
    nh.param<std::string>("camera_info_url",   camera_info_url,   "");
    nh.param<std::string>("img_path", img_path, "");
    nh.param<std::string>("frame_id", frame_id, "");
    nh.param("pub_rate",  pub_rate, 30.0f);
    nh.param("start_sec", start_sec, 0);
    nh.param("repeat",    repeat, false);

    ROS_INFO("CTopic : %s", camera_topic.c_str());
    ROS_INFO("ITopic : %s", camera_info_topic.c_str());
    ROS_INFO("CI URL : %s", camera_info_url.c_str());
    ROS_INFO("Source : %s", img_path.c_str());
    ROS_INFO("Rate   : %.1f", pub_rate);
    ROS_INFO("Start  : %d", start_sec);
    ROS_INFO("Repeat : %s", repeat ? "yes" : "no");
    ROS_INFO("FrameID: %s", frame_id.c_str());

    camera_info_manager::CameraInfoManager camera_info_manager(nh);
    if (camera_info_manager.validateURL(camera_info_url))
        camera_info_manager.loadCameraInfo(camera_info_url);

    ros::Publisher img_pub  = nh.advertise<sensor_msgs::Image>(camera_topic, 1);
    ros::Publisher info_pub = nh.advertise<sensor_msgs::CameraInfo>(camera_info_topic, 1);

    cv::VideoCapture vid_cap(img_path.c_str());
    if (start_sec > 0)
        vid_cap.set(CV_CAP_PROP_POS_MSEC, 1000.0 * start_sec);

    ros::Rate rate(pub_rate);
    while (ros::ok())
    {
        cv::Mat img;
        if (!vid_cap.read(img))
        {
            if (repeat)
            {
                vid_cap.open(img_path.c_str());
                if (start_sec > 0)
                    vid_cap.set(CV_CAP_PROP_POS_MSEC, 1000.0 * start_sec);
                continue;
            }
            ROS_ERROR("Failed to capture frame.");
            ros::shutdown();
        }
        else
        {
            //ROS_DEBUG("Image: %dx%dx%d, %zu, %d", img.rows, img.cols, img.channels(), img.elemSize(), img.type() == CV_8UC3);
            if (img.type() != CV_8UC3)
                img.convertTo(img, CV_8UC3);
            // Convert image from BGR format used by OpenCV to RGB.
            cv::cvtColor(img, img, CV_BGR2RGB);

            auto img_msg = boost::make_shared<sensor_msgs::Image>();
            img_msg->header.stamp    = ros::Time::now();
            img_msg->header.frame_id = frame_id;
            img_msg->encoding = "rgb8";
            img_msg->width = img.cols;
            img_msg->height = img.rows;
            img_msg->step = img_msg->width * img.channels();
            auto ptr = img.ptr<unsigned char>(0);
            img_msg->data = std::vector<unsigned char>(ptr, ptr + img_msg->step * img_msg->height);
            img_pub.publish(img_msg);

            if (camera_info_manager.isCalibrated())
            {
                auto info = boost::make_shared<sensor_msgs::CameraInfo>(camera_info_manager.getCameraInfo());
                info->header = img_msg->header;
                info_pub.publish(info);
            }
        }
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
