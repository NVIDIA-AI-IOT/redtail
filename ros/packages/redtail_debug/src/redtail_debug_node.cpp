// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseStamped.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eigen_conversions/eigen_msg.h>

//#include <opencv2/opencv.hpp>

static sensor_msgs::Image::ConstPtr dnn_msg;

void dnnCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    if (dnn_msg != nullptr)
        return;
    dnn_msg = msg;
}

int main(int argc, char **argv)
{
    ROS_INFO("Starting redtail_debug ROS node...\n");

    ros::init(argc, argv, "redtail_debug");
    ros::NodeHandle nh("~");

    std::string caffe_ros_topic;
    float       pub_rate;
    nh.param<std::string>("caffe_ros_topic", caffe_ros_topic, "/trails_dnn/network/output");
    nh.param("pub_rate",  pub_rate, 30.0f);

    ROS_INFO("Topic : %s", caffe_ros_topic.c_str());
    ROS_INFO("Rate  : %.1f", pub_rate);

    const int queue_size = 10;
    ros::Subscriber dnn_sub;
    dnn_sub = nh.subscribe<sensor_msgs::Image>(caffe_ros_topic, queue_size, dnnCallback);

    ros::Publisher  debug_output_pub;
    debug_output_pub = nh.advertise<geometry_msgs::PoseStamped>("network/output_debug", queue_size);

    ros::Rate rate(pub_rate);
    while (ros::ok())
    {
        if (dnn_msg != nullptr)
        {
            auto pose_msg = boost::make_shared<geometry_msgs::PoseStamped>();
            pose_msg->header = dnn_msg->header;

            size_t dnn_out_size = dnn_msg->data.size() / sizeof(float);
            ROS_ASSERT(dnn_out_size * sizeof(float) == dnn_msg->data.size());
            ROS_ASSERT(dnn_out_size == 3 || dnn_out_size == 6 || dnn_out_size == 12);

            const float* probs = (const float*)(dnn_msg->data.data());

            // Orientation head.
            // Scale from -1..1 to -pi/2..pi/2. probs[0] is left turn, probs[2] - right.
            const float pi = std::acos(-1);
            float angle    = 0.5 * pi * (probs[0] - probs[2]);
            Eigen::Vector3d target_dir(std::cos(angle), std::sin(angle), 0);
            Eigen::Vector3d center_dir(1, 0, 0);
            geometry_msgs::Quaternion rotq_msg;
            tf::quaternionEigenToMsg(Eigen::Quaterniond::FromTwoVectors(center_dir, target_dir),
                                     rotq_msg);
            pose_msg->pose.orientation = rotq_msg;

            // Translation head.
            if (dnn_out_size >= 6)
                pose_msg->pose.position.y = probs[3] - probs[5];

            debug_output_pub.publish(pose_msg);
            dnn_msg = nullptr;
        }

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
