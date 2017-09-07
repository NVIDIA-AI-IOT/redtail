// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

/**
    px4_controller ROS node. Implements simple waypoint based controller for PX4/Mavros flightstack.
    It accepts input from either game controllers (Xbox and Shield) or from DNN that decides
    what direction the drone should fly. Once control signal is received it sets a waypoint at the
    right distance in correct direction. Also, allows finer grain controls over drone position.
    Authors/maintainers: Nikolai Smolyanskiy, Alexey Kamenev
*/

#include <ros/ros.h>
#include "px4_controller/px4_controller.h"

int main(int argc, char **argv)
{
    ROS_INFO("Starting px4_controller ROS node...");

    ros::init(argc, argv, "px4_controller");
    ros::NodeHandle nh("~");

    px4_control::PX4Controller controller;
    if(!controller.init(nh))
    {
        ROS_ERROR("Could not initialize PX4Controller node!");
        return -1;
    }

    if(!controller.arm())
    {
        ROS_ERROR("Could not arm PX4/FCU!");
        return -1;
    }

    // Loop and process commands/state
    controller.spin();

    return 0;
}
