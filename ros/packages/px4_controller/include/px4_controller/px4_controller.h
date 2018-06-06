// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

/**
    px4_controller ROS node. Implements simple waypoint based controller for PX4/Mavros flightstack.
    It accepts input from either game controllers (Xbox and Shield) or from DNN that decides
    what direction the drone should fly. Once control signal is received it sets a waypoint at the
    right distance in correct direction. Also, allows finer grain controls over drone position.
    Authors/maintainers: Nikolai Smolyanskiy, Alexey Kamenev
*/

#ifndef PX4_CONTROLLER_PX4_CONTROLLER_H
#define PX4_CONTROLLER_PX4_CONTROLLER_H

#include <math.h>

#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Joy.h>
#include <sensor_msgs/Image.h>

#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/OverrideRCIn.h>

#include <tf/tf.h>
#include <tf2/buffer_core.h>
#include <tf2/LinearMath/Transform.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <angles/angles.h>
#include <eigen_conversions/eigen_msg.h>

#include <algorithm>
#include <memory>

namespace px4_control
{

class PX4Controller
{
public:
    PX4Controller();
    ~PX4Controller() = default;

    bool init(ros::NodeHandle& nh);
    bool arm();
    void spin();

    void px4StateCallback(const mavros_msgs::State::ConstPtr &msg);
    void poseCallback(const geometry_msgs::PoseStamped::ConstPtr &msg);
    void joystickCallback(const sensor_msgs::Joy::ConstPtr& msg);
    void dnnCallback(const sensor_msgs::Image::ConstPtr& msg);
    void objDnnCallback(const sensor_msgs::Image::ConstPtr& msg);

private:
    class Vehicle
    {
    public:
        virtual ~Vehicle() = default;
        virtual std::string getName() = 0;
        virtual std::string getOffboardModeName() = 0;
        virtual bool init(ros::NodeHandle& nh) = 0;
        virtual void printArgs() {}
        virtual void executeCommand(const PX4Controller& ctl, const geometry_msgs::PoseStamped& goto_pose,
                                    float linear_control_val, float angular_control_val, bool has_command) = 0;
        //virtual arm() = 0;
    protected:
        bool is_initialized_ = false;
    };

    class Drone: public Vehicle
    {
        std::string getName()             override { return "Drone"; }
        std::string getOffboardModeName() override { return "OFFBOARD"; }
        bool init(ros::NodeHandle& nh) override;
        void executeCommand(const PX4Controller& ctl, const geometry_msgs::PoseStamped& goto_pose,
                            float /*linear_control_val*/, float /*angular_control_val*/, bool /*has_command*/) override;
    };

    class APMRoverRC: public Vehicle
    {
        std::string getName()             override { return "APMRoverRC"; }
        std::string getOffboardModeName() override { return "MANUAL"; }
        bool init(ros::NodeHandle& nh) override;
        void printArgs() override;
        void executeCommand(const PX4Controller& ctl, const geometry_msgs::PoseStamped& goto_pose,
                            float linear_control_val, float angular_control_val, bool has_command) override;
    private:
        float linear_speed_scale_ = 1;
        float turn_angle_scale_   = 1;
        ros::Publisher rc_pub_;
        int rc_steer_trim_ = 1500;
        int rc_steer_dz_   = 30;
        int rc_steer_min_  = 1100;
        int rc_steer_max_  = 1900;
        int rc_throttle_trim_ = 1500;
        int rc_throttle_dz_   = 30;
        int rc_throttle_min_  = 1100;
        int rc_throttle_max_  = 1900;
    };

    class APMRoverWaypoint: public Vehicle
    {
        std::string getName()             override { return "APMRoverWaypoint"; }
        std::string getOffboardModeName() override { return "GUIDED"; }
        bool init(ros::NodeHandle& nh) override;
        void executeCommand(const PX4Controller& ctl, const geometry_msgs::PoseStamped& goto_pose,
                            float /*linear_control_val*/, float /*angular_control_val*/, bool /*has_command*/) override;
    };

private:
    const int QUEUE_SIZE = 5;
    const int DNN_FRAME_WIDTH = 320;
    const int DNN_FRAME_HEIGHT = 180;
    const int CLASS_OBJ_STOP = 14;
    const float OBJ_STOP_HEIGHT_RATIO = 0.5f;

    enum class ControllerState
    {
        Noop,
        Armed,
        Takeoff,
        Navigating
    };

    bool parseArguments(const ros::NodeHandle& nh);
    bool setJoystickParams(std::string joy_type);
    void initAutopilot();
    void computeDNNControl(const float class_probabilities[6], float& linear_control_val, float& angular_control_val);
    float doExpSmoothing(float cur, float prev, float factor) const;
    float getPoseDistance(geometry_msgs::PoseStamped& pose1, geometry_msgs::PoseStamped& pose2) const;
    geometry_msgs::Quaternion getRotationTo(geometry_msgs::Point& position, geometry_msgs::Point& target_position) const;
    geometry_msgs::Point computeNextWaypoint(geometry_msgs::PoseStamped& current_pose,
                                             float linear_control_val, float angular_control_val, float linear_speed) const;

private:
    std::unique_ptr<Vehicle> vehicle_;

    ControllerState controller_state_ = ControllerState::Noop;
    mavros_msgs::State fcu_state_;
    int command_queue_size_      = 5;
    float spin_rate_             = 20.0;
    float wait_for_arming_sec_   = 30.0;
    float takeoff_altitude_gain_ = 1.5f;
    float position_tolerance_    = 0.3f;
    int   dnn_class_count_       = 6;       // how many dnn classes are used for control
    float dnn_turn_angle_        = 10.0f;   // how much dnn turns each time to keep orientation (in degrees)
    float dnn_lateralcorr_angle_ = 10.0f;   // how much dnn turns each time to keep middle of the path position (in degrees)
    float smoothing_factor_      = 0.9f;
    float direction_filter_innov_coeff_ = 1.0f; // 0..1.0f how much of new control to integrate to the current command
    ros::Time timeof_last_logmsg_;

    // System state
    bool is_moving_ = false;
    geometry_msgs::PoseStamped current_pose_;   // updated in the MAVROS local pose subsriber
    float linear_speed_ = 2.0f;
    float altitude_     = 0;
    float turn_angle_   = 0;   // used for filtering
    long dnn_commands_count_;  // number of executed dnn commands
    long joy_commands_count_;  // number of executed teleop commands

    // Control commands
    // linear control and angular control should be coordinates of a point on a unit circle (controllers produce them naturally)
    // altitude control is in -1..1 range and can be combined with movement controls
    // yaw control is independent and blocks other movements
    bool  got_new_joy_command_  = false; // whether we've got a new command from Joystick/teleop
    float linear_control_val_   = 0;  // forward control: "+" is forward and "-" is back (-1..1), updated in the JOYSTICK subsriber
    float angular_control_val_  = 0; // turn control: "+" turns left and "-" turns right (-1..1), updated in the JOYSTICK subsriber
    float yaw_control_val_      = 0;     // yaw control: "+" rotates in place to left and "-" rotates right (-1..1)
    float altitude_control_val_ = 0; // altitude control: "+" is up and "-" is down (-1..1)
    ros::Time timeof_last_joy_command_;

    // DNN controls are used when enabled and when joystick is not in use
    bool  use_dnn_data_            = false; // whether to use data from DNN or not.
    bool  got_new_dnn_command_     = false; // whether we've got a new command from DNN
    float dnn_linear_control_val_  = 0 ;    // dnn forward control: "+" is forward and "-" is back (-1..1), updated in the DNN subsriber
    float dnn_angular_control_val_ = 0;     // dnn turn control: "+" turns left and "-" turns right (-1..1), updated in the DNN subsriber
    ros::Time timeof_last_dnn_command_;

    // Object detection control
    float obj_det_limit_  = -1.0f;

    // Joystick axis
    int joystick_linear_axis_   = 4;
    int joystick_angular_axis_  = 3;
    int joystick_yaw_axis_      = 0;
    int joystick_altitude_axis_ = 1;
    float joystick_deadzone_    = 0.3f;
    // Joystick buttons
    int joystick_dnn_on_button_    = 0;
    int joystick_dnn_off_button_   = 1;
    int joystick_dnn_left_button_  = 4;
    int joystick_dnn_right_button_ = 5;

    // ROS pub/sub
    ros::Subscriber fcu_state_sub_;
    ros::Subscriber local_pose_sub_;
    ros::Publisher local_pose_pub_;
    ros::ServiceClient arming_client_;
    ros::ServiceClient setmode_client_;
    ros::Subscriber joy_sub_;
    ros::Subscriber dnn_sub_;
    ros::Subscriber objdnn_sub_;
};

}

#endif