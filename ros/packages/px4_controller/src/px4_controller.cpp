// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

/**
    px4_controller ROS node. Implements simple waypoint based controller for PX4/Mavros flightstack.
    It accepts input from either game controllers (Xbox and Shield) or from DNN that decides
    what direction the drone should fly. Once control signal is received it sets a waypoint at the
    right distance in correct direction. Also, allows finer grain controls over drone position.
    Authors/maintainers: Nikolai Smolyanskiy, Alexey Kamenev
*/

#include "px4_controller/px4_controller.h"
#include <boost/algorithm/string.hpp>
#include <mavros_msgs/ParamGet.h>
#include <mavros_msgs/ParamSet.h>

namespace px4_control
{

PX4Controller::PX4Controller()
{
    timeof_last_logmsg_ = ros::Time::now();
    timeof_last_joy_command_ = ros::Time::now();
    timeof_last_dnn_command_ = ros::Time::now();

    initAutopilot();
}

bool PX4Controller::Drone::init(ros::NodeHandle& nh)
{
    is_initialized_ = true;
    return true;
}

void PX4Controller::Drone::executeCommand(const PX4Controller& ctl, const geometry_msgs::PoseStamped& goto_pose,
                                          float /*linear_control_val*/, float /*angular_control_val*/, bool /*has_command*/)
{
    ROS_ASSERT(is_initialized_);
    // Publish pose update to MAVROS
    ctl.local_pose_pub_.publish(goto_pose);
}

bool PX4Controller::APMRoverRC::init(ros::NodeHandle& nh)
{
    nh.param("linear_speed_scale", linear_speed_scale_, 1.0f);
    nh.param("turn_angle_scale",   turn_angle_scale_,   1.0f);

    const int QUEUE_SIZE = 1;
    rc_pub_ = nh.advertise<mavros_msgs::OverrideRCIn>("/mavros/rc/override", QUEUE_SIZE);
    if(!rc_pub_)
    {
        ROS_INFO("Could not advertise to /mavros/rc/override");
        return false;
    }

    auto get_param_client = nh.serviceClient<mavros_msgs::ParamGet>("/mavros/param/get");
    mavros_msgs::ParamGet param_get;
    param_get.request.param_id = "RC1_TRIM";
    if (get_param_client.call(param_get) && param_get.response.success)
        rc_steer_trim_ = (int)param_get.response.value.integer;
    nh.param("rc_steer_trim", rc_steer_trim_, rc_steer_trim_);

    param_get.request.param_id = "RC1_DZ";
    if (get_param_client.call(param_get) && param_get.response.success)
        rc_steer_dz_ = (int)param_get.response.value.integer;
    nh.param("rc_steer_dz", rc_steer_dz_, rc_steer_dz_);

    param_get.request.param_id = "RC3_TRIM";
    if (get_param_client.call(param_get) && param_get.response.success)
        rc_throttle_trim_ = (int)param_get.response.value.integer;
    nh.param("rc_throttle_trim", rc_throttle_trim_, rc_throttle_trim_);

    param_get.request.param_id = "RC3_DZ";
    if (get_param_client.call(param_get) && param_get.response.success)
        rc_throttle_dz_ = (int)param_get.response.value.integer;
    nh.param("rc_throttle_dz", rc_throttle_dz_, rc_throttle_dz_);

    // APM requires setting SYSID_MYGCS to enable RC override.
    auto set_param_client = nh.serviceClient<mavros_msgs::ParamSet>("/mavros/param/set");
    mavros_msgs::ParamSet param_set;

    param_set.request.param_id = "SYSID_MYGCS";
    // REVIEW alexeyk: make a parameter?
    const int gcs_id = 1;
    param_set.request.value.integer = gcs_id;
    if (set_param_client.call(param_set) && param_set.response.success)
        ROS_INFO("(%s) Set SYSID_MYGCS to %d", getName().c_str(), gcs_id);
    else
        ROS_WARN("(%s) Failed to set SYSID_MYGCS to %d", getName().c_str(), gcs_id);

    is_initialized_ = true;
    return true;
}

void PX4Controller::APMRoverRC::printArgs()
{
    ROS_INFO("(%s) Speed scale      : %.1f", getName().c_str(), linear_speed_scale_);
    ROS_INFO("(%s) Turn angle scale : %.1f", getName().c_str(), turn_angle_scale_);
    ROS_INFO("(%s) Steer trim       : %d",   getName().c_str(), rc_steer_trim_);
    ROS_INFO("(%s) Steer deadzone   : %d",   getName().c_str(), rc_steer_dz_);
    ROS_INFO("(%s) Steer min        : %d",   getName().c_str(), rc_steer_min_);
    ROS_INFO("(%s) Steer max        : %d",   getName().c_str(), rc_steer_max_);
    ROS_INFO("(%s) Throttle trim    : %d",   getName().c_str(), rc_throttle_trim_);
    ROS_INFO("(%s) Throttle deadzone: %d",   getName().c_str(), rc_throttle_dz_);
    ROS_INFO("(%s) Throttle min     : %d",   getName().c_str(), rc_throttle_min_);
    ROS_INFO("(%s) Throttle max     : %d",   getName().c_str(), rc_throttle_max_);
}

void PX4Controller::APMRoverRC::executeCommand(const PX4Controller& ctl, const geometry_msgs::PoseStamped& goto_pose,
                                             float linear_control_val, float angular_control_val, bool has_command)
{
    ROS_ASSERT(is_initialized_);

    // REVIEW alexeyk: should not use RC override.
    mavros_msgs::OverrideRCIn rc_override;
    for (int c = 0; c < 8; c++)
        rc_override.channels[c] = mavros_msgs::OverrideRCIn::CHAN_NOCHANGE;
    int steer_delta = turn_angle_scale_ * angular_control_val;
    int steer_dz    = steer_delta != 0 ? copysign(rc_steer_dz_, steer_delta) : 0;
    rc_override.channels[0] = rc_steer_trim_ + steer_dz + steer_delta;
    int throttle_delta = linear_speed_scale_ * ctl.linear_speed_ * linear_control_val;
    int throttle_dz    = throttle_delta != 0 ? copysign(rc_throttle_dz_, throttle_delta) : 0;
    rc_override.channels[2] = rc_throttle_trim_ + throttle_dz + throttle_delta;
    if(has_command)
    {
        ROS_DEBUG("APMRoverRC::executeCommand: %d, %d (%.2f, %.2f)", (int)rc_override.channels[0], (int)rc_override.channels[2], linear_control_val, angular_control_val);
        rc_pub_.publish(rc_override);
    }
}

bool PX4Controller::APMRoverWaypoint::init(ros::NodeHandle& nh)
{
    is_initialized_ = true;
    return true;
}

void PX4Controller::APMRoverWaypoint::executeCommand(const PX4Controller& ctl, const geometry_msgs::PoseStamped& goto_pose,
                                          float /*linear_control_val*/, float /*angular_control_val*/, bool /*has_command*/)
{
    ROS_ASSERT(is_initialized_);
    // Publish pose update to MAVROS
    ctl.local_pose_pub_.publish(goto_pose);
}


void PX4Controller::px4StateCallback(const mavros_msgs::State::ConstPtr &msg)
{
    fcu_state_ = *msg;
}

void PX4Controller::poseCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    // Update state, TODO: add spinlock if threaded?
    current_pose_ = *msg;
    // end of lock

    if(ros::Time::now() - timeof_last_logmsg_ > ros::Duration(1.0))
    {
        Eigen::Quaterniond current_orientation;
        tf::quaternionMsgToEigen(current_pose_.pose.orientation, current_orientation);
        Eigen::Matrix3d rotation_mat = current_orientation.toRotationMatrix();
        Eigen::Vector3d euler_angles = rotation_mat.eulerAngles(0,1,2);

        ROS_INFO("Pos:%4.2f, %4.2f, %4.2f, Att:%4.2f, %4.2f, %4.2f, SetAlt:%4.2f",
                 current_pose_.pose.position.x, current_pose_.pose.position.y, current_pose_.pose.position.z,
                 angles::to_degrees(euler_angles[0]), angles::to_degrees(euler_angles[1]), angles::to_degrees(euler_angles[2]),
                 altitude_);

        if(use_dnn_data_ && dnn_commands_count_!=0)
        {
            ROS_INFO(">>>>> AI score:%4.2f", (float)dnn_commands_count_ / float(dnn_commands_count_ + joy_commands_count_));
        }

        timeof_last_logmsg_ = ros::Time::now();
    }
}

void PX4Controller::joystickCallback(const sensor_msgs::Joy::ConstPtr& msg)
{
    float mov_stick_updown = msg->axes[joystick_linear_axis_];
    float mov_stick_leftright = msg->axes[joystick_angular_axis_];
    float pos_stick_updown = msg->axes[joystick_altitude_axis_];
    float pos_stick_leftright = msg->axes[joystick_yaw_axis_];

    ROS_DEBUG("JOY_VALS: l=%2.2f, a=%2.2f, y=%2.2f, alt=%2.2f, ltb=%d, rtb=%d, dnn_onb=%d, dnn_offb=%d",
              mov_stick_updown, mov_stick_leftright, pos_stick_leftright, pos_stick_updown,
              msg->buttons[joystick_dnn_left_button_], msg->buttons[joystick_dnn_right_button_],
              msg->buttons[joystick_dnn_on_button_], msg->buttons[joystick_dnn_off_button_]);

    // Update state, TODO: add spinlock if threaded?
    linear_control_val_   = fabs(mov_stick_updown)>joystick_deadzone_    ? mov_stick_updown : 0;
    angular_control_val_  = fabs(mov_stick_leftright)>joystick_deadzone_ ? mov_stick_leftright : 0;
    altitude_control_val_ = fabs(pos_stick_updown)>joystick_deadzone_    ? pos_stick_updown : 0;
    yaw_control_val_      = fabs(pos_stick_leftright)>joystick_deadzone_ ? pos_stick_leftright : 0;
    // end of lock

    // This code is to debug DNN actions in flight by simulating the same control!
    float class_probabilities[6] = {0,0,0,0,1,0};
    if(msg->buttons[joystick_dnn_left_button_] == 1)
    {
        class_probabilities[0] = 0.0f;
        class_probabilities[1] = 0.0f;
        class_probabilities[2] = 1.0f;
        computeDNNControl(class_probabilities, linear_control_val_, angular_control_val_);
    }
    else if(msg->buttons[joystick_dnn_right_button_] == 1)
    {
        class_probabilities[0] = 1.0f;
        class_probabilities[1] = 0.0f;
        class_probabilities[2] = 0.0f;
        computeDNNControl(class_probabilities, linear_control_val_, angular_control_val_);
    }
    // end of DNN debug control code

    // Turn ON DNN control when A button is pressed, turn off DNN control when B button is pressed
    if(!use_dnn_data_ && msg->buttons[joystick_dnn_on_button_] == 1)
    {
        // We are switching to DNN control. Initialize it
        use_dnn_data_ = true;
        initAutopilot();
        ROS_INFO("DNN control is activated! Filter innovation coeff=%4.2f", direction_filter_innov_coeff_);
    }
    else if(use_dnn_data_ && msg->buttons[joystick_dnn_off_button_] == 1)
    {
        use_dnn_data_ = false;
        ROS_INFO("DNN control is de-activated!");
    }

    if(linear_control_val_ != 0 || angular_control_val_ != 0 || yaw_control_val_ != 0 || altitude_control_val_ != 0)
    {
        ROS_INFO("joy controls: lin=%f, ang=%f, yaw=%f, alt=%f, use_dnn=%d",
                 linear_control_val_, angular_control_val_, yaw_control_val_, altitude_control_val_, use_dnn_data_);
    }
    timeof_last_joy_command_ = ros::Time::now();
    got_new_joy_command_ = true;
}

void PX4Controller::dnnCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    std::string expected_encoding("32FC3"); // 3 channel float
    if(dnn_class_count_ == 6)
    {
        expected_encoding = std::string("32FC6"); // 6 channel float
    }

    if(msg->width != 1 || msg->height != 1 || expected_encoding.compare(msg->encoding) != 0)
    {
        ROS_INFO("DNN CALLBACK ERROR: This node expects to receive width=1,height=1,encoding=%s", expected_encoding.c_str());
        assert(false);
        return;
    }

    // Do not use DNN outputs until operator presses "enable DNN" button (usually A).
    if (!use_dnn_data_)
    {
        got_new_dnn_command_ = false;
        return;
    }

    const float* probs = (const float*)(msg->data.data());
    float class_probabilities[6] = { probs[0], probs[1], probs[2], 0.0f, 1.0f, 0.0f };

    if(dnn_class_count_ == 6)
    {
        class_probabilities[3] = probs[3];
        class_probabilities[4] = probs[4];
        class_probabilities[5] = probs[5];
    }
    ROS_INFO("DNN state/message: on=%d, %4.2f, %4.2f, %4.2f, %4.2f, %4.2f, %4.2f", (int)use_dnn_data_,
             class_probabilities[0], class_probabilities[1], class_probabilities[2],
             class_probabilities[3], class_probabilities[4], class_probabilities[5]);

    // If we have no joystick available OR no control input from joystick, then follow DNN directions.
    computeDNNControl(class_probabilities, dnn_linear_control_val_, dnn_angular_control_val_);
    ROS_DEBUG("dnn controls: linear=%f, angular=%f", dnn_linear_control_val_, dnn_angular_control_val_);
    timeof_last_dnn_command_ = ros::Time::now();
    got_new_dnn_command_ = true;
}

void PX4Controller::objDnnCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    if (obj_det_limit_==-1.0f)
    {
        // If disabled
        return;
    }

    std::string expected_encoding("32FC1"); // 1 channel float array
    const unsigned int elem_count = 6;

    if(msg->width!=elem_count || expected_encoding.compare(msg->encoding)!=0)
    {
        ROS_INFO("OBJ DNN CALLBACK ERROR: This node expects to receive width=6,encoding=%s", expected_encoding.c_str());
        assert(false);
        return;
    }

    const float* objects = (const float*)(msg->data.data());
    unsigned int obj_count = msg->height;

    // Do not use DNN outputs until operator presses "enable DNN" button (usually A).
    if (!use_dnn_data_)
    {
        return;
    }

    int obj_class = -1;
    float obj_prob = -1;
    int obj_x = -1;
    int obj_y = -1;
    int obj_height = 0;
    int obj_width = 0;
    bool should_stop = false;
    for(unsigned int i = 0; i<obj_count; i++)
    {
        float class_id  = objects[i*elem_count + 0];
        float prob      = objects[i*elem_count + 1];
        float x         = objects[i*elem_count + 2];
        float y         = objects[i*elem_count + 3];
        float w         = objects[i*elem_count + 4];
        float h         = objects[i*elem_count + 5];

        // Stop if object's height more than some limit relative to dnn frame height
        // for correct class and sufficient probability
        if( (int)class_id==CLASS_OBJ_STOP &&
            prob >= obj_det_limit_ &&
            h/float(DNN_FRAME_HEIGHT)>OBJ_STOP_HEIGHT_RATIO
        )
        {
            should_stop = true;
            obj_class = (int)class_id;
            obj_prob = prob;
            obj_x = (int)x;
            obj_y = (int)y;
            obj_height = (int)h;
            obj_width = (int)w;
            break;
        }
    }

    if(should_stop)
    {
        use_dnn_data_ = false;  // Turn OFF AI control and stop
        linear_control_val_ = 0;
        angular_control_val_ = 0;
        ROS_INFO("OBJ DNN STOP DETECTED: class=%d, prob=%4.2f, x=%d, y=%d, width=%d, height=%d", obj_class, obj_prob, obj_x, obj_y, obj_width, obj_height);
        ROS_INFO("DNN control is de-activated!");
    }
}

void PX4Controller::computeDNNControl(const float class_probabilities[6], float& linear_control_val, float& angular_control_val)
{
    // Normalize probabilities just in case. We have 6 classes, they are disjoint - 1st 3 are rotations and 2nd 3 are translations
    float prob_sum = class_probabilities[0] + class_probabilities[1] + class_probabilities[2];
    assert(prob_sum!=0);
    float left_view_p   = class_probabilities[0] / prob_sum;
    float right_view_p  = class_probabilities[2] / prob_sum;

    prob_sum = class_probabilities[3] + class_probabilities[4] + class_probabilities[5];
    assert(prob_sum!=0);
    float left_side_p   = class_probabilities[3] / prob_sum;
    float right_side_p  = class_probabilities[5] / prob_sum;

    // Compute turn angle from probabilities. Positive angle - turn left, negative - turn right, 0 - go straight
    float current_turn_angle_deg =  dnn_turn_angle_*(right_view_p - left_view_p) + dnn_lateralcorr_angle_*(right_side_p - left_side_p);

    // Do sanity check and convert to radians
    current_turn_angle_deg = std::max(-90.0f, std::min(current_turn_angle_deg, 90.0f));   // just in case to avoid bad control
    float current_turn_angle_rad = (float)angles::from_degrees((float)current_turn_angle_deg);

    // Filter computed turning angle with the exponential filter
    turn_angle_ = turn_angle_*(1-direction_filter_innov_coeff_) + current_turn_angle_rad*direction_filter_innov_coeff_; // TODO: should this protected by a lock?
    float turn_angle_rad = turn_angle_;
    // end of turning angle filtering

    ROS_INFO("DNN turn angle: %4.2f deg.", (float)angles::to_degrees(turn_angle_rad));

    // Create control values that lie on a unit circle to mimic max joystick control values that are on a unit circle
    linear_control_val  = cosf(turn_angle_rad);
    angular_control_val = sinf(turn_angle_rad);
}

bool PX4Controller::parseArguments(const ros::NodeHandle& nh)
{
    nh.param("spin_rate", spin_rate_, 20.0f);
    ROS_ASSERT(spin_rate_ > 0);

    std::string vehicle_type;
    nh.param<std::string>("vehicle_type", vehicle_type, "drone");
    if (vehicle_type == "drone")
        vehicle_ = std::make_unique<Drone>();
    else if (vehicle_type == "apmroverrc")
        vehicle_ = std::make_unique<APMRoverRC>();
    else if (vehicle_type == "apmroverwaypoint")
        vehicle_ = std::make_unique<APMRoverWaypoint>();
    else
    {
        ROS_ERROR("Unknown vehicle type: %s", vehicle_type.c_str());
        return false;
    }

    std::string joy_type;
    nh.param<std::string>("joy_type", joy_type, "shield");
    if (!setJoystickParams(joy_type))
        return false;

    nh.param("command_queue_size", command_queue_size_, 5);
    if(command_queue_size_<0 || command_queue_size_>30.0f)
    {
        ROS_ERROR("Command queue size must be in 0..30 range!");
        return false;
    }

    nh.param("linear_speed", linear_speed_, 2.0f);
    nh.param("altitude_gain", takeoff_altitude_gain_, 0.0f); // Do not go up by default! We are taking off in manual mode!
    nh.param("dnn_class_count", dnn_class_count_, 6);
    if(dnn_class_count_!=3 && dnn_class_count_!=6)
    {
        ROS_ERROR("DNN class count used for control must be set to 3 or 6!");
        return false;
    }
    nh.param("dnn_turn_angle", dnn_turn_angle_, 10.0f);
    if(dnn_turn_angle_<0 || dnn_turn_angle_>90.0f)
    {
        ROS_ERROR("DNN turn angle parameter must be in 0..90 range!");
        return false;
    }
    nh.param("dnn_lateralcorr_angle", dnn_lateralcorr_angle_, 10.0f);
    if(dnn_lateralcorr_angle_<0 || dnn_lateralcorr_angle_>90.0f)
    {
        ROS_ERROR("DNN lateral correction angle parameter must be in 0..90 range!");
        return false;
    }

    nh.param("filter_innov_coeff", direction_filter_innov_coeff_, 1.0f);
    if(direction_filter_innov_coeff_<0 || direction_filter_innov_coeff_>1.0f)
    {
        ROS_ERROR("Direction filter innovation coefficient must be in 0..1 range!");
        return false;
    }
    nh.param("obj_det_limit", obj_det_limit_, -1.0f);
    if(obj_det_limit_!=-1.0f && (obj_det_limit_<0 || obj_det_limit_>1.0f))
    {
        ROS_ERROR("Object detection probability limit must be in 0..1 range or set to -1 to disable!");
        return false;
    }
    
    ROS_INFO("ROS spin rate                 : %.1f Hz", spin_rate_);
    ROS_INFO("Vehicle type                  : %s", vehicle_type.c_str());
    ROS_INFO("Joystick type                 : %s", joy_type.c_str());
    ROS_INFO("ROS command queue size        : %d", command_queue_size_);
    ROS_INFO("Linear speed                  : %.1f m/s", linear_speed_);
    ROS_INFO("Altitude gain                 : %.1f m", takeoff_altitude_gain_);
    ROS_INFO("DNN control class count       : %.d", dnn_class_count_);
    ROS_INFO("DNN turn angle (0..90 deg)    : %.1f deg", dnn_turn_angle_);
    ROS_INFO("DNN lat corr angle (0..90 deg): %.1f deg", dnn_lateralcorr_angle_);
    ROS_INFO("Direction filter innov coeff  : %4.2f", direction_filter_innov_coeff_);
    ROS_INFO("Object detection limit        : %2.2f", obj_det_limit_);

    return true;
}

bool PX4Controller::init(ros::NodeHandle& nh)
{
    if (!parseArguments(nh))
        return false;

    if (!vehicle_->init(nh))
    {
        ROS_ERROR("%s initialization failed.", vehicle_->getName().c_str());
        return false;
    }
    vehicle_->printArgs();

    auto now = ros::Time::now();
    timeof_last_logmsg_      = now;
    timeof_last_joy_command_ = now;
    timeof_last_dnn_command_ = now;

    // Mavros subsribers and publishers
    fcu_state_sub_ = nh.subscribe<mavros_msgs::State>("/mavros/state", QUEUE_SIZE, &PX4Controller::px4StateCallback, this);
    if(!fcu_state_sub_)
    {
        ROS_INFO("Could not subscribe to /mavros/state");
        return false;
    }

    local_pose_sub_ = nh.subscribe<geometry_msgs::PoseStamped>("/mavros/local_position/pose", QUEUE_SIZE, &PX4Controller::poseCallback, this);
    if(!local_pose_sub_)
    {
        ROS_INFO("Could not subscribe to /mavros/local_position/pose");
        return false;
    }

    local_pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/mavros/setpoint_position/local", QUEUE_SIZE);
    if(!local_pose_pub_)
    {
        ROS_INFO("Could not advertise to /mavros/setpoint_position/local");
        return false;
    }

    arming_client_ = nh.serviceClient<mavros_msgs::CommandBool>("/mavros/cmd/arming");
    setmode_client_ = nh.serviceClient<mavros_msgs::SetMode>("/mavros/set_mode");

    // Subscribe to a JOY (joystick) node if available
    joy_sub_ = nh.subscribe<sensor_msgs::Joy>("/joy", command_queue_size_, &PX4Controller::joystickCallback, this);
    if(joy_sub_)
    {
        ROS_INFO("Subscribed to /joy topic (joystick)");
    }

    // Subscribe to a higher level DNN based planner (if available)
    dnn_sub_ = nh.subscribe<sensor_msgs::Image>("/trails_dnn/network/output", command_queue_size_, &PX4Controller::dnnCallback, this);
    if(dnn_sub_)
    {
        ROS_INFO("Subscribed to /trails_dnn/network/output topic (DNN based planner)");
    }

    // Subscribe to an object detection DNN (if available)
    objdnn_sub_ = nh.subscribe<sensor_msgs::Image>("/object_dnn/network/output", command_queue_size_, &PX4Controller::objDnnCallback, this);
    if(objdnn_sub_)
    {
        ROS_INFO("Subscribed to /object_dnn/network/output topic (object detection DNN)");
    }

    controller_state_ = ControllerState::Noop;

    initAutopilot();

    return true;
}

void PX4Controller::initAutopilot()
{
    ROS_INFO("Initializing DNN Autopilot...");

    turn_angle_ = 0;

    dnn_commands_count_ = 0;
    joy_commands_count_ = 0;
}

bool PX4Controller::setJoystickParams(std::string joy_type)
{
    // Axes values: 1 = left/top, 0 = center, -1 = right/bottom
    boost::algorithm::to_lower(joy_type);
    if (joy_type == "shield" || joy_type == "xbox_wireless")
    {
        joystick_linear_axis_ = 3;
        joystick_angular_axis_ = 2;
        joystick_altitude_axis_ = 1;
        joystick_yaw_axis_ = 0;
    }
    else if (joy_type == "xbox_wired")
    {
        joystick_linear_axis_ = 4;
        joystick_angular_axis_ = 3;
        joystick_altitude_axis_ = 1;
        joystick_yaw_axis_ = 0;
    }
    else if (joy_type == "shield_2017")
    {
        joystick_linear_axis_ = 5;
        joystick_angular_axis_ = 2;
        joystick_altitude_axis_ = 1;
        joystick_yaw_axis_ = 0;
    }
    else
    {
        ROS_FATAL("Unsupported joystick type: %s. Supported types: shield, shield_2017, xbox_wireless, xbox_wired.",
                  joy_type.c_str());
        return false;
    }

    // These are the same on all supported controllers
    joystick_dnn_on_button_ = 0;
    joystick_dnn_off_button_ = 1;
    joystick_dnn_left_button_ = 4;
    joystick_dnn_right_button_ = 5;
    return true;
}

bool PX4Controller::arm()
{
    // The setpoint publishing rate MUST be faster than 2Hz
    ros::Rate rate(spin_rate_);

    // Wait for FCU board connection
    ROS_INFO("Waiting for FCU board...");
    while (ros::ok() && fcu_state_.connected)
    {
        ros::spinOnce();
        rate.sleep();
    }
    ROS_INFO("Connected to FCU board.");

    geometry_msgs::PoseStamped goto_pose;
    goto_pose.header.stamp    = ros::Time::now();
    goto_pose.header.frame_id = 1;
    goto_pose.header.seq      = 0;

    goto_pose.pose.position.x = 0;
    goto_pose.pose.position.y = 0;
    goto_pose.pose.position.z = 0;

    ROS_INFO("Getting current pose...");
    for (int i = 100; ros::ok() && i > 0; --i)
    {
        ros::spinOnce();
        rate.sleep();
        geometry_msgs::PoseStamped current_pose = current_pose_; // current_pose_ is updated elsewhere, use its fixed time value for computations
        goto_pose.pose.position.x = doExpSmoothing(current_pose.pose.position.x, goto_pose.pose.position.x, smoothing_factor_);
        goto_pose.pose.position.y = doExpSmoothing(current_pose.pose.position.y, goto_pose.pose.position.y, smoothing_factor_);
        goto_pose.pose.position.z = doExpSmoothing(current_pose.pose.position.z, goto_pose.pose.position.z, smoothing_factor_);
    }
    altitude_ = goto_pose.pose.position.z;

    // Send a few setpoints before starting. Note that this will run before current
    // position coordinates are obtained but that's ok as the vehicle is not armed/OFFBOARD yet.
    ROS_INFO("Sending warmup messages...");
    for (int i = 100; ros::ok() && i > 0; --i)
    {
        goto_pose.header.stamp    = ros::Time::now();
        goto_pose.header.frame_id = 1;
        local_pose_pub_.publish(goto_pose);

        ros::spinOnce();
        rate.sleep();
    }

    mavros_msgs::SetMode offb_setmode;
    offb_setmode.request.custom_mode = vehicle_->getOffboardModeName();

    mavros_msgs::CommandBool arm_cmd;
    arm_cmd.request.value = true;

    ros::Time last_request = ros::Time::now();
    ros::Time init_start   = ros::Time::now();
    ROS_INFO("Switching to %s and arming...", vehicle_->getOffboardModeName().c_str());
    while ( ros::ok() && (ros::Time::now() - init_start < ros::Duration(wait_for_arming_sec_)) )
    {
        geometry_msgs::PoseStamped current_pose = current_pose_; // current_pose_ is updated elsewhere, use its fixed time value for computations

        if (fcu_state_.mode != vehicle_->getOffboardModeName() && (ros::Time::now() - last_request > ros::Duration(5.0)))
        {
            // Enable OFFBOARD mode
            if (setmode_client_.call(offb_setmode) && offb_setmode.response.mode_sent)
            {
                ROS_INFO("Offboard enabled");
            }
            last_request = ros::Time::now();
        }
        else
        {
            if (!fcu_state_.armed && (ros::Time::now() - last_request > ros::Duration(5.0)))
            {
                // Arm FCU
                if (arming_client_.call(arm_cmd) && arm_cmd.response.success)
                {
                    ROS_INFO("Vehicle armed");
                    controller_state_ = ControllerState::Armed;
                    return true;
                }
                last_request = ros::Time::now();
            }
            else if(fcu_state_.armed)
            {
                ROS_INFO("Vehicle was already armed");
                controller_state_ = ControllerState::Armed;
                return true;
            }
        }

        // Keep updating start coordinates until the vehicle is in OFFBOARD mode and armed.
        // This prevents flying to 0,0,0 in some cases. goto_pose is initialized to real pose in the end
        if (fcu_state_.mode != vehicle_->getOffboardModeName() || !fcu_state_.armed)
        {
            goto_pose.pose.position.x = doExpSmoothing(current_pose.pose.position.x, goto_pose.pose.position.x, smoothing_factor_);
            goto_pose.pose.position.y = doExpSmoothing(current_pose.pose.position.y, goto_pose.pose.position.y, smoothing_factor_);
            goto_pose.pose.position.z = doExpSmoothing(current_pose.pose.position.z, goto_pose.pose.position.z, smoothing_factor_);
        }

        goto_pose.header.stamp = ros::Time::now();
        goto_pose.header.frame_id = 1;
        local_pose_pub_.publish(goto_pose);

        ros::spinOnce();
        rate.sleep();
    }

    return false;
}

void PX4Controller::spin()
{
    if(controller_state_ != ControllerState::Armed)
    {
        ROS_INFO("Cannot spin in PX4Controller node! Vehicle is not armed!");
        return;
    }

    // The setpoint publishing rate MUST be faster than 2Hz
    ros::Rate rate(spin_rate_);
    float distance = 0;

    Eigen::Quaterniond orientation_quaternion;
    Eigen::Matrix3d rotation_matrix;
    Eigen::Vector3d euler_angles;

    geometry_msgs::PoseStamped current_pose = current_pose_;

    geometry_msgs::PoseStamped goto_pose = current_pose;
    goto_pose.header.stamp    = ros::Time::now();
    goto_pose.header.frame_id = 1;
    goto_pose.header.seq      = 0;

    ROS_INFO("PX4Controller is processing requests...");

    while (ros::ok())
    {
        float linear_control_val   = 0;
        float angular_control_val  = 0;
        float yaw_control_val      = 0;
        float altitude_control_val = 0;

        bool has_command = false;

        // Get latest state and control inputs. 
        current_pose = current_pose_; // current_pose_ is updated elsewhere, use its fixed time value for computations

        switch (controller_state_)
        {
        case ControllerState::Armed:
            ROS_INFO("Armed mode.");
            goto_pose.pose.position.z += takeoff_altitude_gain_;
            controller_state_ = ControllerState::Takeoff;
            ROS_INFO("Switching to Takeoff...");
            break;

        case ControllerState::Takeoff:
            distance = getPoseDistance(current_pose, goto_pose);
            ROS_INFO("Takeoff mode. Distance to end point = %f", distance);
            if (distance <= position_tolerance_)
            {
                controller_state_ = ControllerState::Navigating;
                is_moving_ = true;
                altitude_ = current_pose.pose.position.z;
                ROS_INFO("Switching to Navigate. Altitude: %f", altitude_);
            }
            break;

        case ControllerState::Navigating:
            // Log
            tf::quaternionMsgToEigen(current_pose.pose.orientation, orientation_quaternion);
            rotation_matrix = orientation_quaternion.toRotationMatrix();
            euler_angles    = rotation_matrix.eulerAngles(0,1,2);
            ROS_DEBUG("NAVPOS: %4.2f, %4.2f, %4.2f, Att: %4.2f, %4.2f, %4.2f, SetAlt: %4.2f",
                      current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z,
                      angles::to_degrees(euler_angles[0]), angles::to_degrees(euler_angles[1]), angles::to_degrees(euler_angles[2]),
                      altitude_
            );

            if (fcu_state_.mode != vehicle_->getOffboardModeName())
            {
                // If offboard is off, don't move, just update goto_pose to be the current pose to avoid flyaway when offboard gets turned on!
                goto_pose = current_pose;
                break;
            }

            has_command = got_new_joy_command_ || got_new_dnn_command_;
            if(!use_dnn_data_)  // Use joystick/teleop control if DNN disabled
            {
                if(got_new_joy_command_)
                {
                    linear_control_val   = linear_control_val_;
                    angular_control_val  = angular_control_val_;
                    yaw_control_val      = yaw_control_val_;
                    altitude_control_val = altitude_control_val_;
                    got_new_joy_command_ = false;
                }
            }
            else // Use DNN control if joystick is not touched
            {
                if(got_new_joy_command_ && (linear_control_val_!=0 || angular_control_val_!=0 ||
                   yaw_control_val_!=0 || altitude_control_val_!=0))
                {
                    linear_control_val   = linear_control_val_;
                    angular_control_val  = angular_control_val_;
                    yaw_control_val      = yaw_control_val_;
                    altitude_control_val = altitude_control_val_;
                    got_new_joy_command_ = false;
                    joy_commands_count_++;
                }
                else if(got_new_dnn_command_)
                {
                    linear_control_val   = dnn_linear_control_val_;
                    angular_control_val  = dnn_angular_control_val_;
                    got_new_dnn_command_ = false;
                    dnn_commands_count_++;
                }
                else
                {
                    // We get here only when there is no commands from DNN or no joystick movements outside of deadzone.
                    // Clearing the flag is currently requried only for rover.
                    has_command = false;
                    break;
                }
            }

            // Log
            ROS_DEBUG("NAVC: st=%d, l=%2.2f, a=%2.2f, y=%2.2f, alt=%2.2f",
                      (int)controller_state_, linear_control_val, angular_control_val,
                      yaw_control_val, altitude_control_val);

            if(altitude_control_val != 0.0f)
            {
                altitude_ = altitude_ + 0.03f*altitude_control_val;
                goto_pose.pose.position.z = altitude_;
            }

            if(yaw_control_val != 0)
            {
                // Rotating in place (left joystick)
                angular_control_val = 0.3f*yaw_control_val;
                linear_control_val = sqrtf(1 - angular_control_val*angular_control_val); // so linear and angular controls are still on a unit circle
                // Use waypoint compute to compute direction, but turn in place!
                geometry_msgs::Point face_new_point = computeNextWaypoint(current_pose, linear_control_val,
                                                                          angular_control_val, 10.0 /*point at some distance*/);
                goto_pose.pose.orientation = getRotationTo(current_pose.pose.position, face_new_point);
            }
            else    // Moving (only right joystick)
            {
                if(linear_control_val == 0 && angular_control_val == 0)
                {
                    if(is_moving_)
                    {
                        goto_pose.pose.orientation = current_pose.pose.orientation;
                        goto_pose.pose.position.z  = altitude_;
                        goto_pose.pose.position.x  = current_pose.pose.position.x;
                        goto_pose.pose.position.y  = current_pose.pose.position.y;

                        is_moving_ = false;
                    }
                }
                else
                {
                    is_moving_ = true;

                    // Compute next waypoint based on current commands
                    geometry_msgs::Point next_waypoint = computeNextWaypoint(current_pose, linear_control_val,
                                                                             angular_control_val, linear_speed_);
                    next_waypoint.z = altitude_; // need to set Z so it holds the set altitude
                    goto_pose.pose.position = next_waypoint;

                    // Compute new orientation for the next waypoint if we are not going backwards or strafing
                    if(linear_control_val > 0)
                    {
                        goto_pose.pose.orientation = getRotationTo(current_pose.pose.position, next_waypoint);
                    }
                }
            }

            break;

        default:
            ROS_ERROR("PX4Controller::spin detected unknown state: %d!", (int)controller_state_);
            break;
        }

        goto_pose.header.stamp = ros::Time::now();
        vehicle_->executeCommand(*this, goto_pose, linear_control_val, angular_control_val, has_command);

        // Log
        tf::quaternionMsgToEigen(goto_pose.pose.orientation, orientation_quaternion);
        rotation_matrix = orientation_quaternion.toRotationMatrix();
        euler_angles = rotation_matrix.eulerAngles(0, 1, 2);
        ROS_DEBUG("NAVGOTO: %4.2f, %4.2f, %4.2f, Att: %4.2f, %4.2f, %4.2f, SetAlt: %4.2f",
                  goto_pose.pose.position.x, goto_pose.pose.position.y, goto_pose.pose.position.z,
                  angles::to_degrees(euler_angles[0]), angles::to_degrees(euler_angles[1]), angles::to_degrees(euler_angles[2]),
                  altitude_);

        ros::spinOnce();
        rate.sleep();
    }
}

// Computes XY rotation to the desired point from the "body" frame to the inertial frame
inline geometry_msgs::Quaternion PX4Controller::getRotationTo(geometry_msgs::Point& position, geometry_msgs::Point& target_position) const
{
    Eigen::Vector3d target_direction(target_position.x-position.x, target_position.y-position.y, 0);
    Eigen::Vector3d init_direction(1, 0, 0);
    Eigen::Quaterniond rotation_q = Eigen::Quaterniond::FromTwoVectors(init_direction, target_direction);

    geometry_msgs::Quaternion rotation_msg;
    tf::quaternionEigenToMsg(rotation_q, rotation_msg);

    return rotation_msg;
}

inline geometry_msgs::Point PX4Controller::computeNextWaypoint(
    geometry_msgs::PoseStamped& current_pose,
    float linear_control_val,
    float angular_control_val,
    float linear_speed ) const
{
    // Create movement vector in the "body" frame (it is still in mavros frame of reference)
    Eigen::Vector3d movement_vector(linear_control_val, angular_control_val, 0);
    movement_vector *= linear_speed;

    // Convert movement_vector to the inertial frame based on current pose and compute new position
    Eigen::Quaterniond current_orientation;
    tf::quaternionMsgToEigen(current_pose.pose.orientation, current_orientation);
    Eigen::Matrix3d rotation_mat = current_orientation.toRotationMatrix();
    movement_vector = rotation_mat * movement_vector;

    Eigen::Vector3d current_position;
    tf::pointMsgToEigen(current_pose.pose.position, current_position);
    Eigen::Vector3d new_position = current_position + movement_vector;
    geometry_msgs::Point next_waypoint;
    tf::pointEigenToMsg(new_position, next_waypoint);

    return next_waypoint;
}

inline float PX4Controller::getPoseDistance(geometry_msgs::PoseStamped& pose1, geometry_msgs::PoseStamped& pose2) const
{
    Eigen::Vector3d point1;
    tf::pointMsgToEigen(pose1.pose.position, point1);
    Eigen::Vector3d point2;
    tf::pointMsgToEigen(pose2.pose.position, point2);
    float distance = (point1-point2).norm();

    return distance;
}

inline float PX4Controller::doExpSmoothing(float cur, float prev, float factor) const
{
    return factor * cur + (1.0f - factor) * prev;
}

} // namespace px4_control
