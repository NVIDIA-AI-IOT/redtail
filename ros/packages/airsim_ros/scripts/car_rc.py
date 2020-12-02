#!/usr/bin/env python

from __future__ import print_function

import setup_path
import airsim

import rospy
import tf
import numpy as np
#import ros_numpy
from std_msgs.msg import String
from mavros_msgs.msg import OverrideRCIn

# ROS Image message
from sensor_msgs.msg import Image

class Car_RC():

    def __init__(self):
        airsim_ip = rospy.get_param('~airsim_ip', '127.0.0.1')

        print(airsim_ip)
        self.client = airsim.CarClient(ip = airsim_ip)
        self.client.confirmConnection()

        #Enable api control of car
        self.client.enableApiControl(True)
        self.car_controls = airsim.CarControls()

        self.listener()


    def callback(self, data):  #Get RC commands from Redtail

        rospy.loginfo(rospy.get_caller_id() + 'Throttle pwm %d  Steering pwm %d', data.channels[2], data.channels[0])

        throttle = data.channels[2]
        throttle = (throttle - 1500.0)/500.0
        steering = np.float32(data.channels[0])

        #Send RC commands to AirSim
        self.car_controls.throttle = throttle
        steering = (steering - 1500)/500
        self.car_controls.steering = steering

        self.client.setCarControls(self.car_controls)

        rospy.loginfo(rospy.get_caller_id() + 'Throttle %f  Steering %f', throttle, steering)


    def listener(self): #Listen for messages from Redtail

        rospy.Subscriber('/mavros/rc/override', OverrideRCIn, self.callback)
        rospy.loginfo(rospy.get_caller_id() + 'Listening to /mavros/rc/override')

        rospy.spin()

if __name__ == '__main__':

    rospy.init_node('car_rc', anonymous=True)

    try:
        carRC = Car_RC()

    except rospy.ROSInterruptException:
        pass
