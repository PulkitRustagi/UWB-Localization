#! /usr/bin/env python
# coding=utf-8

import rospy
from geometry_msgs.msg import Twist


class JackalMotion(object):
    """
    Class meant to publish to the Jackal motion commands
    """

    def __init__(self, namespace):
        """
        Setups publisher helper. Publishes to <namespace>jackal_velocity_controller/cmd_vel
        @param namespace: the robot you want to publish to .i.e Jackal1/
        """
        self.v = 0
        self.w = 0

        self.cmd = Twist()

        self.vel_pub = rospy.Publisher(namespace + 'jackal_velocity_controller/cmd_vel', Twist, queue_size=1)

    def step(self):
        """
        Step function that publishes the velocity commands
        """
        self.cmd.linear.x = self.v
        self.cmd.angular.z = self.w

        self.vel_pub.publish(self.cmd)
