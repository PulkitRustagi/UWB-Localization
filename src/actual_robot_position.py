#! /usr/bin/env python
# coding=utf-8
from __future__ import print_function

import sys

import rospy
import tf
from geometry_msgs.msg import Point

if __name__ == "__main__":
    rospy.init_node("robot_position_publisher_node")

    if len(sys.argv) < 3:
        raise ValueError("Expecting arugments: world_link robot_base_link")

    else:
        world_link, robot_base_link = sys.argv[1], sys.argv[2]

    name = "/data_drawer/robot_pose"

    pub = rospy.Publisher(name, Point, queue_size=1)
    tf_listener = tf.TransformListener()

    rate = rospy.Rate(100)

    point = Point()

    while not rospy.is_shutdown():
        try:
            (trans, rot) = tf_listener.lookupTransform(world_link, robot_base_link, rospy.Time(0))

            point.x, point.y = trans[0], trans[1]

            pub.publish(point)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("Unable to find transform")

        rate.sleep()
