#! /usr/bin/env python
# coding=utf-8

import matplotlib

matplotlib.use('Qt5Agg')

import rospy
from nav_msgs.msg import Odometry
from live_plotter import LivePlotter
import sys
from ukf_uwb_localization import get_time


class RSMEPlotter(object):
    """
    Do not use as it is not finished.
    Class is meant to plot the RSME of the robot estimate position in live
    """

    def __init__(self, target, actual):
        """
        Setups the RMSE live plotter
        @param target: the target Odometry topic
        @param actual: the actual Odometry topic
        """
        self.live_plotter = LivePlotter(alpha=0.5, window_name="RMSE Drawer")
        # self.live_plotter.ax.set_aspect("equal")

        self.subscribers = {
            'actual': [],
            'target': []
        }

        self.actual_position_sub = rospy.Subscriber(actual, Odometry, self.add_actual)
        self.target_position_sub = rospy.Subscriber(target, Odometry, self.add_target)

    def get_odometry(self, msg):
        """
        Process Odometery topic data
        @param msg: the Odometry topic message
        @return: A list with the x, y, z, v, theta, theta yaw
        """
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z

        v = msg.twist.twist.linear.x
        theta = euler_from_quaternion((
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ))[2]

        theta_yaw = msg.twist.twist.angular.z

        return [px, py, pz, v, theta, theta_yaw]

    def add_actual(self, msg):
        """
        Add the expected position data
        @param msg: the Odometry topic data
        """
        t = get_time()

        data = self.get_odometry(msg)

        o = [t]
        o.extend(data)

        self.subscribers['actual'].append(o)

    def add_target(self, msg):
        """
        Add the ground truth odometry data
        @param msg: the Odometry topic data
        """
        t = get_time()

        data = self.get_odometry(msg)

        o = [t]
        o.extend(data)

        self.subscribers['target'].append(o)

    def run(self):
        """
        Started live plotter
        """
        self.live_plotter.show()


if __name__ == "__main__":
    rospy.init_node("rsme_drawer_node")

    myargv = rospy.myargv(argv=sys.argv)[1:]

    if len(myargv) != 2:
        raise AttributeError("RSME need two arguments: actual target")

    data_plotter = RSMEPlotter(myargv[0], myargv[1])
    data_plotter.run()
    rospy.spin()
