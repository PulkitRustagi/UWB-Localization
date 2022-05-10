#! /usr/bin/env python
# coding=utf-8

import rospy
from gtec_msgs.msg import Ranging
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import MarkerArray

from ukf.datapoint import DataType


class Recorder(object):
    def __init__(self, out="out.csv"):
        self.data = []
        self.out = out

        self.anchor_poses = dict()

        self.tag_offset = {
            1: [0, 0.162, 0.184],
            0: [0, -0.162, 0.184]
        }

        anchors = '/gtec/toa/anchors'
        toa_ranging = '/gtec/toa/ranging'

        self.anchors_sub = rospy.Subscriber(anchors, MarkerArray, callback=self.add_anchors)
        ranging_sub = rospy.Subscriber(toa_ranging, Ranging, callback=self.add_ranging)

        odometry = '/odometry/filtered'
        odometry = rospy.Subscriber(odometry, Odometry, callback=self.create_odometry_callback(DataType.ODOMETRY))

        odometry = '/ground_truth/state'
        odometry = rospy.Subscriber(odometry, Odometry, callback=self.create_odometry_callback(DataType.GROUND_TRUTH))

    def create_odometry_callback(self, id):

        def add_odometry(msg):
            t = self.get_time()

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

            self.data.append((id, px, py, pz, v, theta, theta_yaw, t))

        return add_odometry

    def add_anchors(self, msg):
        # type: (MarkerArray) -> None

        for marker in msg.markers:
            self.anchor_poses[marker.id] = [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z]

    def get_time(self):
        return rospy.Time.now().to_nsec()

    def add_ranging(self, msg):
        # type: (Ranging) -> None
        t = self.get_time()

        if msg.anchorId in self.anchor_poses:
            anchor_pose = self.anchor_poses[msg.anchorId]
            anchor_distance = msg.range / 1000.

            tag = self.tag_offset[msg.tagId]

            self.data.append((DataType.UWB, anchor_distance,
                              anchor_pose[0], anchor_pose[1], anchor_pose[2],
                              tag[0], tag[1], tag[2], t))

    def save(self):
        print("Saving", len(self.data), "datapoints")

        with open(self.out, "w") as file:
            file.writelines(",".join(map(str, d)) + '\n' for d in self.data)


if __name__ == "__main__":
    rospy.init_node("csv_recorder", anonymous=True)

    rec = Recorder()

    rospy.spin()

    rec.save()
